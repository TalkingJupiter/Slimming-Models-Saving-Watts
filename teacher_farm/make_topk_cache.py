import argparse, os, json, math, sys, time, datetime, threading
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import pyarrow as pa, pyarrow.parquet as pq
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import monitor #noqa: E402

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

class TelemetrySampler:
    def __init__(self, output_path:str, interval: float = 1.0, phase: str = "response_topk_cache") -> None:
        self.output_path = os.path.abspath(output_path)
        self.interval = interval
        self.phase = phase
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tz = ZoneInfo("America/Chicago") if ZoneInfo is not None else None

    def start(self) -> None:
        out_dir = os.path.dirname(self.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
    
    def _run(self) -> None:
        with open(self.output_path, "a", encoding="utf-8") as f:
            while not self._stop_event.is_set():
                now = datetime.datetime.now(self._tz) if self._tz else datetime.datetime.now()
                entry: Dict[str, Any] = {
                    "timestamp": now.isoformat(),
                    "gpus": monitor.get_gpu_info(),
                    "cpu": monitor.get_cpu_info(),
                    "phase": self.phase,
                }
                f.write(json.dumps(entry) + "\n")
                f.flush()
                time.sleep(self.interval)


def batched(iterable, n):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def read_jsonl_shard(path, shard_index: int, num_shards: int):
    record_idx = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if record_idx % num_shards == shard_index:
                yield json.loads(line)
            record_idx += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--input_jsonl', required=True, help='Each line {\"text\": ...}')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--batch_size', type=int, default=1, help='1 recommended for very long sequences')
    ap.add_argument('--max_length', type=int, default=8192)
    ap.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    ap.add_argument('--shard_index', type=int, default=0,
                    help='Process records where record_index %% num_shards == shard_index')
    ap.add_argument('--num_shards', type=int, default=1,
                    help='Total number of dynamic input shards')
    ap.add_argument('--telemetry', action='store_true')
    ap.add_argument('--telemetry_output', type=str, default="results/cache/telemetry/response/telemetry.jsonl")
    ap.add_argument('--telemetry_interval', type=float, default=1.0)
    args = ap.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")

    os.makedirs(args.out_dir, exist_ok=True)

    dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}
    dtype = dtype_map[args.dtype]

    print("[INFO] loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading teacher model {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map='auto')
    model.eval()

    print("[INFO] Loading input dataset...")
    texts = [rec['text'] for rec in read_jsonl_shard(args.input_jsonl, args.shard_index, args.num_shards)]
    print(f"[INFO] Processing dynamic shard {args.shard_index}/{args.num_shards} with {len(texts)} records")
    shard_size = 128

    rows = []
    shard_idx = 0

    sampler: Optional[TelemetrySampler] = None

    if args.telemetry:
        print(f"[Telemetry] Logging to {args.telemetry_output}")
        sampler = TelemetrySampler(
            output_path=args.telemetry_output,
            interval=args.telemetry_interval,
            phase="response_embedding_cache",
        )
        sampler.start()

    try:
        with torch.no_grad():
            for batch in tqdm(batched(texts, args.batch_size), total=math.ceil(len(texts)/args.batch_size)):
                enc = tok(batch, padding=True, truncation=True, max_length=args.max_length, return_tensors='pt')
                enc = {k: v.to(model.device) for k, v in enc.items()}
                out = model(**enc, use_cache=False)
                logits = out.logits ### [B,T,V]
                topk_vals, topk_idx = torch.topk(logits[:, :-1, :], k=args.k, dim=-1)  ## [B, T-1, k]

                input_ids = enc['input_ids'].cpu()
                attn_mask = enc['attention_mask'].cpu()
                logprobs = torch.log_softmax(topk_vals, dim=-1).cpu()

                for b in range(input_ids.size(0)):
                    L = int(attn_mask[b].sum().item())
                    eff = max(L-1, 0)
                    rows.append({
                        'input_ids': input_ids[b, :L].tolist(),
                        'attn_mask': attn_mask[b, :L].tolist(),
                        'topk_ids': topk_idx[b, :eff, :].cpu().tolist(),
                        'topk_logprobs': logprobs[b, :eff, :].tolist()
                    })

                if len(rows) >= shard_size:
                    table = pa.Table.from_pylist(rows)
                    if args.num_shards > 1:
                        filename = f'rb_topk_s{args.shard_index:03d}_{shard_idx:06d}.parquet'
                    else:
                        filename = f'rb_topk_{shard_idx:06d}.parquet'
                    out_path = os.path.join(args.out_dir, filename)
                    pq.write_table(table, out_path, compression='zstd')
                    print('Wrote', out_path)
                    rows, shard_idx = [], shard_idx+1

        if rows:
            table = pa.Table.from_pylist(rows)
            if args.num_shards > 1:
                filename = f'rb_topk_s{args.shard_index:03d}_{shard_idx:06d}.parquet'
            else:
                filename = f'rb_topk_{shard_idx:06d}.parquet'
            out_path = os.path.join(args.out_dir, filename)
            pq.write_table(table, out_path, compression='zstd')
            print('Wrote', out_path)
    finally:
        if sampler is not None:
            print("[Telemetry] Stopping sampler")
            sampler.stop()

if __name__ == "__main__":
    main()
