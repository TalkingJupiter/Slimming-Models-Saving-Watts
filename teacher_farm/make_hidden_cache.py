import argparse, os, json, math, gc, sys, time, datetime, threading
from typing import List, Dict, Any, Optional
import numpy as np
import torch
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
    def __init__(self, output_path:str, interval: float = 1.0, phase: str = "feature_hidden_cache") -> None:
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

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


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
    ap.add_argument('--input_jsonl', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--layers', type=int, nargs='+', required=True,
                    help='Teacher layers to save, e.g. 22 30 (Llama blocks index)')
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--max_length', type=int, default=2048)
    ap.add_argument('--dtype', default='bfloat16', choices=['bfloat16','bf16','float16','fp16','float32','fp32'])
    ap.add_argument('--flush_every', type=int, default=256,
                    help='Write to parquet every N samples to avoid high RAM')
    ap.add_argument('--shard_index', type=int, default=0,
                    help='Process records where record_index %% num_shards == shard_index')
    ap.add_argument('--num_shards', type=int, default=1,
                    help='Total number of dynamic input shards')
    ap.add_argument('--telemetry', action='store_true')
    ap.add_argument('--telemetry_output', type=str, default="results/cache/telemetry/feature/telemetry.jsonl")
    ap.add_argument('--telemetry_interval', type=float, default=1.0)
    args = ap.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")

    os.makedirs(args.out_dir, exist_ok=True)

    # --- dtype parsing
    dmap = {
        'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16,
        'float16': torch.float16,   'fp16': torch.float16,
        'float32': torch.float32,   'fp32': torch.float32,
    }
    torch_dtype = dmap[args.dtype]

    # --- tokenizer
    print("[INFO] loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- load teacher across 4 GPUs
    # Keep inputs on CPU; HF will dispatch internally with device_map='auto'
    print("[INFO] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map='auto',
    )
    model.eval()
    # Avoid KV-cache to save memory
    if hasattr(model, "config"):
        model.config.use_cache = False

    # --- forward hooks for ONLY requested layers
    target_layers: List[int] = sorted(set(args.layers))
    captured: Dict[int, torch.Tensor] = {}

    def make_hook(li: int):
        def hook(_m, _inp, out):
            # out: (B,T,H) on some CUDA device; move to CPU immediately
            hidden = out[0] if isinstance(out, tuple) else out
            captured[li] = hidden.detach().cpu()
        return hook

    # Llama-style: model.model.layers[li]
    handles = []
    for li in target_layers:
        layer_mod = model.model.layers[li]
        handles.append(layer_mod.register_forward_hook(make_hook(li)))

    # --- streaming read
    print("[INFO] Loading input dataset...")
    num_texts = sum(1 for _ in read_jsonl_shard(args.input_jsonl, args.shard_index, args.num_shards))
    texts = (rec['text'] for rec in read_jsonl_shard(args.input_jsonl, args.shard_index, args.num_shards))
    print(f"[INFO] Processing dynamic shard {args.shard_index}/{args.num_shards} with {num_texts} records")

    shard_idx = 0
    rows: List[Dict[str, Any]] = []

    def flush_rows():
        nonlocal rows, shard_idx
        if not rows:
            return
        table = pa.Table.from_pylist(rows)
        if args.num_shards > 1:
            filename = f'fb_hints_s{args.shard_index:03d}_{shard_idx:06d}.parquet'
        else:
            filename = f'fb_hints_{shard_idx:06d}.parquet'
        out_path = os.path.join(args.out_dir, filename)
        pq.write_table(table, out_path, compression='zstd')
        print('Wrote', out_path, f'({len(rows)} samples)')
        shard_idx += 1
        rows = []
        gc.collect()

    sampler: Optional[TelemetrySampler] = None

    if args.telemetry:
        print(f"[Telemetry] Logging to {args.telemetry_output}")
        sampler = TelemetrySampler(
            output_path=args.telemetry_output,
            interval=args.telemetry_interval,
            phase="feature_embedding_cache",
        )
        sampler.start()
    try: 
        with torch.inference_mode():
            for batch in tqdm(batched(texts, args.batch_size), total=math.ceil(num_texts / args.batch_size)):
                # Tokenize on CPU; keep tensors on CPU (device_map dispatch handles placement)
                enc = tok(list(batch), padding=True, truncation=True,
                        max_length=args.max_length, return_tensors='pt')
                # Run forward WITHOUT output_hidden_states big tuple; hooks will capture our layers
                captured.clear()
                _ = model(**enc, output_hidden_states=False, use_cache=False, return_dict=True)

                input_ids = enc['input_ids'].cpu()
                attn_mask = enc['attention_mask'].cpu()

                # Build rows from captured CPU tensors
                for b in range(input_ids.size(0)):
                    L = int(attn_mask[b].sum().item())
                    row = {
                        'input_ids': input_ids[b, :L].tolist(),
                        'attn_mask': attn_mask[b, :L].tolist(),
                    }
                    for li in target_layers:
                        ht = captured[li][b, :L, :]  # (L, H) on CPU
                        # NOTE: Parquet likes lists; we keep float16/bfloat16 as float32 to be safe
                        row[f'hidden_L{li}'] = ht.to(torch.float32).tolist()
                    rows.append(row)

                del enc, input_ids, attn_mask
                captured.clear()
                torch.cuda.empty_cache()

                if len(rows) >= args.flush_every:
                    flush_rows()

        # final flush
        flush_rows()

    finally:
        for h in handles:
            h.remove()

        if sampler is not None:
            print("[Telemetry] Stopping sampler")
            sampler.stop()

if __name__ == '__main__':
    main()
