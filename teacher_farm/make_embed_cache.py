import argparse, os, json, math, sys, time, datetime, threading
from typing import Any, Dict, Optional
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
    def __init__(self, output_path:str, interval: float = 1.0, phase: str = "relation_embedding_cache") -> None:
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--input_jsonl', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--max_length', type=int, default=8192)
    ap.add_argument('--telemetry', action='store_true')
    ap.add_argument('--telemetry_output', type=str, default="results/cache/telemetry/relation/telemetry.jsonl")
    ap.add_argument('--telemetry_interval', type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    print("[INFO] loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("[INFO] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map='auto')
    model.eval()

    print("[INFO] Loading input dataset...")
    texts = [json.loads(l)['text'] for l in open(args.input_jsonl) if l.strip()]
    shard_size = 128
    rows, shard_idx = [], 0

    sampler: Optional[TelemetrySampler] = None

    if args.telemetry:
        print(f"[Telemetry] Logging to {args.telemetry_output}")
        sampler = TelemetrySampler(
            output_path=args.telemetry_output,
            interval=args.telemetry_interval,
            phase="relation_embedding_cache",
        )
        sampler.start()
    try:
        with torch.no_grad():
            for batch in tqdm(batched(texts, args.batch_size), total=math.ceil(len(texts)/args.batch_size)):
                enc = tok(batch, padding=True, truncation=True, max_length=args.max_length, return_tensors='pt')
                enc = {k: v.to(model.device) for k, v in enc.items()}
                out = model(**enc, output_hidden_states=True, use_cache=False)

                last_hidden = out.hidden_states[-1]  # [B, T, d]
                attn = enc['attention_mask']
                input_ids = enc['input_ids']

                mask = attn.unsqueeze(-1)  # [B, T, 1]
                summed = (last_hidden * mask).sum(dim=1)  # [B, d]
                counts = mask.sum(dim=1).clamp_min(1)     # [B, 1]
                pooled = (summed / counts).cpu()          # [B, d]

                for b in range(pooled.size(0)):
                    L = int(attn[b].sum().item())
                    rows.append({
                        'input_ids': input_ids[b, :L].cpu().tolist(),
                        'attn_mask': attn[b, :L].cpu().tolist(),
                        'pooled_embedding': pooled[b].tolist()
                    })

                if len(rows) >= shard_size:
                    table = pa.Table.from_pylist(rows)
                    out_path = os.path.join(args.out_dir, f'relb_embeds_{shard_idx:06d}.parquet')
                    pq.write_table(table, out_path, compression='zstd')
                    print('Wrote', out_path)
                    rows, shard_idx = [], shard_idx + 1

        if rows:
            table = pa.Table.from_pylist(rows)
            out_path = os.path.join(args.out_dir, f'relb_embeds_{shard_idx:06d}.parquet')
            pq.write_table(table, out_path, compression='zstd')
            print('Wrote', out_path)
    finally:
        if sampler is not None:
            print("[Telemetry] Stopping sampler")
            sampler.stop()

if __name__ == '__main__':
    main()


    