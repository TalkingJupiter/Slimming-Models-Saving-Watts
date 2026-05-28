import argparse, os, json, math, sys, time, datetime, threading, uuid
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


def read_jsonl_shard(path, shard_index: int, num_shards: int):
    record_idx = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if record_idx % num_shards == shard_index:
                yield json.loads(line)
            record_idx += 1


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def write_parquet_atomic(table: pa.Table, out_path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    final_exists = os.path.exists(out_path)
    temp_path = os.path.join(
        out_dir,
        f".{os.path.basename(out_path)}.{os.getpid()}.{uuid.uuid4().hex}.tmp",
    )
    try:
        pq.write_table(table, temp_path, compression="zstd")
        os.replace(temp_path, out_path)
    except Exception as exc:
        try:
            usage = os.statvfs(out_dir)
            free_bytes = usage.f_bavail * usage.f_frsize
            free_text = _format_bytes(free_bytes)
        except OSError:
            free_text = "unknown"
        try:
            temp_size = os.path.getsize(temp_path)
            temp_size_text = _format_bytes(temp_size)
        except OSError:
            temp_size_text = "not created"
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        raise OSError(
            "failed to write parquet shard "
            f"{out_path!r} via temporary file {temp_path!r}; "
            f"directory free space={free_text}; temp size={temp_size_text}; "
            f"final file already existed={final_exists}"
        ) from exc


def default_hf_cache_dir() -> str:
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    if os.environ.get("HF_HOME"):
        return os.path.join(os.environ["HF_HOME"], "hub")
    return os.path.join(ROOT_DIR, ".hf_cache", "hub")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--input_jsonl', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--max_length', type=int, default=8192)
    ap.add_argument('--shard_index', type=int, default=0,
                    help='Process records where record_index %% num_shards == shard_index')
    ap.add_argument('--num_shards', type=int, default=1,
                    help='Total number of dynamic input shards')
    ap.add_argument('--telemetry', action='store_true')
    ap.add_argument('--telemetry_output', type=str, default="results/cache/telemetry/relation/telemetry.jsonl")
    ap.add_argument('--telemetry_interval', type=float, default=1.0)
    ap.add_argument('--cache_dir', type=str, default=None,
                    help='Hugging Face model cache directory; defaults to HF_HUB_CACHE/HF_HOME or repo-local .hf_cache')
    ap.add_argument('--local_files_only', action='store_true',
                    help='Load only from the local Hugging Face cache; do not download during cache generation')
    args = ap.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.cache_dir is None:
        args.cache_dir = default_hf_cache_dir()
    os.makedirs(args.cache_dir, exist_ok=True)
    print(f"[INFO] Hugging Face cache: {args.cache_dir}")

    
    print("[INFO] loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("[INFO] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map='auto', cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    model.eval()

    print("[INFO] Loading input dataset...")
    texts = [rec['text'] for rec in read_jsonl_shard(args.input_jsonl, args.shard_index, args.num_shards)]
    print(f"[INFO] Processing dynamic shard {args.shard_index}/{args.num_shards} with {len(texts)} records")
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
                    if args.num_shards > 1:
                        filename = f'relb_embeds_s{args.shard_index:03d}_{shard_idx:06d}.parquet'
                    else:
                        filename = f'relb_embeds_{shard_idx:06d}.parquet'
                    out_path = os.path.join(args.out_dir, filename)
                    write_parquet_atomic(table, out_path)
                    print('Wrote', out_path)
                    rows, shard_idx = [], shard_idx + 1

        if rows:
            table = pa.Table.from_pylist(rows)
            if args.num_shards > 1:
                filename = f'relb_embeds_s{args.shard_index:03d}_{shard_idx:06d}.parquet'
            else:
                filename = f'relb_embeds_{shard_idx:06d}.parquet'
            out_path = os.path.join(args.out_dir, filename)
            write_parquet_atomic(table, out_path)
            print('Wrote', out_path)
    finally:
        if sampler is not None:
            print("[Telemetry] Stopping sampler")
            sampler.stop()

if __name__ == '__main__':
    main()


    
