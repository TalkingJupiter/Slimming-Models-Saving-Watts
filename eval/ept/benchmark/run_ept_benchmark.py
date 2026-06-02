import json
import os
import statistics
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from ept_monitor import EnergyMonitor
from ept_data import load_dolly_prompts


def load_prompts_from_txt(path: str) -> List[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def parse_int_list(raw: str) -> List[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("token list must contain at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError(f"token list values must be positive: {values}")
    return values


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def encode_batch(tokenizer: Any, prompts: List[str], device: str) -> Dict[str, torch.Tensor]:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return {key: value.to(device) for key, value in enc.items()}


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    max_new_tokens: int,
    device: str,
) -> Dict[str, Any]:
    enc = encode_batch(tokenizer, prompts, device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    input_tokens = int(attention_mask.sum().item())

    sync_cuda()
    start = time.perf_counter()
    with torch.no_grad():
        gen_out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    sync_cuda()
    elapsed_s = time.perf_counter() - start

    if isinstance(gen_out, torch.Tensor):
        sequences = gen_out
    else:
        sequences = cast(torch.LongTensor, gen_out.sequences)  # type: ignore[attr-defined]

    output_tokens = 0
    for j in range(sequences.size(0)):
        out_ids = sequences[j]
        out_len = int((out_ids != tokenizer.pad_token_id).sum().item())
        in_len = int((input_ids[j] != tokenizer.pad_token_id).sum().item())
        output_tokens += max(out_len - in_len, 0)

    return {
        "elapsed_s": elapsed_s,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "batch_size": len(prompts),
    }


def load_model_and_tokenizer(
    model_name_or_path: str,
    base_model_name_or_path: Optional[str] = None,
    adapter_name_or_path: Optional[str] = None,
) -> tuple[Any, Any]:
    tokenizer_source = base_model_name_or_path or model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_source = base_model_name_or_path or model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto",
    )
    if adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(model, adapter_name_or_path)
    model.eval()
    return model, tokenizer


def run_one_token_length(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    max_new_tokens: int,
    batch_size: int,
    warmup_batches: int,
    device: str,
    gpu_indices: Optional[List[int]],
    sample_interval: float,
) -> Dict[str, Any]:
    batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
    warmup = batches[: max(0, warmup_batches)]

    for batch_prompts in warmup:
        generate_batch(model, tokenizer, batch_prompts, max_new_tokens, device)

    total_input_tokens = 0
    total_output_tokens = 0
    batch_latencies_s: List[float] = []
    batch_sizes: List[int] = []

    sync_cuda()
    measured_start = time.perf_counter()
    with EnergyMonitor(sample_interval=sample_interval, gpu_indices=gpu_indices) as mon:
        for batch_prompts in batches:
            batch_result = generate_batch(model, tokenizer, batch_prompts, max_new_tokens, device)
            total_input_tokens += int(batch_result["input_tokens"])
            total_output_tokens += int(batch_result["output_tokens"])
            batch_latencies_s.append(float(batch_result["elapsed_s"]))
            batch_sizes.append(int(batch_result["batch_size"]))
    sync_cuda()
    measured_elapsed_s = time.perf_counter() - measured_start

    energy_j = mon.energy_joules
    eps = 1e-9
    total_tokens = total_input_tokens + total_output_tokens

    return {
        "max_new_tokens": max_new_tokens,
        "E_run_J": energy_j,
        "T_in": total_input_tokens,
        "T_out": total_output_tokens,
        "T_total": total_tokens,
        "EPT_in_J_per_tok": energy_j / (total_input_tokens + eps),
        "EPT_out_J_per_tok": energy_j / (total_output_tokens + eps),
        "EPT_total_J_per_tok": energy_j / (total_tokens + eps),
        "measured_elapsed_s": measured_elapsed_s,
        "tokens_per_second_out": total_output_tokens / max(measured_elapsed_s, eps),
        "tokens_per_second_total": total_tokens / max(measured_elapsed_s, eps),
        "mean_batch_latency_s": statistics.mean(batch_latencies_s) if batch_latencies_s else 0.0,
        "p50_batch_latency_s": percentile(batch_latencies_s, 0.50),
        "p90_batch_latency_s": percentile(batch_latencies_s, 0.90),
        "p95_batch_latency_s": percentile(batch_latencies_s, 0.95),
        "max_batch_latency_s": max(batch_latencies_s) if batch_latencies_s else 0.0,
        "num_batches": len(batch_latencies_s),
        "batch_sizes": batch_sizes,
        "warmup_batches": len(warmup),
    }


def run_ept_benchmark(
    model_name_or_path: str,
    prompts: List[str],
    base_model_name_or_path: Optional[str] = None,
    adapter_name_or_path: Optional[str] = None,
    max_new_tokens_list: Optional[List[int]] = None,
    batch_size: int = 4,
    warmup_batches: int = 2,
    device: str = "cuda",
    gpu_indices: Optional[List[int]] = None,
    sample_interval: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    token_lengths = max_new_tokens_list or [64]
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        base_model_name_or_path=base_model_name_or_path,
        adapter_name_or_path=adapter_name_or_path,
    )

    results = []
    for max_new_tokens in token_lengths:
        results.append(
            run_one_token_length(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                warmup_batches=warmup_batches,
                device=device,
                gpu_indices=gpu_indices,
                sample_interval=sample_interval,
            )
        )

    return {
        "metadata": metadata or {},
        "model": model_name_or_path,
        "base_model": base_model_name_or_path,
        "adapter": adapter_name_or_path,
        "num_prompts": len(prompts),
        "batch_size": batch_size,
        "warmup_batches": warmup_batches,
        "sample_interval_s": sample_interval,
        "gpu_indices": gpu_indices,
        "decoding": {
            "do_sample": False,
            "max_new_tokens_list": token_lengths,
        },
        "results": results,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run EPT benchmark for a HF model.")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Base model to load before applying --adapter")
    parser.add_argument("--adapter", type=str, default=None,
                        help="PEFT adapter path to apply on top of --base-model")
    parser.add_argument("--method", type=str, default=None,
                        help="Method label, e.g. feature, relation, response, traditional, base, teacher")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Exact checkpoint path being evaluated")
    parser.add_argument("--prompts", type=str, default=None,
                        help="Path to .txt file with one prompt per line")
    parser.add_argument("--use-dolly", action="store_true",
                        help="Use Dolly dataset prompts instead of a file")
    parser.add_argument("--num-prompts", type=int, default=100,
                        help="Number of prompts to use")
    parser.add_argument("--dolly-cache-dir", type=str,
                        default="~/.cache/huggingface/datasets",
                        help="Cache dir for Dolly")
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="Single generation length; kept for backwards compatibility")
    parser.add_argument("--max-new-tokens-list", type=str, default=None,
                        help="Comma-separated generation lengths, e.g. 32,64,128,256,512,1024")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--gpu-indices", type=str, default=None, help="Comma-separated GPUs to monitor, e.g. '0' or '0,1'")
    parser.add_argument("--out", type=str, default=None, help="Path to write JSON results")

    args = parser.parse_args()

    if args.use_dolly:
        prompts = load_dolly_prompts(
            num_prompts=args.num_prompts,
            seed=42,
            cache_dir=args.dolly_cache_dir,
        )
        prompt_source = "dolly"
    elif args.prompts:
        prompts = load_prompts_from_txt(args.prompts)[: args.num_prompts]
        prompt_source = args.prompts
    else:
        raise ValueError("Must specify either --use-dolly or --prompts")

    if args.gpu_indices:
        gpu_indices = [int(x) for x in args.gpu_indices.split(",")]
    else:
        gpu_indices = None

    if args.max_new_tokens_list:
        max_new_tokens_list = parse_int_list(args.max_new_tokens_list)
    else:
        max_new_tokens_list = [args.max_new_tokens]

    metadata = {
        "method": args.method,
        "checkpoint": args.checkpoint or args.adapter or args.model,
        "prompt_source": prompt_source,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }

    results = run_ept_benchmark(
        model_name_or_path=args.model,
        prompts=prompts,
        base_model_name_or_path=args.base_model,
        adapter_name_or_path=args.adapter,
        max_new_tokens_list=max_new_tokens_list,
        batch_size=args.batch_size,
        warmup_batches=args.warmup_batches,
        gpu_indices=gpu_indices,
        sample_interval=args.sample_interval,
        metadata=metadata,
    )

    print(json.dumps(results, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
