import json
from pathlib import Path
from typing import List, Dict, Optional, Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ept_monitor import EnergyMonitor
from ept_data import load_dolly_prompts


def load_prompts_from_txt(path: str) -> List[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def get_model_input_device(model) -> torch.device:
    """
    Find the device where inputs should be placed.
    For sharded HF models loaded with device_map='auto',
    this is usually the first CUDA device in hf_device_map.
    """
    if hasattr(model, "hf_device_map"):
        hf_device_map = getattr(model, "hf_device_map")
        cuda_devices = []

        for _, dev in hf_device_map.items():
            if isinstance(dev, str) and dev.startswith("cuda"):
                cuda_devices.append(dev)
            elif isinstance(dev, int):
                cuda_devices.append(f"cuda:{dev}")

        if cuda_devices:
            cuda_devices = sorted(set(cuda_devices), key=lambda x: int(x.split(":")[1]))
            return torch.device(cuda_devices[0])

    return next(model.parameters()).device


def sync_all_visible_gpus():
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.synchronize(i)
        except Exception:
            pass


def run_ept_benchmark(
    model_name_or_path: str,
    prompts: List[str],
    max_new_tokens: int = 64,
    batch_size: int = 4,
    gpu_indices: Optional[List[int]] = None,
) -> Dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    input_device = get_model_input_device(model)

    total_input_tokens = 0
    total_output_tokens = 0

    with EnergyMonitor(sample_interval=0.1, gpu_indices=gpu_indices) as mon:
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]

            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            input_ids = enc["input_ids"].to(input_device)
            attention_mask = enc["attention_mask"].to(input_device)

            total_input_tokens += int(attention_mask.sum().item())

            with torch.no_grad():
                gen_out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            if isinstance(gen_out, torch.Tensor):
                sequences = gen_out
            else:
                sequences = cast(torch.LongTensor, gen_out.sequences)  # type: ignore[attr-defined]

            batch_size_actual = sequences.size(0)
            for j in range(batch_size_actual):
                out_ids = sequences[j]
                out_len = int((out_ids != tokenizer.pad_token_id).sum().item())
                in_len = int((input_ids[j] != tokenizer.pad_token_id).sum().item())
                total_output_tokens += max(out_len - in_len, 0)

        sync_all_visible_gpus()

    E_run_J = mon.energy_joules
    eps = 1e-9

    return {
        "E_run_J": E_run_J,
        "T_in": total_input_tokens,
        "T_out": total_output_tokens,
        "EPT_in_J_per_tok": E_run_J / (total_input_tokens + eps),
        "EPT_out_J_per_tok": E_run_J / (total_output_tokens + eps),
        "EPT_total_J_per_tok": E_run_J / (total_input_tokens + total_output_tokens + eps),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run EPT benchmark for a HF model.")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--prompts", type=str, default=None, help="Path to .txt file with one prompt per line")
    parser.add_argument("--use-dolly", action="store_true", help="Use Dolly dataset prompts instead of a file")
    parser.add_argument("--num-prompts", type=int, default=100, help="Number of Dolly prompts to use")
    parser.add_argument(
        "--dolly-cache-dir",
        type=str,
        default="~/.cache/huggingface/datasets",
        help="Cache dir for Dolly",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--gpu-indices",
        type=str,
        default=None,
        help="Comma-separated GPUs to monitor, e.g. '0' or '0,1,2,3'",
    )
    parser.add_argument("--out", type=str, default=None, help="Path to write JSON results")

    args = parser.parse_args()

    if args.use_dolly:
        prompts = load_dolly_prompts(
            num_prompts=args.num_prompts,
            seed=42,
            cache_dir=args.dolly_cache_dir,
        )
    elif args.prompts:
        prompts = load_prompts_from_txt(args.prompts)
    else:
        raise ValueError("Must specify either --use-dolly or --prompts")

    gpu_indices = None
    if args.gpu_indices:
        gpu_indices = [int(x.strip()) for x in args.gpu_indices.split(",") if x.strip()]

    results = run_ept_benchmark(
        model_name_or_path=args.model,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        gpu_indices=gpu_indices,
    )

    print(json.dumps(results, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()