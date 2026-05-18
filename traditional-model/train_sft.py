#!/usr/bin/env python3
"""Traditional SFT training on JSONL shards.

This is the maintained replacement for Base/train_base_from_shards.py. It keeps
the Accelerate-based training path and adds the old Base script's useful
features: max-step training, periodic/final checkpoints, and training-scoped
telemetry.
"""

import argparse
import datetime
import json
import math
import os
import sys
import threading
import time
from typing import Any, Dict, Optional, cast

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset as TorchDataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from datasets import Dataset as HFDataset  # type: ignore[import]
from data_loader import load_sharded_dataset
from collator import SFTCollator

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import monitor  # noqa: E402

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]


class TelemetrySampler:
    def __init__(self, output_path: str, interval: float = 1.0) -> None:
        self.output_path = os.path.abspath(output_path)
        self.interval = interval
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
                    "phase": "traditional_sft",
                }
                f.write(json.dumps(entry) + "\n")
                f.flush()
                time.sleep(self.interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "--model", dest="model_name", type=str, required=True)
    parser.add_argument(
        "--shards_file",
        "--data_path",
        dest="shards_file",
        type=str,
        required=True,
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", "--learning_rate", dest="lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", "--num_train_epochs", dest="num_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--telemetry", action="store_true")
    parser.add_argument(
        "--telemetry_output",
        type=str,
        default="traditional-model/telemetry/train_sft.jsonl",
    )
    parser.add_argument("--telemetry_interval", type=float, default=1.0)
    return parser.parse_args()


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: str,
    name: str,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    save_path = os.path.join(output_dir, name)
    os.makedirs(save_path, exist_ok=True)
    model_to_save = accelerator.unwrap_model(model)
    model_to_save.save_pretrained(
        save_path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    tokenizer.save_pretrained(save_path)
    accelerator.print(f"Saved checkpoint to {save_path}")


def main() -> None:
    args = parse_args()

    accelerator = Accelerator()
    accelerator.print("==== train_sft: args ====")
    accelerator.print(args)
    accelerator.print("Loading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    accelerator.print("Loading dataset shards...")
    dataset_hf: HFDataset = load_sharded_dataset(args.shards_file)  # type: ignore[assignment]
    torch_dataset: TorchDataset[Any] = cast(TorchDataset[Any], dataset_hf)

    accelerator.print("Building dataloader...")
    collator = SFTCollator(tokenizer=tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        torch_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(grouped_params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    steps_per_epoch = max(1, math.ceil(len(dataloader) / args.grad_accum))
    if args.max_train_steps > 0:
        total_steps = args.max_train_steps
        num_epochs = max(1, math.ceil(total_steps / steps_per_epoch))
    else:
        num_epochs = args.num_epochs
        total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    accelerator.print(
        f"num_epochs={num_epochs}, total_steps={total_steps}, warmup_steps={warmup_steps}"
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    sampler: Optional[TelemetrySampler] = None
    if args.telemetry and accelerator.is_main_process:
        accelerator.print(f"[Telemetry] Logging to {args.telemetry_output}")
        sampler = TelemetrySampler(args.telemetry_output, args.telemetry_interval)
        sampler.start()

    accelerator.print("Starting training...")
    global_step = 0
    optimizer.zero_grad()

    try:
        for epoch in range(num_epochs):
            model.train()
            for step, batch in enumerate(dataloader):
                outputs: Any = model(**batch)
                raw_loss: torch.Tensor = outputs.loss
                loss = raw_loss / args.grad_accum
                accelerator.backward(loss)

                is_accum_step = (step + 1) % args.grad_accum == 0
                is_last_step = step + 1 == len(dataloader)
                if not (is_accum_step or is_last_step):
                    continue

                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0 or global_step == 1:
                    accelerator.print(
                        f"[Epoch {epoch}] Step {global_step}/{total_steps} | "
                        f"Loss: {float(raw_loss.detach().item()):.4f}"
                    )

                if args.save_every > 0 and global_step % args.save_every == 0:
                    save_checkpoint(
                        accelerator,
                        model,
                        tokenizer,
                        args.output_dir,
                        f"step_{global_step}",
                    )

                if global_step >= total_steps:
                    break

            if global_step >= total_steps:
                break
    finally:
        if sampler is not None:
            accelerator.print("[Telemetry] Stopping sampler")
            sampler.stop()

    save_checkpoint(accelerator, model, tokenizer, args.output_dir, "final")
    accelerator.print("Training complete.")


if __name__ == "__main__":
    main()
