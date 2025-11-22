# SPDX-License-Identifier: MIT
"""
Simple SFT-style training of the student base model on data/shards.jsonl.

- Student: Llama-3.1-8B-Instruct (or any HF causal LM)
- Data: JSONL with field "text" (already prompt+response combined)
- Telemetry: uses monitor.get_gpu_info/get_cpu_info from repo root,
             and only samples during the TRAINING loop (not tokenization).

Run (single GPU):

  python Base/train_base_from_shards.py \
      --model meta-llama/Meta-Llama-3.1-8B-Instruct \
      --data_path data/shards.jsonl \
      --output_dir Base/llama3.1-8B-sft_from_shards \
      --telemetry \
      --telemetry_output telemetry/base_8B_sft/train.jsonl
"""

import os
import sys
import math
import json
import time
import threading
import argparse
import datetime
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)

from datasets import load_dataset
from zoneinfo import ZoneInfo


# -------------------------------------------------------------------
# Import monitor.py from repo root (one level above this file)
# -------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import monitor  # noqa: E402  (after sys.path tweak)


# -------------------------------------------------------------------
# Telemetry sampler using monitor.get_gpu_info/cpu_info
# -------------------------------------------------------------------
class TelemetrySampler:
    """
    Small background thread that writes JSONL telemetry using the
    monitor.get_gpu_info/get_cpu_info helpers already in this repo.

    We start it immediately before the training loop and stop it
    right after training finishes, so tokenization/preprocessing
    are *not* included in the power trace.
    """

    def __init__(self, output_path: str, interval: float = 5.0) -> None:
        # Resolve to absolute path so there's no confusion about cwd
        self.output_path = os.path.abspath(output_path)
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._tz = ZoneInfo("America/Chicago")

    def start(self) -> None:
        out_dir = os.path.dirname(self.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        print(
            f"[Telemetry] Starting sampler; writing to {self.output_path} "
            f"every {self.interval} seconds",
            flush=True,
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        print("[Telemetry] Sampler stopped", flush=True)

    def _run(self) -> None:
        try:
            with open(self.output_path, "a", encoding="utf-8") as f:
                while not self._stop_event.is_set():
                    ts = datetime.datetime.now(self._tz).isoformat()
                    try:
                        entry: Dict[str, Any] = {
                            "timestamp": ts,
                            "gpus": monitor.get_gpu_info(),
                            "cpu": monitor.get_cpu_info(),
                            "phase": "train_base",
                        }
                        f.write(json.dumps(entry) + "\n")
                        f.flush()
                    except Exception as e:
                        # Don't kill the thread; just log the error
                        print(
                            f"[Telemetry] ERROR while sampling: {e}",
                            file=sys.stderr,
                            flush=True,
                        )
                    time.sleep(self.interval)
        except Exception as e:
            # If open() itself fails, you'll see it in the logs
            print(
                f"[Telemetry] FATAL: could not open {self.output_path}: {e}",
                file=sys.stderr,
                flush=True,
            )


# -------------------------------------------------------------------
# Arg parsing
# -------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()

    # Core
    ap.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="HF model id or local path for the student base.",
    )
    ap.add_argument(
        "--data_path",
        type=str,
        default="data/shards.jsonl",
        help="Path to JSONL file with field 'text'.",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="Base/llama3.1-8B-sft_from_shards",
        help="Directory to save checkpoints.",
    )

    # Training knobs (single-GPU)
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument(
        "--max_train_steps",
        type=int,
        default=0,
        help="If >0, overrides num_train_epochs.",
    )
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    # Logging / saving
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=1000)

    # Telemetry
    ap.add_argument(
        "--telemetry",
        action="store_true",
        help="Enable telemetry during training.",
    )
    ap.add_argument(
        "--telemetry_output",
        type=str,
        default="telemetry/base_8B_sft/train_base_telemetry.jsonl",
        help="JSONL file for telemetry samples.",
    )
    ap.add_argument(
        "--telemetry_interval",
        type=float,
        default=5.0,
        help="Seconds between telemetry samples.",
    )

    return ap.parse_args()


# -------------------------------------------------------------------
# Data
# -------------------------------------------------------------------
def make_dataloader(tokenizer, data_path: str, batch_size: int, max_length: int):
    """
    Load JSONL with field 'text' and tokenize.
    We compute causal LM loss on all tokens (labels == input_ids).
    This is *preprocessing*, not covered by telemetry.
    """
    ds = load_dataset("json", data_files={"train": data_path}, split="train")

    def preprocess(batch: Dict[str, Any]) -> Dict[str, Any]:
        texts = batch["text"]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = ds.map(
        preprocess,
        batched=True,
        remove_columns=ds.column_names,
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    return DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    args = parse_args()
    print("==== train_base_from_shards: args ====")
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Model & tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    # ---- Data (tokenization happens here; NOT timed by telemetry) ----
    train_loader = make_dataloader(
        tokenizer=tokenizer,
        data_path=args.data_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    steps_per_epoch = math.ceil(len(train_loader))
    if args.max_train_steps > 0:
        max_steps = args.max_train_steps
        num_epochs = math.ceil(max_steps / steps_per_epoch)
    else:
        num_epochs = args.num_train_epochs
        max_steps = num_epochs * steps_per_epoch

    warmup_steps = int(args.warmup_ratio * max_steps)
    print(f"num_epochs={num_epochs}, max_steps={max_steps}, warmup_steps={warmup_steps}")

    # ---- Optimizer & scheduler ----
    no_decay = ["bias", "LayerNorm.weight"]
    decay_params, nodecay_params = [], []
    for n, p in model.named_parameters():
        (decay_params if not any(nd in n for nd in no_decay) else nodecay_params).append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Telemetry: start AFTER data/optimizer/scheduler are ready ----
    sampler: TelemetrySampler | None = None
    if args.telemetry:
        print(f"[Telemetry] Logging to {args.telemetry_output}")
        sampler = TelemetrySampler(
            output_path=args.telemetry_output,
            interval=args.telemetry_interval,
        )
        sampler.start()

    # ---- Training loop (this is what telemetry covers) ----
    global_step = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            if global_step % args.log_every == 0:
                print(f"Step {global_step}/{max_steps} | loss={loss.item():.4f}")

            if global_step % args.save_every == 0 or global_step == max_steps:
                ckpt_dir = os.path.join(args.output_dir, f"step_{global_step}")
                print(f"Saving checkpoint to {ckpt_dir}")
                os.makedirs(ckpt_dir, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

            if global_step >= max_steps:
                break

        if global_step >= max_steps:
            break

    # ---- Stop telemetry AFTER training ----
    if sampler is not None:
        print("[Telemetry] Stopping sampler")
        sampler.stop()

    # ---- Final save ----
    final_dir = os.path.join(args.output_dir, "final")
    print(f"Saving final model to {final_dir}")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("Training complete.")


if __name__ == "__main__":
    main()
