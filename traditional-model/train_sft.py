#!/usr/bin/env python3
import argparse
import os
from typing import Any, cast

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset 
from torch.optim import AdamW
from accelerate import Accelerator

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)

from datasets import Dataset as HFDataset  # type: ignore[import]
from data_loader import load_sharded_dataset
from collator import SFTCollator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--shards_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--log_every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    accelerator = Accelerator()
    accelerator.print("Loading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    accelerator.print("Loading dataset shards...")
    dataset_hf: HFDataset = load_sharded_dataset(args.shards_file)  # type: ignore[assignment]

    # Tell Pylance this is "some kind of" torch Dataset; at runtime it's fine.
    torch_dataset: TorchDataset[Any] = cast(TorchDataset[Any], dataset_hf)

    accelerator.print("Building dataloader...")
    collator = SFTCollator(tokenizer=tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        torch_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )



    # ---------- OPTIMIZER ----------
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
    optimizer = AdamW(grouped_params, lr=args.lr)

    steps_per_epoch = max(1, len(dataloader) // args.grad_accum)
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    accelerator.print("Starting training...")
    global_step = 0

    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(dataloader):
            outputs: Any = model(**batch)
            # explicit name so Pylance doesn't confuse it with a function
            loss_tensor: torch.Tensor = outputs.loss / args.grad_accum
            accelerator.backward(loss_tensor)

            if (step + 1) % args.grad_accum == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    loss_val: float = float(loss_tensor.detach().item())
                    accelerator.print(
                        f"[Epoch {epoch}] Step {global_step} | Loss: {loss_val:.4f}"
                    )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_path = os.path.join(args.output_dir, f"epoch_{epoch}")
            os.makedirs(save_path, exist_ok=True)
            model_to_save = accelerator.unwrap_model(model)
            model_to_save.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            accelerator.print(f"Saved checkpoint → {save_path}")

    accelerator.print("Training complete.")


if __name__ == "__main__":
    main()
