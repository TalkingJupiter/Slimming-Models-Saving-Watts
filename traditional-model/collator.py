import torch
from dataclasses import dataclass
from typing import List, Dict, Any

from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class SFTCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 2048

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts: List[str] = []
        for ex in batch:
            if "text" in ex:
                texts.append(ex["text"])
            else:
                instr = ex.get("instruction", "")
                output = ex.get("output", "")
                texts.append(instr + "\n" + output)

        enc = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        # Explicitly tell Pylance this is a Tensor
        input_ids: torch.Tensor = enc["input_ids"]  # type: ignore[assignment]
        attention_mask: torch.Tensor = enc["attention_mask"]  # type: ignore[assignment]

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # overwrite with tensors to keep types consistent
        enc["input_ids"] = input_ids
        enc["attention_mask"] = attention_mask
        enc["labels"] = labels

        # ensure return type is Dict[str, Tensor]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
