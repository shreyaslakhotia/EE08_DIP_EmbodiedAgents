#!/usr/bin/env python3
"""
Train Qwen2.5-VL-3B with lightweight QLoRA for scenario-aware study-buddy behavior.

Input JSONL schema expected:
{
  "image": "path/to/frame.jpg",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}

Notes:
- Optimized for practical 11GB-class GPUs (batch=1 + grad accumulation + 4-bit base model).
- Handles missing/corrupt image samples by skipping them instead of crashing.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def read_jsonl(path: Path) -> List[dict]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def maybe_split_train_val(records: List[dict], val_ratio: float, seed: int) -> tuple[List[dict], List[dict]]:
    rng = random.Random(seed)
    items = records[:]
    rng.shuffle(items)
    n_val = int(len(items) * val_ratio)
    if len(items) > 1:
        n_val = max(1, n_val)
    else:
        n_val = 0
    return items[n_val:], items[:n_val]


class JsonlVisionDataset(Dataset):
    def __init__(self, records: List[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        image_path = rec.get("image", "")
        messages = rec.get("messages", [])

        user_msg = ""
        assistant_msg = ""
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                user_msg = content
            elif role == "assistant":
                assistant_msg = content

        return {
            "image_path": image_path,
            "user_text": user_msg,
            "assistant_text": assistant_msg,
        }


@dataclass
class VisionCollator:
    processor: AutoProcessor
    max_length: int

    def __call__(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        input_id_list: List[torch.Tensor] = []
        attn_list: List[torch.Tensor] = []
        label_list: List[torch.Tensor] = []
        pixel_values_list: List[torch.Tensor] = []
        image_grid_list: List[torch.Tensor] = []

        for item in batch:
            image_path = item["image_path"]
            user_text = item["user_text"]
            assistant_text = item["assistant_text"]

            # Skip unusable samples silently at collate time by substituting a minimal sample.
            # This keeps long jobs running even if a few files are bad.
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                continue

            user_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_text},
                    ],
                }
            ]
            full_messages = user_messages + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                }
            ]

            prompt_text = self.processor.apply_chat_template(
                user_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = self.processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            enc_full = self.processor(
                text=[full_text],
                images=[image],
                padding=False,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            enc_prompt = self.processor(
                text=[prompt_text],
                images=[image],
                padding=False,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            input_ids = enc_full["input_ids"][0]
            attention_mask = enc_full["attention_mask"][0]
            labels = input_ids.clone()

            prompt_len = int(enc_prompt["input_ids"].shape[1])
            labels[:prompt_len] = -100

            input_id_list.append(input_ids)
            attn_list.append(attention_mask)
            label_list.append(labels)

            if "pixel_values" in enc_full:
                pixel_values_list.append(enc_full["pixel_values"])
            if "image_grid_thw" in enc_full:
                image_grid_list.append(enc_full["image_grid_thw"])

        if not input_id_list:
            # Very rare edge case: all batch samples failed image load.
            # Return one dummy sample to avoid Trainer crash.
            dummy = torch.tensor([[self.processor.tokenizer.eos_token_id]], dtype=torch.long)
            return {
                "input_ids": dummy,
                "attention_mask": torch.ones_like(dummy),
                "labels": torch.full_like(dummy, -100),
            }

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id

        max_seq = max(x.shape[0] for x in input_id_list)

        def pad_1d(x: torch.Tensor, value: int) -> torch.Tensor:
            if x.shape[0] == max_seq:
                return x
            out = torch.full((max_seq,), value, dtype=x.dtype)
            out[: x.shape[0]] = x
            return out

        input_ids = torch.stack([pad_1d(x, pad_id) for x in input_id_list], dim=0)
        attention_mask = torch.stack([pad_1d(x, 0) for x in attn_list], dim=0)
        labels = torch.stack([pad_1d(x, -100) for x in label_list], dim=0)

        batch_out: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if pixel_values_list:
            batch_out["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        if image_grid_list:
            batch_out["image_grid_thw"] = torch.cat(image_grid_list, dim=0)

        return batch_out


def filter_records_with_existing_images(records: List[dict]) -> List[dict]:
    filtered = []
    for rec in records:
        img = rec.get("image")
        if not img:
            continue
        if Path(img).exists():
            filtered.append(rec)
    return filtered


def build_model_and_processor(
    model_name_or_path: str,
    use_4bit: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
) -> tuple[torch.nn.Module, AutoProcessor]:
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForVision2Seq.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

    model = prepare_model_for_kbit_training(model) if use_4bit else model

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, processor


def main() -> int:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Qwen2.5-VL-3B scenario-aware study buddy")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--dataset_path", type=Path, default=None, help="Path to dataset_all.jsonl")
    parser.add_argument("--train_jsonl", type=Path, default=None, help="Optional explicit train split JSONL")
    parser.add_argument("--val_jsonl", type=Path, default=None, help="Optional explicit val split JSONL")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    if args.train_jsonl and args.val_jsonl:
        train_records = read_jsonl(args.train_jsonl)
        val_records = read_jsonl(args.val_jsonl)
    elif args.dataset_path:
        all_records = read_jsonl(args.dataset_path)
        train_records, val_records = maybe_split_train_val(all_records, args.val_ratio, args.seed)
    else:
        print("[ERROR] Provide either --dataset_path or both --train_jsonl and --val_jsonl")
        return 1

    train_records = filter_records_with_existing_images(train_records)
    val_records = filter_records_with_existing_images(val_records)

    if not train_records:
        print("[ERROR] Empty training dataset")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== TRAIN CONFIG ===")
    print(f"model: {args.model_name_or_path}")
    print(f"train samples: {len(train_records)}")
    print(f"val samples: {len(val_records)}")
    print(f"output_dir: {args.output_dir}")
    print(f"epochs: {args.epochs}")
    print(f"lr: {args.learning_rate}")
    print(f"4bit: {args.use_4bit}")

    model, processor = build_model_and_processor(
        model_name_or_path=args.model_name_or_path,
        use_4bit=args.use_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )

    train_ds = JsonlVisionDataset(train_records)
    val_ds = JsonlVisionDataset(val_records)
    collator = VisionCollator(processor=processor, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps" if len(val_ds) > 0 else "no",
        save_strategy="steps",
        fp16=True,
        bf16=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=len(val_ds) > 0,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_first_step=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_ds) > 0 else None,
        data_collator=collator,
    )

    result = trainer.train()

    final_adapter_dir = args.output_dir / "adapter"
    final_adapter_dir.mkdir(parents=True, exist_ok=True)

    trainer.model.save_pretrained(str(final_adapter_dir))
    processor.save_pretrained(str(final_adapter_dir))

    metrics_path = args.output_dir / "train_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(result.metrics, f, indent=2)

    print("\n=== TRAIN COMPLETE ===")
    print(f"adapter dir: {final_adapter_dir}")
    print(f"metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
