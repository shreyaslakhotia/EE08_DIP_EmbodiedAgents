#!/usr/bin/env python3
"""
Inference smoke test for Qwen2.5-VL-3B + LoRA adapter.

Loads a few samples from JSONL and prints generated outputs clearly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from transformers import AutoModelForVision2Seq as AutoVisionModel
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoVisionModel


def read_jsonl(path: Path, max_samples: int) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= max_samples:
                break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inference smoke test on fine-tuned Qwen2.5-VL LoRA adapter")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter_path", type=Path, required=True)
    parser.add_argument("--dataset_jsonl", type=Path, required=True)
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--use_4bit", action="store_true")
    args = parser.parse_args()

    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    base_model = AutoVisionModel.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, str(args.adapter_path))
    model.eval()

    samples = read_jsonl(args.dataset_jsonl, args.max_samples)
    if not samples:
        print("[ERROR] No samples found in dataset_jsonl")
        return 1

    print("\n=== INFERENCE TEST START ===")
    print(f"base model: {args.base_model}")
    print(f"adapter: {args.adapter_path}")
    print(f"samples: {len(samples)}")

    for i, rec in enumerate(samples, start=1):
        image_path = rec.get("image", "")
        messages = rec.get("messages", [])

        user_text = ""
        expected = ""
        for m in messages:
            if m.get("role") == "user":
                user_text = m.get("content", "")
            if m.get("role") == "assistant":
                expected = m.get("content", "")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            print(f"\n[{i}] SKIP bad image: {image_path} :: {exc}")
            continue

        chat_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        prompt = processor.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
            )

        input_len = inputs["input_ids"].shape[1]
        gen_tokens = generated[:, input_len:]
        output_text = processor.batch_decode(gen_tokens, skip_special_tokens=True)[0].strip()

        print("\n" + "=" * 90)
        print(f"Sample {i}")
        print(f"Image: {image_path}")
        print("- User prompt:")
        print(user_text[:1000])
        print("- Model output:")
        print(output_text)
        print("- Expected style (reference target):")
        print(expected[:500])

    print("\n=== INFERENCE TEST COMPLETE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
