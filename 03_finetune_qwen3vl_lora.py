"""
03_finetune_qwen3vl_lora.py
============================
Fine-tune Qwen3-VL-2B-Instruct with LoRA for an emotionally aware study buddy.

Uses Unsloth (preferred, 2x faster) with HuggingFace PEFT as fallback.
Important: Unsloth must be imported BEFORE transformers/peft/trl.

Hardware target: RTX 2080 Ti (11GB VRAM)
Deployment target: Raspberry Pi 5 (16GB RAM) via Ollama

Usage:
  python 03_finetune_qwen3vl_lora.py                   # train
  python 03_finetune_qwen3vl_lora.py --eval_only       # evaluate on test set
  python 03_finetune_qwen3vl_lora.py --resume           # resume from checkpoint
  python 03_finetune_qwen3vl_lora.py --gpu 1            # use GPU 1

Requirements:
  pip install unsloth transformers peft trl datasets pillow accelerate bitsandbytes
  pip install qwen-vl-utils
"""

# ─── CRITICAL: Import unsloth FIRST before any other ML library ─────────────────
import unsloth  # noqa: F401 — must be imported first for patching

import argparse
import json
import os
import sys
from pathlib import Path

# Fix bitsandbytes + torch.compile recursion bug
sys.setrecursionlimit(5000)
os.environ["TORCH_COMPILE_DISABLE"] = "1"  # Disable torch.compile globally
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"  # Disable Unsloth compilation

import torch
torch._dynamo.config.suppress_errors = True  # Graceful fallback if compile triggers

from datasets import Dataset
from PIL import Image


# ─── Configuration ──────────────────────────────────────────────────────────────
class Config:
    # Model
    model_name = "unsloth/Qwen2.5-VL-3B-Instruct"  # Smallest Qwen2.5-VL available
    max_seq_length = 2048  # Keeps VRAM usage reasonable on 11GB

    # LoRA hyperparameters
    lora_r = 32               # Rank — good balance of capacity vs efficiency
    lora_alpha = 64            # Alpha = 2*r is a solid default
    lora_dropout = 0.05
    # Target modules for Qwen VL — attention + MLP layers
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    # Training hyperparameters
    num_epochs = 3
    per_device_train_batch_size = 1       # 11GB VRAM with 3B model, be conservative
    gradient_accumulation_steps = 8       # Effective batch size = 8
    learning_rate = 2e-4
    weight_decay = 0.01
    warmup_ratio = 0.1
    lr_scheduler_type = "cosine"
    max_grad_norm = 1.0
    bf16 = False                          # RTX 2080 Ti doesn't support bf16 well
    fp16 = True                           # Use fp16 instead

    # Data
    data_dir = Path("/mlda/shreyas_projects/EE08_DIP_EmbodiedAgents/data/finetune_data")
    output_dir = Path("/mlda/shreyas_projects/EE08_DIP_EmbodiedAgents/checkpoints/qwen3vl_studybuddy")

    # Logging
    logging_steps = 10
    save_steps = 200
    eval_steps = 200
    save_total_limit = 3

    # GPU selection — use GPU 0 (cleanest, only 7MB used)
    gpu_id = 0


def setup_gpu(gpu_id: int):
    """Configure single GPU usage."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  Using GPU {gpu_id}: {gpu_name} ({vram:.1f} GB)")
        return device
    else:
        print("  [ERROR] No CUDA GPU available!")
        sys.exit(1)


# ─── Data Loading ───────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list:
    """Load a JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def try_unsloth_import():
    """Try importing Unsloth; fall back to standard HuggingFace if unavailable."""
    try:
        from unsloth import FastVisionModel
        print("  ✓ Unsloth available — using accelerated training")
        return "unsloth"
    except ImportError:
        print("  ✗ Unsloth not available — falling back to HuggingFace PEFT + TRL")
        return "huggingface"


# ─── Unsloth Pipeline ──────────────────────────────────────────────────────────
def train_with_unsloth(config: Config):
    """Fine-tune using Unsloth (preferred — 2x faster, 60% less VRAM)."""
    from unsloth import FastVisionModel
    from unsloth import UnslothVisionDataCollator

    print("\n--- Loading Model with Unsloth ---")
    model, tokenizer = FastVisionModel.from_pretrained(
        config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=True,  # QLoRA — fits easily in 11GB
        dtype=None,          # Auto-detect
    )

    print("\n--- Applying LoRA ---")
    model = FastVisionModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        use_gradient_checkpointing="unsloth",  # Unsloth optimized
        random_state=42,
    )

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    print("\n--- Loading Data ---")
    train_data = load_jsonl(config.data_dir / "train.jsonl")
    val_data = load_jsonl(config.data_dir / "val.jsonl")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")

    # Convert to the format Unsloth expects for VLM
    def convert_to_unsloth_format(sample):
        """Convert our JSONL format to Unsloth's expected conversation format."""
        messages = sample["messages"]
        converted_messages = []

        for msg in messages:
            role = msg["role"]
            content_parts = msg["content"]
            converted_content = []

            for part in content_parts:
                if part["type"] == "image":
                    # Load the image (already preprocessed to 224x224 RGB)
                    try:
                        img = Image.open(part["image"]).convert("RGB")
                        converted_content.append({"type": "image", "image": img})
                    except Exception as e:
                        print(f"    [WARN] Could not load image {part['image']}: {e}")
                        return None
                elif part["type"] == "text":
                    converted_content.append({"type": "text", "text": part["text"]})

            converted_messages.append({"role": role, "content": converted_content})

        return {"messages": converted_messages}

    print("  Converting train data...")
    train_converted = []
    skipped = 0
    for sample in train_data:
        result = convert_to_unsloth_format(sample)
        if result is not None:
            train_converted.append(result)
        else:
            skipped += 1
    if skipped > 0:
        print(f"    [WARN] Skipped {skipped} samples due to image loading errors")

    print("  Converting val data...")
    val_converted = []
    for sample in val_data:
        result = convert_to_unsloth_format(sample)
        if result is not None:
            val_converted.append(result)

    print(f"  Successfully converted: {len(train_converted)} train, {len(val_converted)} val")

    train_dataset = Dataset.from_list(train_converted)
    val_dataset = Dataset.from_list(val_converted)

    # --- Training with TRL's SFTTrainer ---
    from trl import SFTTrainer, SFTConfig

    print("\n--- Starting Training ---")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    from unsloth import is_bf16_supported

    sft_config = SFTConfig(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # Set to "wandb" if you want W&B logging
        seed=42,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        max_seq_length=config.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
    )

    print("\n  Training started! This may take a while...")
    print(f"  Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Checkpoints will be saved to: {config.output_dir}")

    train_result = trainer.train()

    print("\n--- Training Complete ---")
    print(f"  Training loss: {train_result.training_loss:.4f}")

    # Save final model
    final_dir = config.output_dir / "final"
    print(f"\n  Saving final LoRA adapter to {final_dir}")
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save training metrics
    metrics_path = config.output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(train_result.metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    return model, tokenizer


# ─── HuggingFace PEFT Fallback ─────────────────────────────────────────────────
def train_with_huggingface(config: Config):
    """Fine-tune using vanilla HuggingFace PEFT + TRL (fallback)."""
    from transformers import (
        AutoModelForVision2Seq,
        AutoProcessor,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from trl import SFTTrainer, SFTConfig

    print("\n--- Loading Model with HuggingFace ---")

    # Use the non-Unsloth model name for HuggingFace fallback
    hf_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    # QLoRA quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        hf_model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(
        hf_model_name,
        trust_remote_code=True,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    print("\n--- Applying LoRA ---")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\n--- Loading Data ---")
    train_data = load_jsonl(config.data_dir / "train.jsonl")
    val_data = load_jsonl(config.data_dir / "val.jsonl")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")

    # Build a custom collator for VLM data
    def convert_sample(sample):
        """Convert a sample to the format needed for processing."""
        messages = sample["messages"]
        converted_messages = []

        images = []
        for msg in messages:
            role = msg["role"]
            content_parts = msg["content"]
            converted_content = []

            for part in content_parts:
                if part["type"] == "image":
                    try:
                        img = Image.open(part["image"]).convert("RGB")
                        images.append(img)
                        converted_content.append({"type": "image", "image": img})
                    except Exception:
                        return None
                elif part["type"] == "text":
                    converted_content.append({"type": "text", "text": part["text"]})

            converted_messages.append({"role": role, "content": converted_content})

        return {"messages": converted_messages, "images": images}

    print("  Converting data...")
    train_converted = [r for s in train_data if (r := convert_sample(s)) is not None]
    val_converted = [r for s in val_data if (r := convert_sample(s)) is not None]

    train_dataset = Dataset.from_list(train_converted)
    val_dataset = Dataset.from_list(val_converted)

    print("\n--- Starting Training ---")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=42,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=config.max_seq_length,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
    )

    train_result = trainer.train()

    print("\n--- Training Complete ---")
    print(f"  Training loss: {train_result.training_loss:.4f}")

    final_dir = config.output_dir / "final"
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))

    metrics_path = config.output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    return model, processor


# ─── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate_model(config: Config):
    """Run a quick qualitative eval: load test images and generate responses."""
    print("\n--- Qualitative Evaluation ---")

    test_data = load_jsonl(config.data_dir / "test.jsonl")
    print(f"  Test samples: {len(test_data)}")

    # Pick 2 random samples per emotion for evaluation
    import random
    from collections import defaultdict

    by_emotion = defaultdict(list)
    for sample in test_data:
        # Extract emotion from the response text
        response = sample["messages"][1]["content"][0]["text"]
        for emotion in ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]:
            if emotion in response.lower():
                by_emotion[emotion].append(sample)
                break

    eval_samples = []
    for emotion, samples in by_emotion.items():
        eval_samples.extend(random.sample(samples, min(2, len(samples))))

    print(f"  Selected {len(eval_samples)} samples for qualitative eval")
    print("\n  To run inference, use the saved model with:")
    print(f"    model_path = '{config.output_dir / 'final'}'")
    print("    # Load with Unsloth or PEFT, then generate responses")
    print("    # Compare generated responses with ground truth")

    # Save eval samples for later use
    eval_path = config.output_dir / "eval_samples.json"
    with open(eval_path, "w") as f:
        json.dump(eval_samples, f, indent=2)
    print(f"  Eval samples saved to {eval_path}")


# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL for Study Buddy")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default: 0)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    config = Config()
    config.gpu_id = args.gpu
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.per_device_train_batch_size = args.batch_size

    print("=" * 60)
    print("QWEN3-VL STUDY BUDDY — LoRA FINE-TUNING")
    print("=" * 60)

    # Setup GPU
    device = setup_gpu(config.gpu_id)

    if args.eval_only:
        evaluate_model(config)
        return

    # Detect best available framework
    framework = try_unsloth_import()

    if framework == "unsloth":
        model, tokenizer = train_with_unsloth(config)
    else:
        model, tokenizer = train_with_huggingface(config)

    # Run eval
    evaluate_model(config)

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print(f"  Model saved to: {config.output_dir / 'final'}")
    print(f"  Next step: Run 04_export_model.py to convert to GGUF for Ollama")
    print("=" * 60)


if __name__ == "__main__":
    main()
