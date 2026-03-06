"""
05_evaluate_test_accuracy.py
=============================
Evaluate the fine-tuned Qwen2.5-VL LoRA model on the test set.

Computes:
  - Per-class accuracy
  - Overall accuracy
  - Confusion matrix
  - Classification report (precision / recall / F1)

Usage:
  python 05_evaluate_test_accuracy.py                       # full 700-sample test set
  python 05_evaluate_test_accuracy.py --max_samples 50      # quick sanity check
  python 05_evaluate_test_accuracy.py --gpu 1               # use a different GPU

Requirements:
  Same as 03_finetune_qwen3vl_lora.py
"""

# ─── CRITICAL: Import unsloth FIRST ─────────────────────────────────────────────
import unsloth  # noqa: F401

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.setrecursionlimit(5000)
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

import torch
torch._dynamo.config.suppress_errors = True

from PIL import Image

# ─── Configuration ──────────────────────────────────────────────────────────────
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

MODEL_NAME = "unsloth/Qwen2.5-VL-3B-Instruct"
ADAPTER_PATH = Path("checkpoints/qwen3vl_studybuddy/final")
TEST_JSONL = Path("data/finetune_data/test.jsonl")
RESULTS_DIR = Path("checkpoints/qwen3vl_studybuddy")

# Emotion keywords that appear in the model's responses
EMOTION_KEYWORDS = {
    "angry":    ["angry", "anger", "frustrated", "frustrat", "furious", "irritat", "mad"],
    "disgust":  ["disgust", "aversion", "repuls", "revolted", "displeasure"],
    "fear":     ["fear", "afraid", "anxious", "anxiety", "worried", "nervous", "scared", "uneasy"],
    "happy":    ["happy", "joy", "cheerful", "delighted", "excited", "smiling", "upbeat", "positive energy", "great mood", "wonderful"],
    "neutral":  ["neutral", "calm", "collected", "composed", "steady", "balanced", "relaxed"],
    "sad":      ["sad", "sadness", "down", "upset", "melanchol", "heavy", "gloomy", "low spirits", "sorrowful"],
    "surprise": ["surprise", "surprised", "shock", "astonish", "startled", "taken aback", "wide-eyed", "unexpected"],
}


def extract_ground_truth(sample: dict) -> str:
    """Extract the ground truth emotion from the image path."""
    image_path = sample["messages"][0]["content"][0]["image"]
    for emotion in EMOTIONS:
        if f"/{emotion}/" in image_path:
            return emotion
    return "unknown"


def extract_predicted_emotion(generated_text: str) -> str:
    """
    Extract the predicted emotion from the model's generated text.
    Uses keyword matching — checks which emotion has the most keyword hits
    in the first ~200 characters (where the emotion identification usually is).
    """
    text_lower = generated_text.lower()
    # Focus on the first part of the response where emotion is usually stated
    text_start = text_lower[:300]

    scores = {}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_start)
        if score > 0:
            scores[emotion] = score

    if not scores:
        # Fallback: search the entire text
        for emotion, keywords in EMOTION_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[emotion] = score

    if scores:
        return max(scores, key=scores.get)

    return "unknown"


def load_model(gpu_id: int = 0):
    """Load the fine-tuned model with LoRA adapter."""
    print(f"\n📦 Loading base model: {MODEL_NAME}")
    print(f"📎 Loading LoRA adapter: {ADAPTER_PATH}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device} (GPU {gpu_id})")

    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=str(ADAPTER_PATH),
        load_in_4bit=True,
        max_seq_length=2048,
    )
    FastVisionModel.for_inference(model)

    print("✅ Model loaded successfully")
    return model, tokenizer


def run_inference(model, tokenizer, sample: dict) -> str:
    """Run inference on a single sample and return the generated text."""
    # Build the messages (with PIL image)
    user_content = sample["messages"][0]["content"]
    image_path = user_content[0]["image"]
    text_prompt = user_content[1]["text"]

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    from qwen_vl_utils import process_vision_info

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = tokenizer(
        text=[input_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,      # Low temp for consistent classification
            do_sample=False,      # Greedy decoding for reproducibility
            top_p=1.0,
        )

    # Decode only the generated tokens (skip the input)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text


def evaluate(model, tokenizer, test_data: list, max_samples: int = None):
    """Run evaluation on the test set."""
    if max_samples and max_samples < len(test_data):
        # Stratified sampling
        import random
        random.seed(42)
        by_emotion = defaultdict(list)
        for s in test_data:
            gt = extract_ground_truth(s)
            by_emotion[gt].append(s)

        per_class = max(1, max_samples // len(EMOTIONS))
        selected = []
        for emotion in EMOTIONS:
            samples = by_emotion.get(emotion, [])
            selected.extend(random.sample(samples, min(per_class, len(samples))))
        test_data = selected[:max_samples]
        print(f"📊 Using {len(test_data)} stratified samples (from {max_samples} requested)")
    else:
        print(f"📊 Evaluating on {len(test_data)} test samples")

    # Results tracking
    results = []
    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))  # confusion[true][pred]

    start_time = time.time()

    for i, sample in enumerate(test_data):
        gt_emotion = extract_ground_truth(sample)
        if gt_emotion == "unknown":
            continue

        try:
            generated_text = run_inference(model, tokenizer, sample)
            pred_emotion = extract_predicted_emotion(generated_text)

            is_correct = (pred_emotion == gt_emotion)
            correct += int(is_correct)
            total += 1
            per_class_correct[gt_emotion] += int(is_correct)
            per_class_total[gt_emotion] += 1
            confusion[gt_emotion][pred_emotion] += 1

            results.append({
                "image": sample["messages"][0]["content"][0]["image"],
                "ground_truth": gt_emotion,
                "predicted": pred_emotion,
                "correct": is_correct,
                "generated_text": generated_text[:200],  # Truncate for storage
            })

            # Progress
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (len(test_data) - i - 1)
            status = "✅" if is_correct else "❌"
            print(
                f"  [{i+1:4d}/{len(test_data)}] {status} "
                f"GT={gt_emotion:8s} | Pred={pred_emotion:8s} | "
                f"Acc={correct/total:.1%} | ETA={eta/60:.1f}min"
            )

        except Exception as e:
            print(f"  [{i+1:4d}/{len(test_data)}] ⚠️  Error: {e}")
            per_class_total[gt_emotion] += 1
            total += 1
            confusion[gt_emotion]["error"] += 1
            results.append({
                "image": sample["messages"][0]["content"][0]["image"],
                "ground_truth": gt_emotion,
                "predicted": "error",
                "correct": False,
                "error": str(e),
            })

    elapsed = time.time() - start_time
    return results, correct, total, per_class_correct, per_class_total, confusion, elapsed


def print_report(correct, total, per_class_correct, per_class_total, confusion, elapsed):
    """Print a nicely formatted evaluation report."""
    print("\n" + "=" * 70)
    print("📋 EVALUATION REPORT")
    print("=" * 70)

    overall_acc = correct / total if total > 0 else 0
    print(f"\n  Overall Accuracy: {correct}/{total} = {overall_acc:.2%}")
    print(f"  Evaluation Time:  {elapsed/60:.1f} minutes ({elapsed/total:.1f}s per sample)")

    # Per-class accuracy
    print(f"\n  {'Emotion':<12} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("  " + "-" * 42)
    for emotion in EMOTIONS:
        c = per_class_correct.get(emotion, 0)
        t = per_class_total.get(emotion, 0)
        acc = c / t if t > 0 else 0
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {emotion:<12} {c:>8} {t:>8} {acc:>9.1%}  {bar}")

    # Confusion matrix
    print(f"\n  Confusion Matrix (rows=true, cols=predicted):")
    header = f"  {'':12s}" + "".join(f"{e[:5]:>7s}" for e in EMOTIONS) + f"{'error':>7s}"
    print(header)
    print("  " + "-" * (12 + 7 * (len(EMOTIONS) + 1)))
    for true_emotion in EMOTIONS:
        row = f"  {true_emotion:<12s}"
        for pred_emotion in EMOTIONS:
            count = confusion.get(true_emotion, {}).get(pred_emotion, 0)
            row += f"{count:>7d}"
        error_count = confusion.get(true_emotion, {}).get("error", 0)
        row += f"{error_count:>7d}"
        print(row)

    # Precision / Recall / F1
    print(f"\n  {'Emotion':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "-" * 44)
    for emotion in EMOTIONS:
        # Recall = TP / (TP + FN)
        tp = confusion.get(emotion, {}).get(emotion, 0)
        total_true = per_class_total.get(emotion, 0)
        recall = tp / total_true if total_true > 0 else 0

        # Precision = TP / (TP + FP)
        total_pred = sum(confusion.get(e, {}).get(emotion, 0) for e in EMOTIONS)
        precision = tp / total_pred if total_pred > 0 else 0

        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  {emotion:<12} {precision:>9.1%} {recall:>9.1%} {f1:>9.1%}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen2.5-VL on test set")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default: 0)")
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max test samples to evaluate (default: all 700). Use 50 for a quick check."
    )
    parser.add_argument(
        "--adapter_path", type=str, default=None,
        help="Path to the LoRA adapter (default: checkpoints/qwen3vl_studybuddy/final)"
    )
    args = parser.parse_args()

    global ADAPTER_PATH
    if args.adapter_path:
        ADAPTER_PATH = Path(args.adapter_path)

    print("=" * 70)
    print("🧪 QWEN2.5-VL STUDY BUDDY — TEST SET EVALUATION")
    print("=" * 70)

    # Load test data
    print(f"\n📂 Loading test data from {TEST_JSONL}")
    test_data = []
    with open(TEST_JSONL) as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    print(f"   Loaded {len(test_data)} test samples")

    # Show class distribution
    gt_dist = defaultdict(int)
    for s in test_data:
        gt_dist[extract_ground_truth(s)] += 1
    for e in EMOTIONS:
        print(f"   {e}: {gt_dist.get(e, 0)}")

    # Load model
    model, tokenizer = load_model(gpu_id=args.gpu)

    # Run evaluation
    results, correct, total, per_class_correct, per_class_total, confusion, elapsed = evaluate(
        model, tokenizer, test_data, max_samples=args.max_samples
    )

    # Print report
    print_report(correct, total, per_class_correct, per_class_total, confusion, elapsed)

    # Save results
    results_path = RESULTS_DIR / "test_evaluation_results.json"
    summary = {
        "overall_accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "elapsed_seconds": elapsed,
        "per_class": {
            e: {
                "correct": per_class_correct.get(e, 0),
                "total": per_class_total.get(e, 0),
                "accuracy": per_class_correct.get(e, 0) / per_class_total.get(e, 0)
                if per_class_total.get(e, 0) > 0 else 0,
            }
            for e in EMOTIONS
        },
        "confusion_matrix": {
            true_e: dict(confusion.get(true_e, {})) for true_e in EMOTIONS
        },
        "sample_results": results[:20],  # Save first 20 for inspection
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n💾 Results saved to {results_path}")

    # Save full per-sample results
    full_results_path = RESULTS_DIR / "test_evaluation_full.json"
    with open(full_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Full per-sample results saved to {full_results_path}")

    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
