"""
01_explore_dataset.py
=====================
Explore the emotion dataset: class distribution, image properties, sample visualization.
Run: python 01_explore_dataset.py
"""

import os
import sys
from pathlib import Path
from collections import Counter
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
import random

DATASET_ROOT = Path("/mlda/shreyas_projects/EE08_DIP_EmbodiedAgents/data/emotion_dataset")
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def count_images(split_dir: Path) -> dict:
    """Count images per emotion class in a split."""
    counts = {}
    for emotion in EMOTIONS:
        emotion_dir = split_dir / emotion
        if emotion_dir.exists():
            files = [f for f in emotion_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
            counts[emotion] = len(files)
        else:
            counts[emotion] = 0
    return counts


def analyze_image_properties(split_dir: Path, sample_n: int = 200) -> dict:
    """Sample images and collect width, height, mode, file size stats."""
    all_files = []
    for emotion in EMOTIONS:
        emotion_dir = split_dir / emotion
        if emotion_dir.exists():
            all_files.extend(list(emotion_dir.glob("*.[jJ][pP][gG]")) +
                             list(emotion_dir.glob("*.[pP][nN][gG]")))

    sample_files = random.sample(all_files, min(sample_n, len(all_files)))
    widths, heights, modes, file_sizes = [], [], [], []

    for f in sample_files:
        try:
            img = Image.open(f)
            widths.append(img.width)
            heights.append(img.height)
            modes.append(img.mode)
            file_sizes.append(f.stat().st_size)
        except Exception as e:
            print(f"  [WARN] Could not open {f}: {e}")

    return {
        "total_sampled": len(widths),
        "width_range": (min(widths), max(widths)) if widths else None,
        "height_range": (min(heights), max(heights)) if heights else None,
        "common_modes": Counter(modes).most_common(),
        "avg_file_size_kb": sum(file_sizes) / len(file_sizes) / 1024 if file_sizes else 0,
        "unique_resolutions": len(set(zip(widths, heights))),
    }


def plot_distribution(train_counts: dict, test_counts: dict, save_path: str):
    """Bar chart of class distribution for train and test splits."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, counts, title in zip(axes, [train_counts, test_counts], ["Train", "Test"]):
        emotions = list(counts.keys())
        values = list(counts.values())
        colors = plt.cm.Set3([i / len(emotions) for i in range(len(emotions))])
        bars = ax.bar(emotions, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(f"{title} Split ({sum(values):,} total)", fontsize=14)
        ax.set_ylabel("Number of Images")
        ax.set_xlabel("Emotion")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    str(val), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Distribution plot saved to {save_path}")
    plt.close()


def plot_samples(split_dir: Path, save_path: str, n_per_class: int = 3):
    """Show n sample images per emotion class."""
    fig, axes = plt.subplots(len(EMOTIONS), n_per_class, figsize=(3 * n_per_class, 3 * len(EMOTIONS)))

    for row, emotion in enumerate(EMOTIONS):
        emotion_dir = split_dir / emotion
        files = sorted(emotion_dir.glob("*.[jJ][pP][gG]"))[:50]
        samples = random.sample(files, min(n_per_class, len(files)))

        for col in range(n_per_class):
            ax = axes[row][col] if len(EMOTIONS) > 1 else axes[col]
            if col < len(samples):
                img = Image.open(samples[col])
                ax.imshow(img, cmap="gray" if img.mode == "L" else None)
                if col == 0:
                    ax.set_ylabel(emotion.upper(), fontsize=12, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Sample Images per Emotion", fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Sample grid saved to {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("EMOTION DATASET EXPLORATION")
    print("=" * 60)

    # --- Class counts ---
    train_dir = DATASET_ROOT / "train"
    test_dir = DATASET_ROOT / "test"

    train_counts = count_images(train_dir)
    test_counts = count_images(test_dir)

    print("\n--- Train Split ---")
    for emotion, count in train_counts.items():
        print(f"  {emotion:>10s}: {count:>5d}")
    print(f"  {'TOTAL':>10s}: {sum(train_counts.values()):>5d}")

    print("\n--- Test Split ---")
    for emotion, count in test_counts.items():
        print(f"  {emotion:>10s}: {count:>5d}")
    print(f"  {'TOTAL':>10s}: {sum(test_counts.values()):>5d}")

    # --- Image properties ---
    print("\n--- Image Properties (sampled from train) ---")
    props = analyze_image_properties(train_dir, sample_n=300)
    for k, v in props.items():
        print(f"  {k}: {v}")

    # --- Plots ---
    out_dir = DATASET_ROOT.parent.parent
    plot_distribution(train_counts, test_counts,
                      str(out_dir / "dataset_class_distribution.png"))
    plot_samples(train_dir, str(out_dir / "dataset_sample_images.png"), n_per_class=4)

    # --- Check for corrupt / unreadable images ---
    print("\n--- Checking for corrupt images (full scan) ---")
    corrupt = []
    for split in ["train", "test"]:
        split_dir = DATASET_ROOT / split
        for emotion in EMOTIONS:
            emotion_dir = split_dir / emotion
            if not emotion_dir.exists():
                continue
            for f in emotion_dir.iterdir():
                try:
                    img = Image.open(f)
                    img.verify()
                except Exception as e:
                    corrupt.append((str(f), str(e)))

    if corrupt:
        print(f"  Found {len(corrupt)} corrupt images:")
        for path, err in corrupt[:10]:
            print(f"    {path}: {err}")
    else:
        print("  All images are readable ✓")

    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
