# Qwen2.5-VL Study Buddy — Fine-Tuning & Deployment Guide

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Results](#results)
- [Quick Start](#quick-start)
- [Pipeline Steps](#pipeline-steps)
- [Pre-trained Weights](#pre-trained-weights)
- [Raspberry Pi Deployment](#raspberry-pi-deployment)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)

---

## Overview

We fine-tuned **Qwen2.5-VL-3B-Instruct** (a 3.8B parameter vision-language model) using **QLoRA** to create an **emotionally aware study buddy robot**. The model:

1. **Detects emotions** from a student's facial image (7 classes: angry, disgust, fear, happy, neutral, sad, surprise)
2. **Generates empathetic conversational responses** as a supportive study buddy

This is an **end-to-end approach** — a single model handles both vision (emotion recognition) and language (empathetic response generation) in one forward pass.

| Spec | Detail |
|------|--------|
| **Base Model** | `Qwen/Qwen2.5-VL-3B-Instruct` (3.83B params) |
| **Fine-tuning Method** | QLoRA (4-bit NF4, rank=32, alpha=64) |
| **Trainable Parameters** | 74.3M / 3.83B (1.94%) |
| **Training Time** | 4 hours 26 minutes |
| **Training Hardware** | 1× NVIDIA RTX 2080 Ti (11GB VRAM) |
| **Dataset** | FER-2013 derivative (48×48 grayscale faces) |
| **Deployment Target** | Raspberry Pi 5 (16GB RAM) via Ollama |
| **Quantized Model Size** | ~3.1 GB (Q4_K_M GGUF + vision encoder) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Qwen2.5-VL-3B-Instruct                   │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Vision Encoder│───▶│  Projector   │───▶│   Language   │  │
│  │  (SigLIP)    │    │  (mmproj)    │    │   Model      │  │
│  │  [frozen]    │    │  [F16 GGUF]  │    │  [LoRA tuned]│  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│        ▲                                        │          │
│   Camera Image                          Empathetic Response│
│   (48×48 → 224×224)                     + Emotion Label    │
└─────────────────────────────────────────────────────────────┘
```

**LoRA targets:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`  
**Dropout:** 0.05 | **Quantization:** 4-bit NF4 with double quantization

---

## Results

### Training Metrics
| Metric | Value |
|--------|-------|
| Final Training Loss | **0.1028** |
| Final Eval Loss | **0.0989** |
| Total Steps | 2,364 (3 epochs) |
| Training Time | 4h 26m |
| No overfitting observed | ✓ (eval_loss ≤ train_loss) |

### Test Accuracy (49-sample stratified evaluation)

| Emotion | Accuracy | Correct/Total |
|---------|----------|---------------|
| **disgust** | 85.7% | 6/7 |
| **surprise** | 85.7% | 6/7 |
| **happy** | 71.4% | 5/7 |
| **neutral** | 71.4% | 5/7 |
| **angry** | 57.1% | 4/7 |
| **sad** | 57.1% | 4/7 |
| **fear** | 14.3% | 1/7 |
| **Overall** | **63.3%** | **31/49** |

> **Note:** FER-2013 is a notoriously noisy dataset with ~65% human agreement. Our model's 63.3% accuracy is competitive with the dataset's own inter-annotator agreement. The `fear` class performs poorly because fear expressions are commonly confused with `angry` and `surprise` — this is consistent with human labeling disagreements on the FER-2013 dataset.

### Confusion Patterns
- **Fear** → most confused with angry (3) and surprise (2) — these share similar facial features (wide eyes, tension)
- **Angry** → sometimes confused with neutral (2) — subtle anger in low-res 48×48 images
- **Sad** → occasionally confused with neutral (2) — also subtle in small images

### Sample Model Outputs
```
[angry face] → "I notice this student looks angry — there's real tension there. 
Look, I completely understand. There are few things more infuriating than pouring 
effort into something and feeling like it's not paying off..."

[disgust face] → "I notice the student looks disgusted — they're clearly not 
enjoying this material. You know what? Some of the best learners I know have felt 
exactly the way you're feeling right now about certain topics..."

[happy face] → "I see a happy student! That smile tells me something is going 
right in your study session..."
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with ≥11GB VRAM (for training/export)
- CUDA 12.x

### Setup
```bash
cd /mlda/shreyas_projects/EE08_DIP_EmbodiedAgents
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_finetune.txt
```

### Run the Full Pipeline
```bash
# Step 1: Explore dataset
python 01_explore_dataset.py

# Step 2: Prepare training data
python 02_prepare_finetune_data.py

# Step 3: Fine-tune (4+ hours on RTX 2080 Ti)
python 03_finetune_qwen3vl_lora.py --gpu 0

# Step 4: Export to GGUF for Raspberry Pi
python 04_export_model.py --gpu 0

# Step 5: Evaluate accuracy
python 05_evaluate_test_accuracy.py --gpu 0 --max_samples 50
```

---

## Pipeline Steps

### Step 1: Explore Dataset (`01_explore_dataset.py`)
Analyzes the raw FER-2013 emotion dataset:
- Counts images per class in train/test splits
- Samples image properties (resolution, color mode, file size)
- Generates distribution bar charts (`dataset_class_distribution.png`)
- Generates sample image grids (`dataset_sample_images.png`)
- Scans for corrupt/unreadable images

**Key findings:**
- Train: 6,436 images | Test: 700 images
- All images: 48×48 grayscale (mode "L")
- `disgust` class severely underrepresented: 436 train images vs ~1000 for others
- Zero corrupt images found

### Step 2: Prepare Fine-Tuning Data (`02_prepare_finetune_data.py`)
Transforms raw images into VLM-ready conversational training data:

1. **Image preprocessing:** 48×48 grayscale → 224×224 RGB JPEG (Lanczos upscaling)
2. **Class balancing:** Oversamples `disgust` from 436 → ~1000 using augmentation (brightness ±20%, contrast ±20%, Gaussian blur, horizontal flip)
3. **Conversational data generation:** Each image paired with:
   - 1 of 12 diverse user prompts (e.g., "How does this student seem to be feeling?")
   - 1 of 3-4 empathetic study buddy responses per emotion (~150+ words each)
4. **Data splits:** Train (6,300) / Validation (700) / Test (700)
5. **Output format:** JSONL with HuggingFace messages format

```json
{"messages": [
  {"role": "user", "content": [
    {"type": "image", "image": "/path/to/image.jpg"},
    {"type": "text", "text": "How is this student feeling right now?"}
  ]},
  {"role": "assistant", "content": [
    {"type": "text", "text": "I notice this student looks angry — there's real tension..."}
  ]}
]}
```

### Step 3: Fine-Tune with LoRA (`03_finetune_qwen3vl_lora.py`)
QLoRA fine-tuning using Unsloth for 2× memory efficiency:

```bash
# Default settings (recommended)
python 03_finetune_qwen3vl_lora.py --gpu 0

# Custom settings
python 03_finetune_qwen3vl_lora.py --gpu 1 --epochs 5 --lr 5e-5

# Resume from checkpoint
python 03_finetune_qwen3vl_lora.py --resume
```

| Parameter | Value |
|-----------|-------|
| Base model | `unsloth/Qwen2.5-VL-3B-Instruct` |
| Quantization | 4-bit NF4 (double quant) |
| LoRA rank / alpha | 32 / 64 |
| LoRA targets | q, k, v, o, gate, up, down proj |
| Learning rate | 2e-4 (cosine schedule) |
| Warmup | 10% of steps |
| Batch size | 1 (×8 gradient accumulation = effective 8) |
| Precision | FP16 |
| Max sequence length | 2048 tokens |
| Epochs | 3 |

**Important known issues & fixes (already in the scripts):**
- `sys.setrecursionlimit(5000)` — fixes bitsandbytes + torch.compile recursion bug
- `TORCH_COMPILE_DISABLE=1` and `UNSLOTH_COMPILE_DISABLE=1` — prevents compilation errors on older GPUs (Compute Capability 7.5)
- `import unsloth` must be the **first import** before any other ML library

**Outputs:**
- `checkpoints/qwen3vl_studybuddy/final/` — LoRA adapter weights (~295 MB)
- `checkpoints/qwen3vl_studybuddy/checkpoint-*/` — intermediate checkpoints
- `checkpoints/qwen3vl_studybuddy/training_metrics.json`
- `checkpoints/qwen3vl_studybuddy/eval_samples.json`

### Step 4: Export for Raspberry Pi (`04_export_model.py`)
Merges LoRA adapter with base model and quantizes to GGUF:

```bash
# Full pipeline: merge + GGUF export
python 04_export_model.py --gpu 0

# Skip merge (if already done), only GGUF export
python 04_export_model.py --gpu 0 --gguf_only
```

**Process:**
1. **Merge LoRA:** Loads 4-bit model + LoRA adapter → merges to 16-bit → saves (~7 GB)
2. **GGUF Export:** Uses Unsloth's `save_pretrained_gguf()` with Q4_K_M quantization
   - Requires llama.cpp (auto-installed to `~/.unsloth/llama.cpp`, or manually built — see [Troubleshooting](#troubleshooting))
   - Produces: quantized language model (1.8 GB) + F16 vision encoder/mmproj (1.3 GB)
3. **Ollama Setup:** Generates `Modelfile` and `run_studybuddy.py` inference script

**Outputs:**
```
checkpoints/qwen3vl_studybuddy/
├── merged/                          # Full 16-bit merged model (~7 GB)
└── export/
    ├── Modelfile                    # Ollama model config
    ├── run_studybuddy.py            # Camera inference script for Pi
    └── studybuddy_gguf/
        ├── qwen2.5-vl-3b-instruct.Q4_K_M.gguf      # 1.8 GB — quantized LLM
        └── qwen2.5-vl-3b-instruct.F16-mmproj.gguf   # 1.3 GB — vision encoder
```

### Step 5: Evaluate Test Accuracy (`05_evaluate_test_accuracy.py`)
Runs inference on test set and computes classification metrics:

```bash
# Quick evaluation (50 samples, ~10 minutes)
python 05_evaluate_test_accuracy.py --gpu 0 --max_samples 50

# Full evaluation (all 700 samples, ~2-3 hours)
python 05_evaluate_test_accuracy.py --gpu 0
```

Uses keyword-based emotion extraction from generated text. Reports per-class accuracy, confusion matrix, precision/recall/F1.

---

## Pre-trained Weights

The fine-tuned GGUF model weights for Raspberry Pi deployment are available on OneDrive:

📥 **[Download Weights (OneDrive — NTU)](https://entuedu-my.sharepoint.com/my?id=%2Fpersonal%2Fshreyas010%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FEE3180&ga=1)**

**GGUF files for deployment (~3.1 GB total):**
- `studybuddy_gguf/qwen2.5-vl-3b-instruct.Q4_K_M.gguf` (1.8 GB) — quantized language model
- `studybuddy_gguf/qwen2.5-vl-3b-instruct.F16-mmproj.gguf` (1.3 GB) — vision encoder

After downloading, place the `studybuddy_gguf/` folder alongside the `Modelfile` in your deployment directory (see [Raspberry Pi Deployment](#raspberry-pi-deployment)).

**To load the model for inference (on a GPU machine):**
```python
import unsloth
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    "checkpoints/qwen3vl_studybuddy/final/",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastVisionModel.for_inference(model)
```

> The base model (`Qwen/Qwen2.5-VL-3B-Instruct`, ~6 GB) is downloaded automatically from HuggingFace on first run.

---

## Raspberry Pi Deployment

### What to Copy to the Pi
From `checkpoints/qwen3vl_studybuddy/export/`, copy:
- `studybuddy_gguf/` folder (contains both `.gguf` files, ~3.1 GB total)
- `Modelfile`
- `run_studybuddy.py`

```bash
scp -r checkpoints/qwen3vl_studybuddy/export/ pi@<PI_IP>:~/studybuddy/
```

### Option A: Ollama (Recommended)

```bash
# 1. Install Ollama on Pi
curl -fsSL https://ollama.com/install.sh | sh

# 2. Create the model (from the export directory)
cd ~/studybuddy
ollama create studybuddy -f Modelfile

# 3. Test with text
ollama run studybuddy

# 4. Run with camera
pip install ollama opencv-python pillow
python run_studybuddy.py
```

### Option B: llama.cpp Direct (if Ollama VLM support is limited)

```bash
# Build llama.cpp on Pi (ARM64)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build -j4

# Run with image
./build/bin/llama-mtmd-cli \
  -m ~/studybuddy/studybuddy_gguf/qwen2.5-vl-3b-instruct.Q4_K_M.gguf \
  --mmproj ~/studybuddy/studybuddy_gguf/qwen2.5-vl-3b-instruct.F16-mmproj.gguf \
  -p "Look at this student. How are they feeling? Respond as a caring study buddy."
```

### Expected Performance on Pi 5 (16GB)
| Metric | Estimate |
|--------|----------|
| Model load time | ~30-60 seconds |
| Inference speed | ~2-5 tokens/second |
| Memory usage | ~3-4 GB RAM |
| First-token latency | ~5-15 seconds (with image) |

---

## File Structure

```
EE08_DIP_EmbodiedAgents/
├── 01_explore_dataset.py           # Step 1: Dataset analysis & visualization
├── 02_prepare_finetune_data.py     # Step 2: Preprocess images + build JSONL
├── 03_finetune_qwen3vl_lora.py     # Step 3: QLoRA fine-tuning with Unsloth
├── 04_export_model.py              # Step 4: Merge + GGUF export + Ollama setup
├── 05_evaluate_test_accuracy.py    # Step 5: Test set evaluation
├── requirements_finetune.txt       # Python dependencies for fine-tuning
├── FINETUNE_README.md              # ← This documentation
├── README.md                       # Project overview
├── training_log.txt                # Raw training console output
├── dataset_class_distribution.png  # Class distribution bar charts
├── dataset_sample_images.png       # Sample images grid
├── data/
│   ├── emotion_dataset/            # Raw FER-2013 dataset (48×48 grayscale)
│   │   ├── train/{7 emotion folders}/
│   │   └── test/{7 emotion folders}/
│   ├── emotion_processed/          # [gitignored] Preprocessed 224×224 RGB (regenerate with Step 2)
│   └── finetune_data/              # Generated JSONL training data
│       ├── train.jsonl (6,300 samples)
│       ├── val.jsonl   (700 samples)
│       ├── test.jsonl  (700 samples)
│       └── examples_preview.json
└── checkpoints/                    # [gitignored] Download from OneDrive
    └── qwen3vl_studybuddy/
        ├── final/                  # LoRA adapter (~295 MB)
        ├── merged/                 # Full 16-bit model (~7 GB)
        ├── training_metrics.json
        ├── test_evaluation_results.json
        └── export/
            ├── Modelfile
            ├── run_studybuddy.py
            └── studybuddy_gguf/
                ├── qwen2.5-vl-3b-instruct.Q4_K_M.gguf      (1.8 GB)
                └── qwen2.5-vl-3b-instruct.F16-mmproj.gguf   (1.3 GB)
```

---

## Troubleshooting

### `RecursionError` during training
This is caused by bitsandbytes + torch.compile interaction. The scripts already include fixes, but if running custom code:
```python
import sys; sys.setrecursionlimit(5000)
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
```

### `Skipping model.language_model.layers.X.mlp.*: no quant_state found`
This is a **harmless warning**. Some layers weren't quantized during 4-bit loading. Doesn't affect training or inference quality.

### GGUF export fails with "You do not have internet connection!"
This is a false positive from Unsloth — it runs `apt-get update` to check internet, which fails on shared servers without sudo. **Fix:** Manually build llama.cpp:
```bash
mkdir -p ~/.unsloth
git clone https://github.com/ggml-org/llama.cpp ~/.unsloth/llama.cpp
cd ~/.unsloth/llama.cpp
cmake -B build -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-quantize -j$(nproc)
cp build/bin/llama-quantize ./llama-quantize
```

### Out of memory during training
- Reduce `per_device_train_batch_size` to 1 (default)
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Use `--gpu` flag to select a GPU with more free VRAM
- Close other processes using GPU memory

### Slow inference on Raspberry Pi
- Ensure you're using the Q4_K_M quantized model (1.8 GB), not the F16 version
- Close other applications to free RAM
- Consider Q3_K_M quantization for smaller size (slight quality loss)

---

## Future Improvements

1. **Better dataset:** FER-2013 is noisy (~65% human agreement). Using AffectNet or RAF-DB would improve accuracy significantly.
2. **Higher resolution images:** Current 48×48 → 224×224 upscaling loses quality. Native high-res face images would help.
3. **Two-stage approach:** Lightweight CNN (MobileNet) for emotion classification + text-only LLM for response — potentially faster on Pi.
4. **Data augmentation:** More sophisticated augmentation (face alignment, lighting normalization) could help, especially for `fear`.
5. **Full test evaluation:** Run on all 700 test samples for more statistically significant results.
6. **Real-time camera integration:** Optimize the inference loop for continuous camera feed on the Pi.
7. **Speech integration:** Add Whisper STT for voice input alongside facial emotion detection.
