"""
04_export_model.py
==================
Export the fine-tuned LoRA model for deployment on Raspberry Pi 5 via Ollama.

Steps:
  1. Merge LoRA adapter with base model (16-bit HuggingFace format)
  2. Export to GGUF Q4_K_M via Unsloth (no llama.cpp needed)
  3. Generate Ollama Modelfile + Pi inference script

Usage:
  python 04_export_model.py                     # full pipeline
  python 04_export_model.py --gguf_only         # skip merge, only GGUF + Modelfile
  python 04_export_model.py --gpu 1             # use a different GPU
"""

# ─── CRITICAL: Import unsloth FIRST ─────────────────────────────────────────────
import unsloth  # noqa: F401

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.setrecursionlimit(5000)
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

import torch
torch._dynamo.config.suppress_errors = True


# ─── Configuration ──────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path("/mlda/shreyas_projects/EE08_DIP_EmbodiedAgents/checkpoints/qwen3vl_studybuddy/final")
MERGED_DIR = Path("/mlda/shreyas_projects/EE08_DIP_EmbodiedAgents/checkpoints/qwen3vl_studybuddy/merged")
EXPORT_DIR = Path("/mlda/shreyas_projects/EE08_DIP_EmbodiedAgents/checkpoints/qwen3vl_studybuddy/export")
BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def step1_merge_lora():
    """Merge LoRA adapter with base model and save as 16-bit HuggingFace format."""
    print("\n" + "=" * 60)
    print("STEP 1: Merge LoRA adapter with base model")
    print("=" * 60)

    from unsloth import FastVisionModel
    print(f"  Loading adapter from: {CHECKPOINT_DIR}")

    model, tokenizer = FastVisionModel.from_pretrained(
        str(CHECKPOINT_DIR),
        max_seq_length=2048,
        load_in_4bit=True,
    )

    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Merging and saving to {MERGED_DIR}...")
    model.save_pretrained_merged(
        str(MERGED_DIR),
        tokenizer,
        save_method="merged_16bit",
    )

    merged_size = sum(f.stat().st_size for f in MERGED_DIR.rglob("*") if f.is_file()) / 1024**3
    print(f"  ✅ Merged model saved ({merged_size:.2f} GB)")


def step2_convert_to_gguf():
    """Convert the LoRA adapter to GGUF Q4_K_M using Unsloth's built-in export."""
    print("\n" + "=" * 60)
    print("STEP 2: Export to GGUF (Q4_K_M quantization)")
    print("=" * 60)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    from unsloth import FastVisionModel
    print(f"  Loading adapter from: {CHECKPOINT_DIR}")

    model, tokenizer = FastVisionModel.from_pretrained(
        str(CHECKPOINT_DIR),
        max_seq_length=2048,
        load_in_4bit=True,
    )

    output_path = str(EXPORT_DIR / "studybuddy")
    print(f"  Exporting to GGUF Q4_K_M → {output_path}")
    print("  This may take 10-20 minutes...")

    model.save_pretrained_gguf(
        output_path,
        tokenizer,
        quantization_method="q4_k_m",
    )

    # Report sizes — Unsloth puts files in a _gguf subdirectory
    gguf_files = list(EXPORT_DIR.rglob("*.gguf"))
    if gguf_files:
        for gf in gguf_files:
            size_gb = gf.stat().st_size / 1024**3
            print(f"  ✅ {gf.name}: {size_gb:.2f} GB")
    else:
        print("  ⚠️  No .gguf files found — check output above for errors")


def step3_create_ollama_modelfile():
    """Generate an Ollama Modelfile for easy deployment."""
    print("\n" + "=" * 60)
    print("STEP 3: Create Ollama Modelfile")
    print("=" * 60)

    gguf_dir = EXPORT_DIR / "studybuddy_gguf"

    # Find the actual GGUF files Unsloth generated
    gguf_model = None
    gguf_mmproj = None
    if gguf_dir.exists():
        for f in gguf_dir.glob("*.gguf"):
            if "mmproj" in f.name:
                gguf_mmproj = f.name
            else:
                gguf_model = f.name

    model_from = f"./studybuddy_gguf/{gguf_model}" if gguf_model else "./studybuddy_gguf/studybuddy-q4_k_m.gguf"

    modelfile_content = f"""# Ollama Modelfile for Study Buddy (Emotion-Aware)
# Based on Qwen2.5-VL-3B-Instruct, fine-tuned with LoRA
# NOTE: Update paths below if you move the files

FROM {model_from}
"""
    # Add multimodal projector if present (required for VLM image input)
    if gguf_mmproj:
        modelfile_content += f"""
# Vision encoder (multimodal projector) — required for image input
PROJECTOR ./studybuddy_gguf/{gguf_mmproj}
"""

    modelfile_content += """
# System prompt for Study Buddy persona
SYSTEM \"\"\"You are a caring and emotionally intelligent study buddy. When you see a student's face,
you can detect their emotional state and respond with empathy and support. You help students
feel understood while encouraging them to keep learning. You are warm, patient, and supportive.
Always acknowledge the student's emotions first, then offer helpful study-related guidance.\"\"\"

# Model parameters optimized for Raspberry Pi 5
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 2048
PARAMETER repeat_penalty 1.1

# Template matching Qwen2.5-VL chat format
TEMPLATE \"\"\"{{{{- range .Messages }}}}
{{{{- if eq .Role "system" }}}}<|im_start|>system
{{{{ .Content }}}}<|im_end|>
{{{{- else if eq .Role "user" }}}}<|im_start|>user
{{{{ .Content }}}}<|im_end|>
{{{{- else if eq .Role "assistant" }}}}<|im_start|>assistant
{{{{ .Content }}}}<|im_end|>
{{{{- end }}}}
{{{{- end }}}}<|im_start|>assistant
\"\"\"
"""

    modelfile_path = EXPORT_DIR / "Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    print(f"  ✓ Modelfile saved to {modelfile_path}")
    print("\n  To deploy on Raspberry Pi:")
    print(f"  1. Copy {EXPORT_DIR}/ to your Raspberry Pi")
    print(f"  2. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh")
    print(f"  3. Create the model: ollama create studybuddy -f Modelfile")
    print(f"  4. Run: ollama run studybuddy")

    # Also create a simple inference script for Pi
    inference_script = EXPORT_DIR / "run_studybuddy.py"
    with open(inference_script, "w") as f:
        f.write('''"""
Simple inference script for Study Buddy on Raspberry Pi.
Uses Ollama API for inference with camera input.

Requirements:
  pip install ollama opencv-python pillow
  ollama create studybuddy -f Modelfile
"""

import ollama
import cv2
import base64
import time
from io import BytesIO
from PIL import Image


def capture_frame(camera_id: int = 0) -> str:
    """Capture a frame from camera and return as base64."""
    cap = cv2.VideoCapture(camera_id)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to capture frame from camera")

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    # Resize to reduce processing time on Pi
    img.thumbnail((384, 384))

    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def study_buddy_respond(image_b64: str) -> str:
    """Send image to study buddy model and get response."""
    response = ollama.chat(
        model="studybuddy",
        messages=[
            {
                "role": "user",
                "content": "Look at this student. How are they feeling? As a caring study buddy, respond appropriately.",
                "images": [image_b64],
            }
        ],
    )
    return response["message"]["content"]


def main():
    print("=" * 50)
    print("STUDY BUDDY — Emotionally Aware Assistant")
    print("=" * 50)
    print("Press Enter to capture a photo, or 'q' to quit.")

    while True:
        user_input = input("\\n> ").strip().lower()
        if user_input == "q":
            print("Goodbye! Keep up the great studying! 📚")
            break

        try:
            print("  📸 Capturing image...")
            image_b64 = capture_frame()

            print("  🤔 Analyzing emotion...")
            start = time.time()
            response = study_buddy_respond(image_b64)
            elapsed = time.time() - start

            print(f"\\n  🤖 Study Buddy says ({elapsed:.1f}s):")
            print(f"  {response}")

        except Exception as e:
            print(f"  [ERROR] {e}")


if __name__ == "__main__":
    main()
''')
    print(f"  ✓ Inference script saved to {inference_script}")


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned model for Raspberry Pi")
    parser.add_argument("--gguf_only", action="store_true", help="Skip merge, only do GGUF conversion")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default: 0)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("=" * 60)
    print("MODEL EXPORT FOR RASPBERRY PI DEPLOYMENT")
    print("=" * 60)
    print(f"  GPU: {args.gpu}")
    print(f"  Adapter: {CHECKPOINT_DIR}")
    print(f"  Merged output: {MERGED_DIR}")
    print(f"  GGUF output: {EXPORT_DIR}")

    if not args.gguf_only:
        step1_merge_lora()

    step2_convert_to_gguf()
    step3_create_ollama_modelfile()

    print("\n" + "=" * 60)
    print("✅ EXPORT COMPLETE!")
    print("=" * 60)
    print("\n  To deploy on Raspberry Pi:")
    print(f"  1. Copy the .gguf file and Modelfile from {EXPORT_DIR}/ to your Pi")
    print("  2. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh")
    print("  3. Create the model: ollama create studybuddy -f Modelfile")
    print("  4. Test: ollama run studybuddy")


if __name__ == "__main__":
    main()
