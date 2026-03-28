#!/usr/bin/env bash
set -euo pipefail

# Fast behavioral upgrade pipeline runner for Linux GPU workstation.
# Usage:
#   chmod +x run_finetune.sh
#   ./run_finetune.sh

DATA_ROOT="/mlda/shreyas_projects/EE08_DIP_EmbodiedAgents/data/OneDrive_2026-03-28/E008 Dataset/"
WORK_DIR="/mlda/shreyas_projects/EE08_DIP_EmbodiedAgents/finetuning/fast_behavior_upgrade"
OUT_DIR="${WORK_DIR}/outputs"
DATASET_ALL="${OUT_DIR}/dataset_all.jsonl"
TRAIN_JSONL="${OUT_DIR}/train.jsonl"
VAL_JSONL="${OUT_DIR}/val.jsonl"
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
RUN_DIR="${OUT_DIR}/run_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUT_DIR}" "${RUN_DIR}"

cd "${WORK_DIR}"

# 1) Environment setup
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 2) Preprocess weak labels + extract frames + optional transcription
python preprocess_dataset.py \
  --data_root "${DATA_ROOT}" \
  --output_path "${DATASET_ALL}" \
  --frames_per_video 3 \
  --enable_transcription \
  --transcriber faster-whisper \
  --whisper_model small

# 3) Train QLoRA adapter (fast, low-effort settings)
python train_lora.py \
  --model_name_or_path "${MODEL_NAME}" \
  --train_jsonl "${TRAIN_JSONL}" \
  --val_jsonl "${VAL_JSONL}" \
  --output_dir "${RUN_DIR}" \
  --epochs 1 \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --use_4bit

# 4) Inference smoke test on a few validation samples
python inference_test.py \
  --base_model "${MODEL_NAME}" \
  --adapter_path "${RUN_DIR}/adapter" \
  --dataset_jsonl "${VAL_JSONL}" \
  --max_samples 5 \
  --max_new_tokens 128 \
  --use_4bit

echo "Pipeline complete. Outputs are in: ${RUN_DIR}"
