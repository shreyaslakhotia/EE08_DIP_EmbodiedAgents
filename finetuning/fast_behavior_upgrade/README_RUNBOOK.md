# Fast Behavioral Upgrade Runbook (Qwen2.5-VL-3B)

This runbook is for a restricted setup:
- code prepared on laptop
- execution done manually on Linux GPU workstation
- objective: fast practical behavior upgrade, not research-grade perfection

## What This Pipeline Does

1. Scans your dataset recursively and uses folder names as weak labels.
2. Extracts a few frames from each video.
3. Optionally transcribes audio with Whisper.
4. Builds synthetic multimodal instruction-response pairs.
5. Fine-tunes Qwen2.5-VL-3B using LoRA/QLoRA.
6. Runs an inference smoke test and prints readable outputs.

## Required Folder

Pipeline folder:
- `finetuning/fast_behavior_upgrade/`

Scripts:
- `preprocess_dataset.py`
- `train_lora.py`
- `inference_test.py`
- `run_finetune.sh`

## Dataset Path (Workstation)

Expected root path:

`/mlda/shreyas_projects/EE08_DIP_EmbodiedAgents/data/OneDrive_2026-03-28/E008 Dataset/`

Folder names are used as weak supervision using this fixed mapping:

- Doomscrolling -> distracted / avoidant, off_task, gentle_redirect
- Actually Studying!!!!! -> focused, on_task, encourage_or_silent
- Fidgeting -> restless / stressed, low_focus, grounding_or_break
- Playing game -> engaged_elsewhere, off_task, soft_redirect
- Stretching -> neutral / recovering, break, positive_reinforcement
- Looking through lecture notes -> engaged, on_task, light_encouragement
- Looking at quiz or exam result -> anxious / reflective, uncertain, reassurance
- Watching lecture videos -> passive_learning, semi_on_task, encourage_active_learning
- Setting up timetable -> planning, on_task, supportive
- Empty table -> no_person, none, no_intervention

## Environment Setup

From the workstation:

```bash
cd /mlda/shreyas_projects/EE08_DIP_EmbodiedAgents/finetuning/fast_behavior_upgrade
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Recommended End-to-End Run

```bash
cd /mlda/shreyas_projects/EE08_DIP_EmbodiedAgents/finetuning/fast_behavior_upgrade
chmod +x run_finetune.sh
./run_finetune.sh
```

## Manual Step-by-Step (Explicit)

### 1) Preprocess

```bash
cd /mlda/shreyas_projects/EE08_DIP_EmbodiedAgents/finetuning/fast_behavior_upgrade
source .venv/bin/activate

python preprocess_dataset.py \
  --data_root "/mlda/shreyas_projects/EE08_DIP_EmbodiedAgents/data/OneDrive_2026-03-28/E008 Dataset/" \
  --output_path "/mlda/shreyas_projects/EE08_DIP_EmbodiedAgents/finetuning/fast_behavior_upgrade/outputs/dataset_all.jsonl" \
  --frames_per_video 3 \
  --enable_transcription \
  --transcriber faster-whisper \
  --whisper_model small
```

Debug mode for quick verification:

```bash
python preprocess_dataset.py \
  --data_root "/mlda/shreyas_projects/EE08_DIP_EmbodiedAgents/data/OneDrive_2026-03-28/E008 Dataset/" \
  --output_path "/mlda/shreyas_projects/EE08_DIP_EmbodiedAgents/finetuning/fast_behavior_upgrade/outputs/dataset_all.jsonl" \
  --frames_per_video 2 \
  --max_files 50
```

Expected outputs:
- `outputs/dataset_all.jsonl`
- `outputs/train.jsonl`
- `outputs/val.jsonl`
- `outputs/preprocess_errors.log`
- `outputs/extracted_frames/...`

### 2) Train LoRA/QLoRA

```bash
python train_lora.py \
  --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct" \
  --train_jsonl "/mlda/shreyas_projects/EE08_DIP_EmbodiedAgents/finetuning/fast_behavior_upgrade/outputs/train.jsonl" \
  --val_jsonl "/mlda/shreyas_projects/EE08_DIP_EmbodiedAgents/finetuning/fast_behavior_upgrade/outputs/val.jsonl" \
  --output_dir "/mlda/shreyas_projects/EE08_DIP_EmbodiedAgents/finetuning/fast_behavior_upgrade/outputs/run_manual" \
  --epochs 1 \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --use_4bit
```

If OOM occurs:
- lower `--max_length` to `768`
- lower `--gradient_accumulation_steps` to `4`
- keep `--per_device_train_batch_size 1`

### 3) Inference Smoke Test

```bash
python inference_test.py \
  --base_model "Qwen/Qwen2.5-VL-3B-Instruct" \
  --adapter_path "/mlda/shreyas_projects/EE08_DIP_EmbodiedAgents/finetuning/fast_behavior_upgrade/outputs/run_manual/adapter" \
  --dataset_jsonl "/mlda/shreyas_projects/EE08_DIP_EmbodiedAgents/finetuning/fast_behavior_upgrade/outputs/val.jsonl" \
  --max_samples 5 \
  --max_new_tokens 128 \
  --use_4bit
```

Expected output behavior:
- model identifies scenario-like behavior (off task, on task, reflective, etc.)
- responses are short, empathetic, and intervention-style aware
- less generic guidance than prior model

## Common Errors and Fixes

### ffmpeg not found
Symptoms:
- preprocessing warns that ffmpeg is missing
- video audio transcription skipped

Fix:
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

### CUDA out of memory
Symptoms:
- training crashes with OOM

Fix order:
1. use `--max_length 768`
2. keep `--per_device_train_batch_size 1`
3. use `--epochs 1`
4. close other GPU jobs (`nvidia-smi`)

### bitsandbytes or 4-bit load issues
Symptoms:
- errors while loading quantized model

Fix:
```bash
python -m pip install --upgrade bitsandbytes accelerate transformers
```
If still broken, run without `--use_4bit` (uses more VRAM).

### Corrupted media files
Symptoms:
- warnings in preprocess logs

Fix:
- check `outputs/preprocess_errors.log`
- keep running; script skips bad files by design

### Folder names with spaces/punctuation
This is already supported. Keep all path arguments wrapped in quotes.

## Practical Notes

- This is intentionally low-effort and fast for behavioral improvement under time pressure.
- Multiple samples per video are created by frame extraction.
- Weak labels come from scenario folder names, not manual annotation.
- Pipeline is resilient to missing modalities and partial data quality issues.

## Optional 2-Epoch Run

If you have enough time and stable GPU memory:

```bash
python train_lora.py \
  --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct" \
  --train_jsonl "/mlda/shreyas_projects/EE08_DIP_EmbodiedAgents/finetuning/fast_behavior_upgrade/outputs/train.jsonl" \
  --val_jsonl "/mlda/shreyas_projects/EE08_DIP_EmbodiedAgents/finetuning/fast_behavior_upgrade/outputs/val.jsonl" \
  --output_dir "/mlda/shreyas_projects/EE08_DIP_EmbodiedAgents/finetuning/fast_behavior_upgrade/outputs/run_epoch2" \
  --epochs 2 \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --use_4bit
```
