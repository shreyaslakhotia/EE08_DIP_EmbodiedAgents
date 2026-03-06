# EE08_DIP_EmbodiedAgents

EE3180 Design & Innovation Project — **Emotionally Aware Study Buddy Robot**

## Project Overview

An embodied AI agent that detects student emotions from camera input and responds with empathetic, supportive study guidance. Deployed on a **Raspberry Pi 5** (16GB).

### Key Components

| Component | Description |
|-----------|-------------|
| **Vision-Language Model** | Qwen2.5-VL-3B fine-tuned with QLoRA for emotion detection + empathetic response |
| **Deployment** | GGUF Q4_K_M quantized (~3.1 GB) running via Ollama on Raspberry Pi 5 |
| **Emotion Classes** | angry, disgust, fear, happy, neutral, sad, surprise |
| **Accuracy** | 63.3% (competitive with FER-2013 human agreement of ~65%) |

## Fine-tuned Weights

📥 **[Download GGUF Weights & LoRA Adapter (OneDrive)](https://entuedu-my.sharepoint.com/my?id=%2Fpersonal%2Fshreyas010%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FEE3180&ga=1)**

The OneDrive folder contains:
- `studybuddy_gguf/` — Quantized GGUF model files for Raspberry Pi deployment (~3.1 GB)
  - `qwen2.5-vl-3b-instruct.Q4_K_M.gguf` (1.8 GB) — quantized language model
  - `qwen2.5-vl-3b-instruct.F16-mmproj.gguf` (1.3 GB) — vision encoder

## Fine-Tuning Pipeline

The complete fine-tuning pipeline is documented in **[FINETUNE_README.md](FINETUNE_README.md)**. Summary:

```bash
python 01_explore_dataset.py          # Analyze FER-2013 dataset
python 02_prepare_finetune_data.py    # Preprocess images + build JSONL
python 03_finetune_qwen3vl_lora.py    # QLoRA fine-tuning (~4.5 hours)
python 04_export_model.py             # Export to GGUF for Raspberry Pi
python 05_evaluate_test_accuracy.py   # Evaluate test accuracy
```

## Quick Deploy to Raspberry Pi

```bash
# 1. Download studybuddy_gguf/ folder from OneDrive (link above)
# 2. Copy to Pi along with Modelfile and run_studybuddy.py from checkpoints/qwen3vl_studybuddy/export/

# On the Pi:
curl -fsSL https://ollama.com/install.sh | sh
cd ~/studybuddy && ollama create studybuddy -f Modelfile
python run_studybuddy.py   # Camera-based inference
```

## Other Components

| Folder | Description |
|--------|-------------|
| `phase0_streamlit/` | Streamlit-based prototype with cloud Qwen API |
| `Qwen3-VL-4B Prototypes/` | Stage 1 & 2 milestone prototypes |
| `modular codes/` | Camera, PIR sensor, and Ollama integration tests |

## License

See [LICENSE](LICENSE).
