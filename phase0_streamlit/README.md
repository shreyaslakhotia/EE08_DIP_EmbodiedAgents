# Phase 0: Emotion Classification Streamlit Demo

Qualitative probing interface for emotion classification using Qwen-VL-Chat.

## Setup

### 1. Create Virtual Environment

```powershell
python -m venv ee3180_dip
ee3180_dip\Scripts\activate
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure Environment

```powershell
# Copy template
copy .env.example .env

# Edit .env and add your Hugging Face API token
# Get token from: https://huggingface.co/settings/tokens
# Model: Qwen/Qwen2-VL-7B-Instruct (uses InferenceClient)
```

### 4. Test API Connection

```powershell
python test_qwen_api.py --image path\to\test_image.jpg
```

### 5. Launch Streamlit App

```powershell
streamlit run app.py
```

## Project Structure

```
phase0_streamlit/
├── app.py              # Streamlit UI
├── provider.py         # Model provider abstraction
├── prompts.py          # Prompt engineering variants
├── utils.py            # Parsing & validation utilities
├── test_qwen_api.py    # API validation script
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template
└── .gitignore          # Git ignore rules
```

## Usage

1. Upload an image with a face showing emotion
2. Select prompt variant (or edit manually)
3. Click "Classify Emotion"
4. View results:
   - Uploaded image
   - Raw model output
   - Parsed emotion label
   - Confidence indicator

## Allowed Emotion Labels

`angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`

## Phase 0 Scope

- ✅ Qualitative probing only
- ✅ Single image upload
- ✅ Provider abstraction (easy cluster swap)
- ✅ Multiple prompt variants
- ❌ No batch processing
- ❌ No accuracy metrics
- ❌ No dataset evaluation

## Next Steps (Phase 1)

- Deploy Qwen-VL on GPU cluster
- Create FastAPI REST endpoint
- Swap provider implementation
- Add batch processing mode
