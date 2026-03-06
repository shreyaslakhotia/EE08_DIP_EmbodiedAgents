"""
Streamlit UI for emotion classification demo.

Phase 0: Qualitative probing interface
- Upload image
- Select/edit prompt
- View raw + parsed results
- Debug information
"""

import streamlit as st
import time
from io import BytesIO

from provider import get_provider
from prompts import ALLOWED_LABELS, PROMPT_VERSIONS, get_prompt
from utils import (
    preprocess_image, 
    extract_label,
    extract_empathy_response,
    validate_image, 
    compute_cache_key
)


# Page configuration
st.set_page_config(
    page_title="Emotion Classification Demo",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Cached model initialization
@st.cache_resource
def initialize_provider():
    """Initialize and cache the model provider."""
    try:
        provider = get_provider()
        return provider, None
    except Exception as e:
        return None, str(e)


# Cached prediction function
@st.cache_data(show_spinner=False)
def predict_emotion_cached(image_bytes: bytes, prompt: str, _provider, use_empathy: bool = False):
    """
    Cached prediction function to avoid redundant API calls.
    
    Note: _provider is prefixed with underscore to exclude from cache key.
    Cache key is based on (image_bytes, prompt, use_empathy) only.
    """
    try:
        # Preprocess image
        processed = preprocess_image(image_bytes)
        
        # Call model
        start_time = time.time()
        raw_output = _provider.predict(processed, prompt)
        elapsed = time.time() - start_time
        
        # Parse output based on mode
        if use_empathy:
            label, confidence, empathy_msg = extract_empathy_response(raw_output, ALLOWED_LABELS)
        else:
            label, confidence = extract_label(raw_output, ALLOWED_LABELS)
            empathy_msg = None
        
        return {
            "success": True,
            "raw_output": raw_output,
            "label": label,
            "confidence": confidence,
            "empathy_message": empathy_msg,
            "elapsed_time": elapsed,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def main():
    # Header
    st.title("😊 Emotion Classification Demo")
    st.markdown("""
    Upload an image with a facial expression and classify the emotion.
    
    **Allowed emotions**: `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Initialize provider
        provider, error = initialize_provider()
        
        if error:
            st.error("❌ Provider Initialization Failed")
            st.code(error, language=None)
            st.markdown("""
            **Troubleshooting**:
            1. Create `.env` file from `.env.example`
            2. Add your `HF_API_TOKEN`
            3. Verify `MODEL_PROVIDER=huggingface`
            """)
            st.stop()
        
        # Show provider info
        info = provider.get_info()
        st.success(f"✓ Connected to {info['provider']}")
        
        with st.expander("Provider Details"):
            st.json(info)
        
        st.divider()
        
        # Prompt selection
        st.subheader("📝 Prompt Selection")
        prompt_version = st.selectbox(
            "Prompt Strategy",
            options=list(PROMPT_VERSIONS.keys()),
            index=0,
            help="Different prompts may yield different output formats"
        )
        
        selected_prompt = get_prompt(prompt_version)
        
        # Allow manual editing
        st.markdown("**Edit prompt (optional)**:")
        custom_prompt = st.text_area(
            "Prompt text",
            value=selected_prompt,
            height=200,
            label_visibility="collapsed"
        )
        
        if custom_prompt != selected_prompt:
            st.info("Using custom prompt")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image with a face",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            # Read image bytes
            image_bytes = uploaded_file.read()
            
            # Validate image
            is_valid, error_msg = validate_image(image_bytes)
            
            if not is_valid:
                st.error(f"❌ Invalid image: {error_msg}")
                st.stop()
            
            # Display uploaded image
            st.image(image_bytes, caption="Uploaded Image", use_container_width=True)
            
            # File info
            st.caption(f"Size: {len(image_bytes) / 1024:.1f} KB | Format: {uploaded_file.type}")
    
    with col2:
        st.subheader("🎯 Classification Results")
        
        if uploaded_file is None:
            st.info("👈 Upload an image to get started")
        else:
            # Detect if using empathy prompt
            use_empathy = prompt_version == "empathy"
            
            # Classify button
            if st.button("🚀 Classify Emotion", type="primary", use_container_width=True):
                with st.spinner("Analyzing image... (may take 20-60s on first call)"):
                    result = predict_emotion_cached(image_bytes, custom_prompt, provider, use_empathy)
                
                if result["success"]:
                    # Display parsed label prominently
                    label = result["label"]
                    confidence = result["confidence"]
                    empathy_msg = result.get("empathy_message")
                    
                    if label == "unknown":
                        st.error("❌ Could not parse emotion label")
                    else:
                        # Color-coded emotion display
                        emotion_colors = {
                            "happy": "🟢",
                            "surprise": "🟡",
                            "neutral": "⚪",
                            "sad": "🔵",
                            "angry": "🔴",
                            "fear": "🟣",
                            "disgust": "🟤",
                        }
                        
                        emoji = emotion_colors.get(label, "⚫")
                        st.markdown(f"### {emoji} **{label.upper()}**")
                        
                        # Confidence meter
                        st.metric(
                            "Confidence", 
                            f"{confidence:.0%}",
                            delta=None,
                            help="How confidently the label was parsed from model output"
                        )
                    
                    # Display empathy message if available
                    if empathy_msg and label != "unknown":
                        st.divider()
                        with st.expander("💬 Emotional Support Message", expanded=True):
                            st.info(empathy_msg, icon="💙")
                            st.caption("""
                            *This is an AI-generated supportive response. 
                            Not a substitute for professional mental health support.*
                            """)
                    
                    st.divider()
                    
                    # Raw output
                    st.markdown("**Raw Model Output:**")
                    st.code(result["raw_output"], language=None)
                    
                    # Timing info
                    st.caption(f"⏱️ Response time: {result['elapsed_time']:.2f}s")
                    
                    # Interpretation guide
                    if confidence < 0.7:
                        st.warning("""
                        ⚠️ Low confidence parsing. The model may be outputting unexpected format.
                        Try a different prompt strategy or check the raw output.
                        """)
                
                else:
                    st.error("❌ Classification Failed")
                    st.code(result["error"], language=None)
                    
                    st.markdown("""
                    **Common issues**:
                    - Model is loading (wait 30s and retry)
                    - Invalid API token
                    - Rate limit exceeded
                    - Network timeout
                    """)
    
    # Debug panel (collapsible)
    with st.expander("🔧 Debug Information"):
        st.markdown("**Allowed Labels:**")
        st.code(", ".join(ALLOWED_LABELS))
        
        st.markdown("**Current Prompt:**")
        st.code(custom_prompt, language=None)
        
        if uploaded_file is not None:
            st.markdown("**Image Info:**")
            st.json({
                "filename": uploaded_file.name,
                "size_bytes": len(image_bytes),
                "type": uploaded_file.type,
            })
    
    # Footer
    st.divider()
    st.caption("""
    **Phase 0 Demo** | Qualitative probing only | 
    Model: Qwen/Qwen-VL-Chat via Hugging Face | 
    [EE3180 DIP Project]
    """)


if __name__ == "__main__":
    main()
