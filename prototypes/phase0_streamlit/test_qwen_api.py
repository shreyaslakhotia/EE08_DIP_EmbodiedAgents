"""
Test script to validate Hugging Face API access.

Usage:
    python test_qwen_api.py --image path/to/image.jpg
    python test_qwen_api.py --image test.jpg --prompt custom

Before running:
    1. Create .env file: copy .env.example .env
    2. Add your HF_API_TOKEN to .env
    3. Ensure MODEL_PROVIDER=huggingface in .env
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from provider import get_provider
from prompts import STRICT_PROMPT, get_prompt, ALLOWED_LABELS
from utils import preprocess_image, extract_label, validate_image


def test_api(image_path: str, prompt_version: str = "strict"):
    """
    Test the Hugging Face API with a single image.
    
    Args:
        image_path: Path to test image
        prompt_version: Which prompt to use ("strict", "json", "chain-of-thought")
    """
    print("=" * 60)
    print("EMOTION CLASSIFICATION API TEST")
    print("=" * 60)
    
    # Load image
    print(f"\n[1/5] Loading image: {image_path}")
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        print(f"✓ Image loaded: {len(image_bytes)} bytes")
    except FileNotFoundError:
        print(f"✗ Error: File not found: {image_path}")
        return False
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        return False
    
    # Validate image
    print("\n[2/5] Validating image...")
    is_valid, error_msg = validate_image(image_bytes)
    if not is_valid:
        print(f"✗ Invalid image: {error_msg}")
        return False
    print("✓ Image is valid")
    
    # Preprocess image
    print("\n[3/5] Preprocessing image...")
    try:
        processed_bytes = preprocess_image(image_bytes, max_size=512)
        print(f"✓ Image preprocessed: {len(processed_bytes)} bytes")
    except Exception as e:
        print(f"✗ Error preprocessing: {e}")
        return False
    
    # Initialize provider
    print("\n[4/5] Connecting to model provider...")
    try:
        provider = get_provider()
        info = provider.get_info()
        print(f"✓ Provider: {info['provider']}")
        print(f"  Model: {info.get('model', 'N/A')}")
        print(f"  Endpoint: {info.get('endpoint', 'N/A')}")
    except Exception as e:
        print(f"✗ Error initializing provider: {e}")
        print("\nTroubleshooting:")
        print("  1. Check that .env file exists")
        print("  2. Verify HF_API_TOKEN is set correctly")
        print("  3. Ensure MODEL_PROVIDER=huggingface")
        return False
    
    # Get prompt
    prompt = get_prompt(prompt_version)
    print(f"\nUsing prompt version: {prompt_version}")
    print(f"Prompt preview: {prompt[:100]}...")
    
    # Call API
    print("\n[5/5] Calling model API...")
    print("⏳ This may take 20-60 seconds on first call (model loading)...")
    
    start_time = time.time()
    try:
        raw_output = provider.predict(processed_bytes, prompt)
        elapsed = time.time() - start_time
        print(f"✓ Response received in {elapsed:.2f}s")
    except Exception as e:
        print(f"✗ API call failed: {e}")
        return False
    
    # Parse output
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nRaw model output:")
    print(f"  {raw_output}")
    
    label, confidence = extract_label(raw_output, ALLOWED_LABELS)
    print(f"\nParsed label: {label}")
    print(f"Confidence: {confidence:.2f}")
    
    if label == "unknown":
        print("\n⚠ WARNING: Could not parse a valid emotion label")
        print("Suggestions:")
        print("  1. Try a different prompt version")
        print("  2. Check if model output format changed")
        print("  3. Update parsing logic in utils.py")
    else:
        print(f"\n✓ Successfully classified as: {label.upper()}")
    
    print("\n" + "=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test emotion classification API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_qwen_api.py --image happy_face.jpg
  python test_qwen_api.py --image sad_face.jpg --prompt json
  python test_qwen_api.py --image neutral.jpg --prompt chain-of-thought
        """
    )
    
    parser.add_argument(
        "--image",
        required=True,
        help="Path to test image (JPG/PNG)"
    )
    
    parser.add_argument(
        "--prompt",
        choices=["strict", "json", "chain-of-thought"],
        default="strict",
        help="Which prompt variant to use (default: strict)"
    )
    
    args = parser.parse_args()
    
    success = test_api(args.image, args.prompt)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
