"""
Utility functions for image preprocessing, output parsing, and validation.
"""

import re
import json
import hashlib
from io import BytesIO
from typing import Tuple, Optional
from PIL import Image


def preprocess_image(image_bytes: bytes, max_size: int = 512) -> bytes:
    """
    Preprocess image for consistent model input.
    
    - Converts to RGB (removes alpha channel)
    - Resizes if too large
    - Normalizes format to JPEG
    
    Args:
        image_bytes: Raw image bytes
        max_size: Maximum dimension (width or height)
    
    Returns:
        Preprocessed image bytes
    """
    img = Image.open(BytesIO(image_bytes))
    
    # Convert to RGB if needed (remove alpha channel)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize if too large (maintain aspect ratio)
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Convert to JPEG bytes
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    return buffer.getvalue()


def normalize_text(text: str) -> str:
    """
    Normalize text for parsing.
    
    - Lowercase
    - Strip whitespace
    - Remove punctuation
    
    Args:
        text: Raw text output
    
    Returns:
        Normalized text
    """
    text = text.lower().strip()
    # Remove common punctuation
    text = re.sub(r'[.,!?;:\'"()]', '', text)
    return text


def extract_label(output_text: str, allowed_labels: list) -> Tuple[str, float]:
    """
    Extract emotion label from model output.
    
    Tries multiple strategies with decreasing confidence:
    1. Exact match (confidence = 1.0)
    2. Single-word response (confidence = 0.9)
    3. JSON parsing (confidence = 0.85)
    4. Substring match (confidence = 0.5)
    5. Unknown (confidence = 0.0)
    
    Args:
        output_text: Raw model output
        allowed_labels: List of valid emotion labels
    
    Returns:
        Tuple of (label, confidence_score)
    """
    if not output_text:
        return "unknown", 0.0
    
    normalized = normalize_text(output_text)
    
    # Strategy 1: Exact match
    if normalized in allowed_labels:
        return normalized, 1.0
    
    # Strategy 2: Single-word response
    words = normalized.split()
    if len(words) == 1 and words[0] in allowed_labels:
        return words[0], 0.9
    
    # Strategy 3: JSON parsing (for JSON prompt variant)
    try:
        data = json.loads(output_text)
        if isinstance(data, dict) and "emotion" in data:
            emotion = normalize_text(data["emotion"])
            if emotion in allowed_labels:
                return emotion, 0.85
    except (json.JSONDecodeError, KeyError):
        pass
    
    # Strategy 4: Substring matching (with careful word boundary check)
    # Sort by length (longest first) to avoid "sad" matching in "unsad"
    sorted_labels = sorted(allowed_labels, key=len, reverse=True)
    for label in sorted_labels:
        # Use word boundary regex to avoid partial matches
        pattern = r'\b' + re.escape(label) + r'\b'
        if re.search(pattern, normalized):
            return label, 0.5
    
    # Strategy 5: No match found
    return "unknown", 0.0


def compute_image_hash(image_bytes: bytes) -> str:
    """
    Compute SHA256 hash of image bytes for caching.
    
    Args:
        image_bytes: Raw image bytes
    
    Returns:
        Hex string of hash
    """
    return hashlib.sha256(image_bytes).hexdigest()


def compute_cache_key(image_bytes: bytes, prompt: str) -> str:
    """
    Compute cache key for (image, prompt) pair.
    
    Args:
        image_bytes: Raw image bytes
        prompt: Prompt text
    
    Returns:
        Hex string of combined hash
    """
    combined = image_bytes + prompt.encode('utf-8')
    return hashlib.sha256(combined).hexdigest()


def validate_image(image_bytes: bytes) -> Tuple[bool, Optional[str]]:
    """
    Validate that bytes represent a valid image.
    
    Args:
        image_bytes: Raw image bytes
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        img = Image.open(BytesIO(image_bytes))
        img.verify()  # Verify it's actually an image
        
        # Check reasonable size constraints
        if img.size[0] < 32 or img.size[1] < 32:
            return False, "Image too small (minimum 32x32 pixels)"
        
        if img.size[0] > 4096 or img.size[1] > 4096:
            return False, "Image too large (maximum 4096x4096 pixels)"
        
        return True, None
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"
