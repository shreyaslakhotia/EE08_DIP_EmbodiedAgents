"""
Prompt engineering for emotion classification.

All prompts are designed to constrain model output to one of the allowed emotion labels.
See README.md for prompt design rationale and version history.
"""

# Allowed emotion labels (canonical set)
ALLOWED_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Prompt Version 1: Strict single-word format
STRICT_PROMPT = """You are an emotion classifier.
Analyze the facial expression in this image and choose exactly one label from this list:
angry, disgust, fear, happy, sad, surprise, neutral

Output ONLY the label in lowercase. No punctuation. No explanations. No extra words.
Just the emotion word."""

# Prompt Version 2: JSON structured output
JSON_PROMPT = """You are an emotion classifier. Analyze the facial expression in this image.

Respond with valid JSON only:
{
  "emotion": "label",
  "confidence": 0.95
}

Valid emotion labels: angry, disgust, fear, happy, sad, surprise, neutral
Use lowercase for the emotion label."""

# Prompt Version 3: Chain-of-thought (more tokens, potentially better accuracy)
COT_PROMPT = """You are an expert emotion classifier. Analyze this image carefully:

Step 1: Observe the facial features (eyes, eyebrows, mouth, overall expression)
Step 2: Identify the dominant emotion
Step 3: Output your final answer

Your final answer must be ONLY ONE WORD from this list:
angry, disgust, fear, happy, sad, surprise, neutral

Output format: Just the emotion word in lowercase, nothing else."""

# Prompt Version 4: Empathy-enabled (classification + supportive message)
EMPATHY_PROMPT = """You are an empathetic AI companion with expertise in emotion recognition.

Analyze the facial expression in this image and:
1. Identify the emotion (choose ONE: angry, disgust, fear, happy, sad, surprise, neutral)
2. Provide a brief, caring response (2-3 sentences)

Your response should be warm, non-judgmental, and genuinely supportive.

Format your response EXACTLY as:
Emotion: [label]
Message: [your supportive message]

Example:
Emotion: happy
Message: I can see you're feeling joyful! It's wonderful to witness such positive energy. Keep embracing these bright moments."""

# Default prompt (easy to swap for experimentation)
DEFAULT_PROMPT = STRICT_PROMPT

# All available prompt variants
PROMPT_VERSIONS = {
    "strict": STRICT_PROMPT,
    "json": JSON_PROMPT,
    "chain-of-thought": COT_PROMPT,
    "empathy": EMPATHY_PROMPT,
}

def get_prompt(version: str = "strict") -> str:
    """
    Get a prompt by version name.
    
    Args:
        version: One of "strict", "json", "chain-of-thought", "empathy"
    
    Returns:
        The prompt string
    """
    return PROMPT_VERSIONS.get(version, DEFAULT_PROMPT)
