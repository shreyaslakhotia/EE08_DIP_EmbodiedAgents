"""
Model provider abstraction for emotion classification.

This module provides a clean interface to swap between different model backends:
- Hugging Face Inference API (Phase 0)
- GPU Cluster REST API (Phase 1+)
"""

import os
import base64
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()


class BaseProvider(ABC):
    """Abstract base class for model providers."""
    
    @abstractmethod
    def predict(self, image_bytes: bytes, prompt: str) -> str:
        """
        Send image + prompt to model and return raw text output.
        
        Args:
            image_bytes: Raw image bytes (JPEG/PNG)
            prompt: Text prompt for the model
        
        Returns:
            Raw model output as string
        
        Raises:
            ValueError: If API call fails
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get provider information for debugging.
        
        Returns:
            Dictionary with provider details (name, endpoint, etc.)
        """
        pass


class HuggingFaceProvider(BaseProvider):
    """
    Hugging Face Inference API provider using InferenceClient.
    
    Supports vision-language models like Qwen/Qwen2-VL-7B-Instruct.
    """
    
    def __init__(self, api_token: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize Hugging Face provider.
        
        Args:
            api_token: HF API token (or reads from HF_API_TOKEN env var)
            model_name: Model name (or reads from HF_MODEL_NAME env var)
        """
        self.api_token = api_token or os.getenv("HF_API_TOKEN")
        self.model_name = model_name or os.getenv("HF_MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")
        
        if not self.api_token:
            raise ValueError("HF_API_TOKEN not found in environment variables")
        
        # Initialize InferenceClient
        self.client = InferenceClient(
            model=self.model_name,
            token=self.api_token
        )
    
    def predict(self, image_bytes: bytes, prompt: str) -> str:
        """
        Call Hugging Face Inference API using InferenceClient.
        
        For vision-language models, we use chat_completion with base64 encoded image.
        """
        try:
            # Encode image as base64 data URI
            img_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Determine image format from bytes
            image_format = "jpeg"
            if image_bytes.startswith(b'\x89PNG'):
                image_format = "png"
            
            # Create data URI
            data_uri = f"data:image/{image_format};base64,{img_b64}"
            
            # Build message with image URL and text
            # Format for VLMs using chat: content as string with image reference
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Call chat completion (conversational task)
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=50,
                temperature=0.1,
            )
            
            # Extract generated text
            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            elif isinstance(response, dict) and 'choices' in response:
                return response['choices'][0]['message']['content']
            else:
                return str(response)
        
        except Exception as e:
            error_msg = str(e)
            
            # Handle common errors with helpful messages
            if "503" in error_msg or "loading" in error_msg.lower():
                raise ValueError(
                    "Model is loading. This can take 20-60 seconds on first request. "
                    "Please wait and try again."
                )
            elif "401" in error_msg or "unauthorized" in error_msg.lower():
                raise ValueError("Invalid API token. Please check your HF_API_TOKEN.")
            elif "404" in error_msg:
                raise ValueError(f"Model not found: {self.model_name}. Please check HF_MODEL_NAME.")
            elif "timeout" in error_msg.lower():
                raise ValueError("Request timed out. The model may be overloaded.")
            else:
                raise ValueError(f"API Error: {error_msg}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider information."""
        return {
            "provider": "Hugging Face Inference API (InferenceClient)",
            "model": self.model_name,
            "has_token": bool(self.api_token),
        }


class ClusterProvider(BaseProvider):
    """
    GPU Cluster REST API provider (Phase 1+).
    
    This will be implemented when the cluster-hosted model is ready.
    """
    
    def __init__(self, endpoint: Optional[str] = None):
        """
        Initialize cluster provider.
        
        Args:
            endpoint: Cluster API endpoint (or reads from CLUSTER_API_ENDPOINT env var)
        """
        self.endpoint = endpoint or os.getenv("CLUSTER_API_ENDPOINT")
        
        if not self.endpoint:
            raise ValueError("CLUSTER_API_ENDPOINT not found in environment variables")
    
    def predict(self, image_bytes: bytes, prompt: str) -> str:
        """
        Call cluster REST API.
        
        Expects a FastAPI endpoint that accepts:
        - multipart/form-data with 'image' file and 'prompt' text
        
        Returns:
        - JSON with {"label": "...", "raw_output": "..."}
        """
        files = {
            "image": ("image.jpg", image_bytes, "image/jpeg")
        }
        data = {
            "prompt": prompt
        }
        
        try:
            response = requests.post(
                f"{self.endpoint}/predict",
                files=files,
                data=data,
                timeout=30
            )
            
            if response.status_code != 200:
                raise ValueError(
                    f"Cluster API Error {response.status_code}: {response.text}"
                )
            
            result = response.json()
            return result.get("raw_output", result.get("label", str(result)))
        
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Cluster connection error: {str(e)}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider information."""
        return {
            "provider": "GPU Cluster (gpu25.dynip.ntu.edu.sg)",
            "endpoint": self.endpoint,
            "status": "Not yet implemented (Phase 1)",
        }


def get_provider(provider_name: Optional[str] = None) -> BaseProvider:
    """
    Factory function to get the appropriate provider.
    
    Args:
        provider_name: "huggingface" or "cluster" (or reads from MODEL_PROVIDER env var)
    
    Returns:
        Initialized provider instance
    
    Raises:
        ValueError: If provider name is invalid or configuration is missing
    """
    provider_name = provider_name or os.getenv("MODEL_PROVIDER", "huggingface")
    provider_name = provider_name.lower()
    
    if provider_name in ["huggingface", "hf"]:
        return HuggingFaceProvider()
    elif provider_name == "cluster":
        return ClusterProvider()
    else:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            "Valid options: 'huggingface', 'cluster'"
        )
