import os
from typing import Any

import requests

SYSTEM_PROMPT = "You are an embodied Study Buddy. Provide concise, helpful answers."
DEFAULT_SERVER_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL_NAME = "stanky2"


class BrainClientError(Exception):
    """Raised when the Ollama model server request fails."""


class BrainClient:
    """Thin stateless HTTP client for Ollama chat completions."""

    def __init__(
        self,
        server_url: str = DEFAULT_SERVER_URL,
        model: str = DEFAULT_MODEL_NAME,
        timeout_seconds: int = 60,
    ) -> None:
        self.server_url = server_url
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate_response(self, messages: list[dict[str, Any]]) -> str:
        if not messages:
            raise BrainClientError("messages cannot be empty")

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": -1,
        }

        try:
            response = requests.post(
                self.server_url,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise BrainClientError(str(exc)) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise BrainClientError("invalid JSON response from Ollama") from exc

        content = data.get("message", {}).get("content")
        if not isinstance(content, str):
            raise BrainClientError("missing message content in Ollama response")
        return content


def _env_or_default(name: str, default_value: str) -> str:
    value = os.getenv(name)
    return value if value else default_value


def generate_response(messages: list[dict[str, Any]]) -> str:
    """Convenience API required by the Telegram bot and simple scripts."""
    model_name = _env_or_default("OLLAMA_MODEL", DEFAULT_MODEL_NAME)
    server_url = _env_or_default("OLLAMA_CHAT_URL", DEFAULT_SERVER_URL)
    client = BrainClient(server_url=server_url, model=model_name)
    return client.generate_response(messages)
