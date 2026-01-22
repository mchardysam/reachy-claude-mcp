"""LLM backend implementations for sentiment analysis and summarization.

Supports multiple backends with automatic fallback:
1. MLX (Apple Silicon) - fastest, local
2. Ollama (cross-platform) - local, requires Ollama running
3. None - falls back to keyword matching in LLMAnalyzer
"""

import json
import urllib.request
import urllib.error
from typing import Protocol, runtime_checkable

from .config import LLM_MODEL, OLLAMA_HOST, OLLAMA_MODEL


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol for LLM backends."""

    @property
    def name(self) -> str:
        """Backend name for logging."""
        ...

    @property
    def available(self) -> bool:
        """Check if this backend is available."""
        ...

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate text from a prompt."""
        ...


class MLXBackend:
    """MLX-based backend for Apple Silicon Macs."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or LLM_MODEL
        self._model = None
        self._tokenizer = None
        self._load_attempted = False
        self._available: bool | None = None

    @property
    def name(self) -> str:
        return "MLX"

    @property
    def available(self) -> bool:
        """Check if MLX is available."""
        if self._available is not None:
            return self._available

        try:
            from mlx_lm import load, generate
            self._available = True
        except ImportError:
            self._available = False

        return self._available

    def _ensure_loaded(self) -> bool:
        """Load the model if not already loaded."""
        if self._model is not None:
            return True

        if self._load_attempted:
            return False

        if not self.available:
            return False

        try:
            self._load_attempted = True
            from mlx_lm import load
            self._model, self._tokenizer = load(self.model_name)
            return True
        except Exception as e:
            print(f"Failed to load MLX model: {e}")
            return False

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate text using MLX."""
        if not self._ensure_loaded():
            return ""

        from mlx_lm import generate
        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        return response.strip()


class OllamaBackend:
    """Ollama-based backend for cross-platform support."""

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ):
        self.host = host or OLLAMA_HOST
        self.model = model or OLLAMA_MODEL
        self.timeout = timeout
        self._available: bool | None = None

    @property
    def name(self) -> str:
        return "Ollama"

    @property
    def available(self) -> bool:
        """Check if Ollama is running and responsive."""
        if self._available is not None:
            return self._available

        try:
            # Quick health check
            req = urllib.request.Request(
                f"{self.host}/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                self._available = resp.status == 200
        except (urllib.error.URLError, TimeoutError, OSError):
            self._available = False

        return self._available or False

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate text using Ollama API."""
        if not self.available:
            return ""

        try:
            data = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                }
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "").strip()

        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
            print(f"Ollama request failed: {e}")
            return ""


# Singleton backends
_mlx_backend: MLXBackend | None = None
_ollama_backend: OllamaBackend | None = None


def get_mlx_backend() -> MLXBackend:
    """Get the MLX backend singleton."""
    global _mlx_backend
    if _mlx_backend is None:
        _mlx_backend = MLXBackend()
    return _mlx_backend


def get_ollama_backend() -> OllamaBackend:
    """Get the Ollama backend singleton."""
    global _ollama_backend
    if _ollama_backend is None:
        _ollama_backend = OllamaBackend()
    return _ollama_backend


def get_best_backend() -> LLMBackend | None:
    """Get the best available LLM backend.

    Priority:
    1. MLX (fastest on Apple Silicon)
    2. Ollama (cross-platform)
    3. None (will use keyword fallback)
    """
    # Try MLX first (fastest on Apple Silicon)
    mlx = get_mlx_backend()
    if mlx.available:
        return mlx

    # Try Ollama (cross-platform)
    ollama = get_ollama_backend()
    if ollama.available:
        return ollama

    # No backend available
    return None
