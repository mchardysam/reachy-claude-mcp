"""Centralized configuration for Reachy Claude MCP.

All settings can be configured via environment variables for easy deployment.
"""

import os
from pathlib import Path

# Data directory - where all persistent data is stored
REACHY_HOME = Path(os.environ.get("REACHY_CLAUDE_HOME", Path.home() / ".reachy-claude"))

# Qdrant vector store settings
QDRANT_HOST = os.environ.get("REACHY_QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("REACHY_QDRANT_PORT", "6333"))

# LLM settings
# MLX (Apple Silicon) - fastest local option
LLM_MODEL = os.environ.get("REACHY_LLM_MODEL", "mlx-community/Qwen2.5-1.5B-Instruct-4bit")

# Ollama (cross-platform) - requires Ollama running: https://ollama.ai
OLLAMA_HOST = os.environ.get("REACHY_OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("REACHY_OLLAMA_MODEL", "qwen2.5:1.5b")

# Voice settings
# If set, use this path directly. Otherwise auto-download to REACHY_HOME/voices/
VOICE_MODEL = os.environ.get("REACHY_VOICE_MODEL", None)

# Default voice model to download if none specified
DEFAULT_VOICE_NAME = "en_US-lessac-medium"
VOICE_MODEL_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
VOICE_CONFIG_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"


def get_voice_dir() -> Path:
    """Get the directory for voice models."""
    return REACHY_HOME / "voices"


def get_database_path() -> Path:
    """Get the path to the SQLite database."""
    return REACHY_HOME / "projects.db"


def get_memory_path() -> Path:
    """Get the path to the memory JSON file."""
    return REACHY_HOME / "memory.json"


def ensure_directories() -> None:
    """Ensure all required directories exist."""
    REACHY_HOME.mkdir(parents=True, exist_ok=True)
    get_voice_dir().mkdir(parents=True, exist_ok=True)
