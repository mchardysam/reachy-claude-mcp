"""Piper TTS wrapper for local text-to-speech synthesis.

Supports cross-platform audio playback (macOS, Linux, Windows) and
auto-downloads voice models on first use.
"""

import asyncio
import platform
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

from .config import (
    VOICE_MODEL,
    VOICE_MODEL_URL,
    VOICE_CONFIG_URL,
    DEFAULT_VOICE_NAME,
    get_voice_dir,
    ensure_directories,
)


def _get_audio_player() -> tuple[str, list[str], bool]:
    """Get platform-appropriate audio player command.

    Returns:
        Tuple of (command, args, uses_file_placeholder)
        If uses_file_placeholder is True, '{file}' in args should be replaced with the file path.
    """
    system = platform.system()

    if system == "Darwin":  # macOS
        return "afplay", [], False
    elif system == "Linux":
        # Try paplay (PulseAudio), then aplay (ALSA)
        if shutil.which("paplay"):
            return "paplay", [], False
        elif shutil.which("aplay"):
            return "aplay", ["-q"], False
        raise RuntimeError(
            "No audio player found. Install pulseaudio (paplay) or alsa-utils (aplay).\n"
            "  Ubuntu/Debian: sudo apt install pulseaudio-utils\n"
            "  Fedora: sudo dnf install pulseaudio-utils"
        )
    elif system == "Windows":
        # Use PowerShell to play audio
        return "powershell", ["-c", "(New-Object Media.SoundPlayer '{file}').PlaySync()"], True
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


async def _download_file(url: str, dest: Path) -> None:
    """Download a file asynchronously."""
    loop = asyncio.get_event_loop()

    def _download():
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, dest)
        print(f"Saved to {dest}")

    await loop.run_in_executor(None, _download)


class PiperTTS:
    """Wrapper for Piper TTS with async speech synthesis."""

    def __init__(self, voice_model: str | None = None):
        """Initialize the TTS wrapper.

        Args:
            voice_model: Path to the ONNX voice model file.
                         If None, will use REACHY_VOICE_MODEL env var or auto-download.
        """
        self._voice_model_path = voice_model or VOICE_MODEL
        self._model_ready = False
        self._audio_player: tuple[str, list[str], bool] | None = None

    async def _ensure_voice_model(self) -> Path:
        """Ensure voice model exists, downloading if necessary.

        Returns:
            Path to the voice model file.
        """
        # If explicitly set and exists, use it
        if self._voice_model_path and Path(self._voice_model_path).exists():
            return Path(self._voice_model_path)

        # Use default location in data directory
        ensure_directories()
        voice_dir = get_voice_dir()
        model_path = voice_dir / f"{DEFAULT_VOICE_NAME}.onnx"
        config_path = voice_dir / f"{DEFAULT_VOICE_NAME}.onnx.json"

        # Download if not present
        if not model_path.exists():
            print(f"Voice model not found. Downloading to {model_path}...")
            await _download_file(VOICE_MODEL_URL, model_path)

        if not config_path.exists():
            await _download_file(VOICE_CONFIG_URL, config_path)

        return model_path

    def _get_audio_player(self) -> tuple[str, list[str], bool]:
        """Get cached audio player info."""
        if self._audio_player is None:
            self._audio_player = _get_audio_player()
        return self._audio_player

    async def speak(self, text: str) -> None:
        """Synthesize and play speech using Piper.

        Args:
            text: The text to speak.
        """
        # Ensure voice model is available
        voice_model = await self._ensure_voice_model()

        # Generate audio asynchronously
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            # Try to find piper in the same venv as the running Python
            venv_piper = Path(sys.executable).parent / "piper"
            if venv_piper.exists():
                piper_path = str(venv_piper)
            else:
                piper_path = shutil.which("piper")
                if not piper_path:
                    raise RuntimeError(
                        "piper executable not found. Install with: pip install piper-tts"
                    )

            # Run piper in a subprocess
            process = await asyncio.create_subprocess_exec(
                piper_path,
                "--model", str(voice_model),
                "--output_file", temp_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Send text as input
            stdout, stderr = await process.communicate(input=text.encode())

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise RuntimeError(f"Piper TTS failed: {error_msg}")

            # Play the audio using platform-appropriate player
            player_cmd, player_args, uses_placeholder = self._get_audio_player()

            if uses_placeholder:
                # Windows PowerShell style - replace {file} placeholder
                args = [arg.replace("{file}", temp_path) for arg in player_args]
                play_process = await asyncio.create_subprocess_exec(
                    player_cmd, *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
            else:
                # Unix style - file path is last argument
                play_process = await asyncio.create_subprocess_exec(
                    player_cmd, *player_args, temp_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

            await play_process.wait()

        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

    def speak_sync(self, text: str) -> None:
        """Synchronous version of speak for testing.

        Args:
            text: The text to speak.
        """
        asyncio.run(self.speak(text))


def test_tts():
    """Test the TTS functionality."""
    tts = PiperTTS()
    tts.speak_sync("Hello! I am your coding companion.")


if __name__ == "__main__":
    test_tts()
