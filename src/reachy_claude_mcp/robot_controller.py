"""Reachy Mini robot controller for playing emotions and animations."""

import asyncio
import os
import subprocess
import time
from typing import Literal

import psutil
from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves


# Mapping from simple dance keywords to library dance names
DANCE_MAPPING: dict[str, str] = {
    # Celebrations
    "celebrate": "groovy_sway_and_roll",
    "victory": "jackson_square",
    "playful": "chicken_peck",
    "party": "polyrhythm_combo",

    # Acknowledgments
    "nod": "simple_nod",
    "agree": "yeah_nod",
    "listening": "uh_huh_tilt",
    "acknowledge": "yeah_nod",

    # Reactions
    "mind_blown": "dizzy_spin",
    "recovered": "stumble_and_recover",
    "fixed_it": "stumble_and_recover",
    "whoa": "dizzy_spin",

    # Subtle/Idle movements
    "idle": "side_to_side_sway",
    "processing": "pendulum_swing",
    "waiting": "side_to_side_sway",
    "thinking_dance": "head_tilt_roll",

    # Expressive
    "peek": "side_peekaboo",
    "glance": "side_glance_flick",
    "sharp": "sharp_side_tilt",
    "funky": "grid_snap",
    "smooth": "chin_lead",
    "spiral": "interwoven_spirals",
    "recoil": "neck_recoil",
}


# Mapping from simple emotion keywords to library emotion names
EMOTION_MAPPING: dict[str, str] = {
    # Positive emotions
    "happy": "cheerful1",
    "excited": "enthusiastic1",
    "proud": "proud1",
    "grateful": "grateful1",
    "relieved": "relief1",
    "loving": "loving1",
    "serene": "serenity1",
    "calm": "calming1",
    "welcoming": "welcoming1",

    # Success/accomplishment
    "success": "success1",
    "celebrate": "success2",
    "yes": "yes1",
    "done": "success1",

    # Thinking/curiosity
    "thinking": "thoughtful1",
    "curious": "curious1",
    "attentive": "attentive1",
    "inquiring": "inquiring1",
    "understanding": "understanding1",

    # Confusion/uncertainty
    "confused": "confused1",
    "uncertain": "uncertain1",
    "lost": "lost1",
    "oops": "oops1",

    # Negative emotions
    "sad": "sad1",
    "tired": "tired1",
    "bored": "boredom1",
    "frustrated": "frustrated1",
    "anxious": "anxiety1",
    "scared": "scared1",
    "angry": "furious1",
    "irritated": "irritated1",
    "disgusted": "disgusted1",
    "lonely": "lonely1",
    "exhausted": "exhausted1",

    # Surprise/amazement
    "surprised": "surprised1",
    "amazed": "amazed1",

    # Actions
    "no": "no1",
    "come": "come1",
    "go_away": "go_away1",
    "sleep": "sleep1",
    "wake_up": "welcoming1",
    "laugh": "laughing1",
    "shy": "shy1",
    "helpful": "helpful1",

    # Neutral/default
    "neutral": "attentive1",
    "default": "attentive1",
}


class RobotController:
    """Controller for Reachy Mini robot with emotion playback."""

    def __init__(
        self,
        host: str = "localhost",
        connection_mode: Literal["auto", "localhost_only", "network"] = "localhost_only",
        spawn_daemon: bool = False,
        use_sim: bool = False,
        scene: str = "minimal",
    ):
        """Initialize the robot controller.

        Args:
            host: Hostname of the robot (for future remote control).
            connection_mode: How to connect to the daemon.
            spawn_daemon: Whether to spawn a daemon if none is running.
            use_sim: Whether to use simulation mode.
            scene: Scene to load in simulation mode ("empty" or "minimal").
        """
        self._host = host
        self._connection_mode = connection_mode
        self._spawn_daemon = spawn_daemon
        self._use_sim = use_sim
        self._scene = scene
        self._mini: ReachyMini | None = None
        self._emotions: RecordedMoves | None = None
        self._dances: RecordedMoves | None = None
        self._connected = False

    def _spawn_daemon_with_scene(self) -> None:
        """Spawn the daemon with the configured scene parameter."""
        # Check if daemon is already running
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                safe_cmdline = proc.info.get("cmdline") or []
                if any("reachy-mini-daemon" in cmd for cmd in safe_cmdline):
                    # Daemon already running, kill it to restart with new scene
                    os.kill(proc.pid, 9)
                    time.sleep(1)
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        # Build daemon command with scene parameter
        cmd = ["reachy-mini-daemon", "--scene", self._scene]
        if self._use_sim:
            cmd.append("--sim")

        subprocess.Popen(cmd, start_new_session=True)
        time.sleep(2)  # Give daemon time to start

    def connect(self) -> None:
        """Connect to the robot and load emotions library."""
        if self._connected:
            return

        # Spawn daemon ourselves if requested (to pass scene parameter)
        if self._spawn_daemon:
            self._spawn_daemon_with_scene()

        self._mini = ReachyMini(
            connection_mode=self._connection_mode,
            spawn_daemon=False,  # We handle spawning ourselves
            use_sim=self._use_sim,
            media_backend="no_media",  # Skip camera initialization
        )

        # Load the emotions library
        self._emotions = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")

        # Load the dances library
        self._dances = RecordedMoves("pollen-robotics/reachy-mini-dances-library")

        self._connected = True

    def disconnect(self) -> None:
        """Disconnect from the robot."""
        if self._mini is not None:
            del self._mini
            self._mini = None
        self._emotions = None
        self._dances = None
        self._connected = False

    def is_connected(self) -> bool:
        """Check if connected to the robot."""
        return self._connected

    def list_emotions(self) -> list[str]:
        """Return available emotion keywords.

        Returns:
            List of simple emotion keywords that can be used with play_emotion.
        """
        return sorted(EMOTION_MAPPING.keys())

    def list_raw_emotions(self) -> list[str]:
        """Return all raw emotion names from the library.

        Returns:
            List of actual emotion animation names.
        """
        if self._emotions is None:
            self.connect()
        return sorted(self._emotions.list_moves()) if self._emotions else []

    def _resolve_emotion(self, emotion: str) -> str:
        """Resolve an emotion keyword to a library emotion name.

        Args:
            emotion: Either a simple keyword or a raw library name.

        Returns:
            The library emotion name.
        """
        # First check if it's a simple keyword
        if emotion.lower() in EMOTION_MAPPING:
            return EMOTION_MAPPING[emotion.lower()]

        # Otherwise, assume it's a raw emotion name
        return emotion

    async def play_emotion(self, emotion: str, sound: bool = True) -> str:
        """Play an emotion animation on the robot.

        Args:
            emotion: Emotion keyword or raw library name.
            sound: Whether to play the sound associated with the move.

        Returns:
            The name of the emotion that was played.
        """
        if not self._connected:
            self.connect()

        resolved_emotion = self._resolve_emotion(emotion)

        if self._emotions is None:
            raise RuntimeError("Emotions library not loaded")

        try:
            move = self._emotions.get(resolved_emotion)
        except KeyError as e:
            available = ", ".join(sorted(EMOTION_MAPPING.keys())[:10])
            raise ValueError(
                f"Unknown emotion: {emotion} (resolved to {resolved_emotion}). "
                f"Available keywords: {available}..."
            ) from e

        if self._mini is None:
            raise RuntimeError("Not connected to robot")

        await self._mini.async_play_move(move, sound=sound)
        return resolved_emotion

    def play_emotion_sync(self, emotion: str, sound: bool = True) -> str:
        """Synchronous version of play_emotion for testing.

        Args:
            emotion: Emotion keyword or raw library name.
            sound: Whether to play the sound.

        Returns:
            The name of the emotion that was played.
        """
        return asyncio.run(self.play_emotion(emotion, sound=sound))

    def list_dances(self) -> list[str]:
        """Return available dance keywords.

        Returns:
            List of simple dance keywords that can be used with play_dance.
        """
        return sorted(DANCE_MAPPING.keys())

    def list_raw_dances(self) -> list[str]:
        """Return all raw dance names from the library.

        Returns:
            List of actual dance animation names.
        """
        if self._dances is None:
            self.connect()
        return sorted(self._dances.list_moves()) if self._dances else []

    def _resolve_dance(self, dance: str) -> str:
        """Resolve a dance keyword to a library dance name.

        Args:
            dance: Either a simple keyword or a raw library name.

        Returns:
            The library dance name.
        """
        # First check if it's a simple keyword
        if dance.lower() in DANCE_MAPPING:
            return DANCE_MAPPING[dance.lower()]

        # Otherwise, assume it's a raw dance name
        return dance

    async def play_dance(self, dance: str, sound: bool = True) -> str:
        """Play a dance animation on the robot.

        Args:
            dance: Dance keyword or raw library name.
            sound: Whether to play the sound associated with the move.

        Returns:
            The name of the dance that was played.
        """
        if not self._connected:
            self.connect()

        resolved_dance = self._resolve_dance(dance)

        if self._dances is None:
            raise RuntimeError("Dances library not loaded")

        try:
            move = self._dances.get(resolved_dance)
        except KeyError as e:
            available = ", ".join(sorted(DANCE_MAPPING.keys())[:10])
            raise ValueError(
                f"Unknown dance: {dance} (resolved to {resolved_dance}). "
                f"Available keywords: {available}..."
            ) from e

        if self._mini is None:
            raise RuntimeError("Not connected to robot")

        await self._mini.async_play_move(move, sound=sound)
        return resolved_dance

    def play_dance_sync(self, dance: str, sound: bool = True) -> str:
        """Synchronous version of play_dance for testing.

        Args:
            dance: Dance keyword or raw library name.
            sound: Whether to play the sound.

        Returns:
            The name of the dance that was played.
        """
        return asyncio.run(self.play_dance(dance, sound=sound))


def test_controller():
    """Test the robot controller."""
    controller = RobotController(spawn_daemon=True, use_sim=True)
    print("Available emotions:", controller.list_emotions())
    controller.connect()
    print("Connected to robot")
    print("Playing 'happy' emotion...")
    result = controller.play_emotion_sync("happy")
    print(f"Played: {result}")
    controller.disconnect()


if __name__ == "__main__":
    test_controller()
