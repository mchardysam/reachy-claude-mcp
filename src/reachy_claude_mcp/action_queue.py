"""Action queue for serializing robot actions from multiple sessions."""

import asyncio
import fcntl
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .robot_controller import RobotController
    from .tts import PiperTTS

from .config import REACHY_HOME

logger = logging.getLogger(__name__)

# Cross-process lock file for serializing playback
LOCK_FILE = REACHY_HOME / "playback.lock"


class ActionType(Enum):
    """Types of robot actions."""
    EMOTION = "emotion"
    DANCE = "dance"
    SPEAK = "speak"
    EMOTION_AND_SPEAK = "emotion_and_speak"
    DANCE_AND_SPEAK = "dance_and_speak"


@dataclass
class Action:
    """A queued robot action."""
    action_type: ActionType
    # For emotion/dance actions
    animation_name: str | None = None
    # For speak actions
    text: str | None = None
    # Tracking
    session_id: str | None = None
    # Completion event for callers who want to wait
    _done_event: asyncio.Event = field(default_factory=asyncio.Event)
    # Result storage
    result: str | None = None
    error: Exception | None = None

    def mark_done(self, result: str | None = None, error: Exception | None = None) -> None:
        """Mark this action as completed."""
        self.result = result
        self.error = error
        self._done_event.set()

    async def wait(self) -> str | None:
        """Wait for this action to complete and return result."""
        await self._done_event.wait()
        if self.error:
            raise self.error
        return self.result


class ActionQueue:
    """Queue for serializing robot actions.

    Accepts actions from multiple sessions and processes them one at a time,
    ensuring animations and speech don't overlap.
    """

    def __init__(self) -> None:
        """Initialize the action queue."""
        self._queue: asyncio.Queue[Action] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._controller: "RobotController | None" = None
        self._tts: "PiperTTS | None" = None
        self._running = False
        self._current_action: Action | None = None

    def set_controller(self, controller: "RobotController") -> None:
        """Set the robot controller."""
        self._controller = controller

    def set_tts(self, tts: "PiperTTS") -> None:
        """Set the TTS instance."""
        self._tts = tts

    def start(self) -> None:
        """Start the queue worker."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("Action queue worker started")

    def stop(self) -> None:
        """Stop the queue worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            self._worker_task = None
        logger.info("Action queue worker stopped")

    @property
    def queue_size(self) -> int:
        """Return the number of pending actions."""
        return self._queue.qsize()

    @property
    def is_busy(self) -> bool:
        """Return True if currently processing an action."""
        return self._current_action is not None

    async def _worker(self) -> None:
        """Background worker that processes actions sequentially."""
        logger.info("Action queue worker running")
        while self._running:
            try:
                # Wait for next action with timeout to check _running flag
                try:
                    action = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                self._current_action = action
                try:
                    result = await self._process_action(action)
                    action.mark_done(result=result)
                except Exception as e:
                    logger.error(f"Error processing action: {e}")
                    action.mark_done(error=e)
                finally:
                    self._current_action = None
                    self._queue.task_done()

            except asyncio.CancelledError:
                logger.info("Action queue worker cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in action queue worker: {e}")

    async def _acquire_playback_lock(self):
        """Acquire cross-process lock for playback.

        Returns:
            File object for the lock file (must be closed to release).
        """
        # Ensure directory exists
        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Open lock file
        fd = open(LOCK_FILE, 'w')

        # Acquire exclusive lock (blocks until available)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: fcntl.flock(fd.fileno(), fcntl.LOCK_EX))
        logger.debug("Acquired playback lock")

        return fd

    def _release_playback_lock(self, fd) -> None:
        """Release the playback lock."""
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
            fd.close()
            logger.debug("Released playback lock")
        except Exception as e:
            logger.warning(f"Error releasing lock: {e}")

    async def _process_action(self, action: Action) -> str:
        """Process a single action.

        Args:
            action: The action to process.

        Returns:
            Result description string.
        """
        if self._controller is None:
            raise RuntimeError("Controller not set")

        # Acquire cross-process lock before any playback
        lock_fd = await self._acquire_playback_lock()

        try:
            return await self._process_action_locked(action)
        finally:
            self._release_playback_lock(lock_fd)

    async def _process_action_locked(self, action: Action) -> str:
        """Process action while holding the playback lock."""
        match action.action_type:
            case ActionType.EMOTION:
                if not action.animation_name:
                    raise ValueError("emotion action requires animation_name")
                result = await self._controller.play_emotion(action.animation_name)
                return f"Played emotion: {result}"

            case ActionType.DANCE:
                if not action.animation_name:
                    raise ValueError("dance action requires animation_name")
                result = await self._controller.play_dance(action.animation_name)
                return f"Performed dance: {result}"

            case ActionType.SPEAK:
                if self._tts is None:
                    raise RuntimeError("TTS not set")
                if not action.text:
                    raise ValueError("speak action requires text")
                await self._tts.speak(action.text)
                return f"Spoke: {action.text}"

            case ActionType.EMOTION_AND_SPEAK:
                if self._tts is None:
                    raise RuntimeError("TTS not set")
                if not action.animation_name or not action.text:
                    raise ValueError("emotion_and_speak requires animation_name and text")
                # Play emotion and speech concurrently (they complement each other)
                results = await asyncio.gather(
                    self._controller.play_emotion(action.animation_name),
                    self._tts.speak(action.text),
                    return_exceptions=True
                )
                emotion_result = results[0]
                if isinstance(emotion_result, Exception):
                    emotion_result = f"Error: {emotion_result}"
                return f"Played emotion '{emotion_result}' and spoke: {action.text}"

            case ActionType.DANCE_AND_SPEAK:
                if self._tts is None:
                    raise RuntimeError("TTS not set")
                if not action.animation_name or not action.text:
                    raise ValueError("dance_and_speak requires animation_name and text")
                # Play dance and speech concurrently
                results = await asyncio.gather(
                    self._controller.play_dance(action.animation_name),
                    self._tts.speak(action.text),
                    return_exceptions=True
                )
                dance_result = results[0]
                if isinstance(dance_result, Exception):
                    dance_result = f"Error: {dance_result}"
                return f"Performed dance '{dance_result}' and spoke: {action.text}"

        # Should never reach here since all ActionType variants are handled
        raise ValueError(f"Unhandled action type: {action.action_type}")  # pragma: no cover

    def enqueue_emotion(
        self,
        emotion: str,
        session_id: str | None = None,
    ) -> Action:
        """Enqueue an emotion animation.

        Args:
            emotion: Emotion keyword to play.
            session_id: Optional session identifier for tracking.

        Returns:
            The queued Action (can be awaited with action.wait()).
        """
        action = Action(
            action_type=ActionType.EMOTION,
            animation_name=emotion,
            session_id=session_id,
        )
        self._queue.put_nowait(action)
        logger.debug(f"Enqueued emotion: {emotion} (queue size: {self.queue_size})")
        return action

    def enqueue_dance(
        self,
        dance: str,
        session_id: str | None = None,
    ) -> Action:
        """Enqueue a dance animation.

        Args:
            dance: Dance keyword to play.
            session_id: Optional session identifier for tracking.

        Returns:
            The queued Action.
        """
        action = Action(
            action_type=ActionType.DANCE,
            animation_name=dance,
            session_id=session_id,
        )
        self._queue.put_nowait(action)
        logger.debug(f"Enqueued dance: {dance} (queue size: {self.queue_size})")
        return action

    def enqueue_speak(
        self,
        text: str,
        session_id: str | None = None,
    ) -> Action:
        """Enqueue speech.

        Args:
            text: Text to speak.
            session_id: Optional session identifier for tracking.

        Returns:
            The queued Action.
        """
        action = Action(
            action_type=ActionType.SPEAK,
            text=text,
            session_id=session_id,
        )
        self._queue.put_nowait(action)
        logger.debug(f"Enqueued speak (queue size: {self.queue_size})")
        return action

    def enqueue_emotion_and_speak(
        self,
        emotion: str,
        text: str,
        session_id: str | None = None,
    ) -> Action:
        """Enqueue an emotion animation with speech.

        The emotion and speech will play concurrently with each other,
        but not with other queued actions.

        Args:
            emotion: Emotion keyword to play.
            text: Text to speak.
            session_id: Optional session identifier for tracking.

        Returns:
            The queued Action.
        """
        action = Action(
            action_type=ActionType.EMOTION_AND_SPEAK,
            animation_name=emotion,
            text=text,
            session_id=session_id,
        )
        self._queue.put_nowait(action)
        logger.debug(f"Enqueued emotion+speak: {emotion} (queue size: {self.queue_size})")
        return action

    def enqueue_dance_and_speak(
        self,
        dance: str,
        text: str,
        session_id: str | None = None,
    ) -> Action:
        """Enqueue a dance animation with speech.

        The dance and speech will play concurrently with each other,
        but not with other queued actions.

        Args:
            dance: Dance keyword to play.
            text: Text to speak.
            session_id: Optional session identifier for tracking.

        Returns:
            The queued Action.
        """
        action = Action(
            action_type=ActionType.DANCE_AND_SPEAK,
            animation_name=dance,
            text=text,
            session_id=session_id,
        )
        self._queue.put_nowait(action)
        logger.debug(f"Enqueued dance+speak: {dance} (queue size: {self.queue_size})")
        return action


# Global queue instance
_action_queue: ActionQueue | None = None


def get_action_queue() -> ActionQueue:
    """Get or create the global action queue."""
    global _action_queue
    if _action_queue is None:
        _action_queue = ActionQueue()
    return _action_queue
