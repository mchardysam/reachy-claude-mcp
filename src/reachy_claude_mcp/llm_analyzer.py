"""LLM analyzer for sentiment analysis and summarization.

Uses pluggable backends (MLX, Ollama) with automatic fallback to keyword matching.
"""

from datetime import datetime
from typing import Literal

from .llm_backends import LLMBackend, get_best_backend


SentimentType = Literal["error", "success", "thinking", "question", "greeting", "farewell", "neutral"]


class LLMAnalyzer:
    """Analyzer using LLM backends for sentiment and summarization."""

    def __init__(self):
        self._backend: LLMBackend | None = None
        self._backend_checked = False

    def _get_backend(self) -> LLMBackend | None:
        """Get the best available backend (cached)."""
        if not self._backend_checked:
            self._backend = get_best_backend()
            self._backend_checked = True
            if self._backend:
                print(f"Using LLM backend: {self._backend.name}")
        return self._backend

    @property
    def available(self) -> bool:
        """Check if any LLM backend is available."""
        return self._get_backend() is not None

    @property
    def backend_name(self) -> str | None:
        """Get the name of the active backend."""
        backend = self._get_backend()
        return backend.name if backend else None

    def classify_sentiment(self, text: str) -> str:
        """Classify the sentiment of text.

        Args:
            text: The text to analyze

        Returns:
            One of: error, success, thinking, question, greeting, farewell, neutral
        """
        backend = self._get_backend()
        if backend is None:
            return "neutral"

        prompt = self._build_sentiment_prompt(text)
        response = backend.generate(prompt, max_tokens=10).lower().strip()

        # Parse response - extract just the category
        valid_sentiments = ["error", "success", "thinking", "question", "greeting", "farewell", "neutral"]
        for sentiment in valid_sentiments:
            if sentiment in response:
                return sentiment

        return "neutral"

    def generate_summary(self, text: str, sentiment: str, context: dict | None = None) -> str:
        """Generate a short spoken summary for Reachy to say.

        Args:
            text: The original text to summarize
            sentiment: The classified sentiment
            context: Optional context dict with error_streak, success_streak, etc.

        Returns:
            A short 1-2 sentence summary suitable for speech
        """
        backend = self._get_backend()
        if backend is None:
            return self._fallback_summary(sentiment, context)

        prompt = self._build_summary_prompt(text, sentiment, context)
        response = backend.generate(prompt, max_tokens=30)

        # Clean up response
        response = response.strip().strip('"').strip("'")

        # Ensure it's not too long
        if len(response) > 100:
            response = response[:100].rsplit(" ", 1)[0] + "..."

        # Fallback if response is empty or weird
        if not response or len(response) < 5:
            return self._fallback_summary(sentiment, context)

        return response

    def _build_sentiment_prompt(self, text: str) -> str:
        """Build the sentiment classification prompt."""
        return f"""<|im_start|>system
You are a sentiment classifier for coding assistant responses. Classify the sentiment into exactly one category.
Categories: error, success, thinking, question, greeting, farewell, neutral

Rules:
- error: errors, exceptions, failures, bugs, crashes
- success: completed tasks, passed tests, fixed issues, working code
- thinking: searching, analyzing, looking into something, reading files
- question: asking questions, seeking clarification
- greeting: hello, hi, starting a session
- farewell: goodbye, ending session, signing off
- neutral: general information, explanations, code without clear sentiment

Respond with ONLY the category name, nothing else.<|im_end|>
<|im_start|>user
Classify this text: {text[:500]}<|im_end|>
<|im_start|>assistant
"""

    def _build_summary_prompt(self, text: str, sentiment: str, context: dict | None = None) -> str:
        """Build the summary generation prompt."""
        context = context or {}
        error_streak = context.get("error_streak", 0)
        success_streak = context.get("success_streak", 0)

        streak_context = ""
        if error_streak >= 3:
            streak_context = "Note: This is the 3rd+ error in a row, be encouraging."
        elif success_streak >= 3:
            streak_context = "Note: This is the 3rd+ success in a row, be extra celebratory!"

        return f"""<|im_start|>system
You are Reachy, a friendly robot coding companion. Generate a SHORT spoken response (1 sentence, max 15 words) to react to the user's coding activity.

Personality: Friendly, supportive, slightly playful. Use simple language suitable for speech synthesis.
{streak_context}

Current sentiment detected: {sentiment}
DO NOT explain what you're doing. Just give the spoken response.<|im_end|>
<|im_start|>user
React to this: {text[:300]}<|im_end|>
<|im_start|>assistant
"""

    def _fallback_summary(self, sentiment: str, context: dict | None = None) -> str:
        """Fallback summaries when no LLM backend is available."""
        context = context or {}
        error_streak = context.get("error_streak", 0)
        success_streak = context.get("success_streak", 0)

        summaries = {
            "error": "Hmm, we hit a snag. Let's fix it!" if error_streak < 3 else "Okay, let's take this step by step.",
            "success": "Nice work!" if success_streak < 3 else "We're on fire! Great job!",
            "thinking": "Let me look into that...",
            "question": "Good question!",
            "greeting": self._time_based_greeting(),
            "farewell": "Great session! See you next time!",
            "neutral": "Got it!",
        }
        return summaries.get(sentiment, "Got it!")

    def _time_based_greeting(self) -> str:
        """Get a time-appropriate greeting."""
        hour = datetime.now().hour
        if hour < 12:
            return "Good morning! Ready to code!"
        elif hour < 17:
            return "Good afternoon! Let's build something!"
        else:
            return "Good evening! Let's get productive!"


# Singleton instance
_analyzer: LLMAnalyzer | None = None


def get_analyzer() -> LLMAnalyzer:
    """Get or create the LLM analyzer singleton."""
    global _analyzer
    if _analyzer is None:
        _analyzer = LLMAnalyzer()
    return _analyzer


# For backwards compatibility - expose whether any LLM is available
def _check_mlx_available() -> bool:
    """Check if MLX is available (for backwards compat)."""
    try:
        from mlx_lm import load  # noqa: F401
        return True
    except ImportError:
        return False


MLX_AVAILABLE = _check_mlx_available()
