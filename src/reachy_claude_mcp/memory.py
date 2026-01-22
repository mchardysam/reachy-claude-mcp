"""Memory module for persistent state across sessions and projects."""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict

from .config import get_memory_path, QDRANT_HOST, QDRANT_PORT
from .llm_analyzer import LLMAnalyzer, MLX_AVAILABLE
from .database import ProjectDatabase, Project, Session
from .vector_store import VectorStore, SemanticMemory, QDRANT_AVAILABLE


# Sentiment keywords for classification
SENTIMENT_PATTERNS = {
    "error": ["error", "failed", "exception", "broken", "bug", "crash", "traceback", "cannot", "unable"],
    "success": ["success", "passed", "done", "fixed", "complete", "works", "solved", "resolved", "finished"],
    "thinking": ["looking", "searching", "let me", "checking", "analyzing", "reading", "exploring", "investigating"],
    "question": ["?", "how do", "what is", "where", "why", "can you", "could you"],
    "greeting": ["hello", "hi ", "hey", "good morning", "good afternoon", "good evening"],
    "farewell": ["bye", "goodbye", "see you", "signing off", "done for", "that's all"],
}

# Map sentiments to emotions
SENTIMENT_TO_EMOTION = {
    "error": "oops",
    "success": "celebrate",
    "thinking": "thinking",
    "question": "curious",
    "greeting": "welcoming",
    "farewell": "sleep",
    "neutral": "neutral",
}


@dataclass
class SessionState:
    """State for the current session."""
    project_path: str | None = None
    error_streak: int = 0
    success_streak: int = 0
    interaction_count: int = 0
    last_sentiment: str = "neutral"
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Memory:
    """Persistent memory across sessions."""
    total_interactions: int = 0
    total_errors: int = 0
    total_successes: int = 0
    projects_seen: list[str] = field(default_factory=list)
    last_session: str | None = None
    favorite_emotion: str = "happy"

    # Current session (not persisted)
    session: SessionState = field(default_factory=SessionState)

    def to_dict(self) -> dict:
        """Convert to dict for serialization (excludes session)."""
        d = asdict(self)
        del d["session"]  # Don't persist current session
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Memory":
        """Create from dict."""
        d.pop("session", None)  # Remove session if present
        return cls(**d)


class MemoryManager:
    """Manages persistent memory for Reachy."""

    def __init__(
        self,
        memory_path: Path | None = None,
        use_llm: bool = True,
        use_vector_store: bool = True,
        qdrant_host: str | None = None,
        qdrant_port: int | None = None,
    ):
        if memory_path is None:
            memory_path = get_memory_path()
        self.memory_path = memory_path

        # Use config defaults if not specified
        qdrant_host = qdrant_host or QDRANT_HOST
        qdrant_port = qdrant_port or QDRANT_PORT
        self.memory = self._load()

        # Initialize LLM analyzer if available and requested
        self.use_llm = use_llm and MLX_AVAILABLE
        self._llm: LLMAnalyzer | None = None
        if self.use_llm:
            self._llm = LLMAnalyzer()

        # Initialize project database (SQLite)
        self._db = ProjectDatabase()

        # Initialize vector store (Qdrant) if available and requested
        self.use_vector_store = use_vector_store and QDRANT_AVAILABLE
        self._vector_store: VectorStore | None = None
        if self.use_vector_store:
            self._vector_store = VectorStore(host=qdrant_host, port=qdrant_port)

        # Current project and session tracking
        self._current_project: Project | None = None
        self._current_session: Session | None = None

    def _load(self) -> Memory:
        """Load memory from disk."""
        if self.memory_path.exists():
            try:
                with open(self.memory_path) as f:
                    data = json.load(f)
                return Memory.from_dict(data)
            except (json.JSONDecodeError, TypeError):
                pass
        return Memory()

    def save(self) -> None:
        """Save memory to disk."""
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_path, "w") as f:
            json.dump(self.memory.to_dict(), f, indent=2)

    def classify_sentiment(self, text: str) -> str:
        """Classify the sentiment of text using LLM or keyword fallback."""
        # Use LLM if available
        if self._llm is not None:
            return self._llm.classify_sentiment(text)

        # Fallback to keyword matching
        text_lower = text.lower()
        for sentiment, patterns in SENTIMENT_PATTERNS.items():
            if any(p in text_lower for p in patterns):
                return sentiment

        return "neutral"

    def get_emotion_for_sentiment(self, sentiment: str) -> str:
        """Get the appropriate emotion for a sentiment."""
        base_emotion = SENTIMENT_TO_EMOTION.get(sentiment, "neutral")

        # Modify based on streaks
        session = self.memory.session

        if sentiment == "error":
            session.error_streak += 1
            session.success_streak = 0
            if session.error_streak >= 3:
                return "frustrated"  # Multiple errors in a row
        elif sentiment == "success":
            session.success_streak += 1
            session.error_streak = 0
            if session.success_streak >= 3:
                return "excited"  # On a roll!

        return base_emotion

    def should_speak(self, sentiment: str) -> bool:
        """Decide if Reachy should speak for this interaction."""
        session = self.memory.session

        # Always speak for errors, successes, greetings, farewells
        if sentiment in ["error", "success", "greeting", "farewell"]:
            return True

        # Speak occasionally for thinking/questions (every 3rd time)
        if sentiment in ["thinking", "question"]:
            return session.interaction_count % 3 == 0

        # Neutral - speak rarely (every 5th interaction)
        return session.interaction_count % 5 == 0

    def generate_summary(self, text: str, sentiment: str) -> str:
        """Generate a short spoken summary based on context.

        Args:
            text: The original text to summarize
            sentiment: The classified sentiment
        """
        session = self.memory.session
        context = {
            "error_streak": session.error_streak,
            "success_streak": session.success_streak,
        }

        # Use LLM if available
        if self._llm is not None:
            return self._llm.generate_summary(text, sentiment, context)

        # Fallback to template responses
        if sentiment == "error":
            if session.error_streak >= 3:
                return "Hmm, we're hitting a few bumps. Let's take it step by step."
            return "Oops! Let me help fix that."

        if sentiment == "success":
            if session.success_streak >= 3:
                return "We're on fire! Great work!"
            return "Nice! That worked!"

        if sentiment == "greeting":
            hour = datetime.now().hour
            if hour < 12:
                return "Good morning! Ready to code!"
            elif hour < 17:
                return "Good afternoon! Let's build something!"
            else:
                return "Good evening! Let's get productive!"

        if sentiment == "farewell":
            return "Great session! See you next time!"

        if sentiment == "thinking":
            return "Let me look into that..."

        if sentiment == "question":
            return "Interesting question!"

        return "Got it!"

    def record_interaction(self, sentiment: str, project: str | None = None) -> None:
        """Record an interaction for statistics."""
        self.memory.total_interactions += 1
        self.memory.session.interaction_count += 1
        self.memory.session.last_sentiment = sentiment

        if sentiment == "error":
            self.memory.total_errors += 1
        elif sentiment == "success":
            self.memory.total_successes += 1

        if project and project not in self.memory.projects_seen:
            self.memory.projects_seen.append(project)

        self.memory.session.project_path = project
        self.memory.last_session = datetime.now().isoformat()

        # Save periodically
        if self.memory.total_interactions % 10 == 0:
            self.save()

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "total_interactions": self.memory.total_interactions,
            "total_errors": self.memory.total_errors,
            "total_successes": self.memory.total_successes,
            "success_rate": (
                self.memory.total_successes / max(1, self.memory.total_successes + self.memory.total_errors)
            ),
            "projects_seen": len(self.memory.projects_seen),
            "session_interactions": self.memory.session.interaction_count,
            "current_error_streak": self.memory.session.error_streak,
            "current_success_streak": self.memory.session.success_streak,
        }

    # Project-aware methods

    def set_project(self, project_path: str) -> Project:
        """Set the current project and start a session.

        Args:
            project_path: Path to the project directory

        Returns:
            The Project object
        """
        # Get or create project in database
        self._current_project = self._db.get_or_create_project(project_path)

        # Start a new session
        if self._current_project.id is not None:
            self._current_session = self._db.start_session(self._current_project.id)

        return self._current_project

    def get_current_project(self) -> Project | None:
        """Get the current project."""
        return self._current_project

    def end_session(self, summary: str | None = None) -> None:
        """End the current session.

        Args:
            summary: Optional summary of what was accomplished
        """
        if self._current_session is None or self._current_session.id is None:
            return

        # Determine outcome based on session stats
        session = self.memory.session
        if session.error_streak > session.success_streak:
            outcome = "error"
        elif session.success_streak > 0:
            outcome = "success"
        else:
            outcome = "neutral"

        self._db.end_session(self._current_session.id, summary=summary, outcome=outcome)
        self._current_session = None

    def record_interaction_full(
        self,
        output: str,
        sentiment: str,
        reachy_response: str | None = None,
        project_path: str | None = None,
    ) -> None:
        """Record an interaction with full context.

        Stores in both SQLite (structured) and Qdrant (semantic).

        Args:
            output: The output text from Claude
            sentiment: Classified sentiment
            reachy_response: What Reachy said in response
            project_path: Optional project path (uses current if not provided)
        """
        # Set project if provided and different
        if project_path and (
            self._current_project is None or
            self._current_project.path != project_path
        ):
            self.set_project(project_path)

        # Record basic stats
        self.record_interaction(sentiment, project_path)

        # Record in SQLite
        if self._current_session and self._current_session.id:
            self._db.record_interaction(
                session_id=self._current_session.id,
                sentiment=sentiment,
                output_summary=output[:500] if output else None,
                reachy_response=reachy_response,
            )

            # Update session stats
            successes = 1 if sentiment == "success" else 0
            errors = 1 if sentiment == "error" else 0
            self._db.update_session_stats(
                self._current_session.id,
                interactions=1,
                successes=successes,
                errors=errors,
            )

            # Update project stats
            if self._current_project and self._current_project.id:
                self._db.update_project_stats(
                    self._current_project.id,
                    successes=successes,
                    errors=errors,
                )

        # Record in vector store for semantic search
        if self._vector_store and self._current_project and self._current_session:
            self._vector_store.store_interaction(
                project_id=self._current_project.id or 0,
                session_id=self._current_session.id or 0,
                content=output,
                sentiment=sentiment,
                reachy_response=reachy_response,
            )

    def store_problem_solution(
        self,
        problem: str,
        solution: str,
        tags: list[str] | None = None,
    ) -> None:
        """Store a problem-solution pair for future reference.

        Args:
            problem: Description of the problem
            solution: How it was solved
            tags: Optional tags for categorization
        """
        if not self._vector_store:
            return

        project_id = self._current_project.id if self._current_project else 0
        session_id = self._current_session.id if self._current_session else None

        self._vector_store.store_problem_solution(
            project_id=project_id or 0,
            session_id=session_id,
            problem=problem,
            solution=solution,
            tags=tags,
        )

    def find_similar_problems(
        self,
        query: str,
        current_project_only: bool = False,
        limit: int = 5,
    ) -> list[SemanticMemory]:
        """Find similar problems from past sessions.

        Args:
            query: The problem to search for
            current_project_only: If True, only search current project
            limit: Max results to return

        Returns:
            List of similar problem-solution memories
        """
        if not self._vector_store:
            return []

        project_id = None
        if current_project_only and self._current_project:
            project_id = self._current_project.id

        return self._vector_store.search_similar_problems(
            query=query,
            project_id=project_id,
            limit=limit,
        )

    def find_related_across_projects(
        self,
        query: str,
        limit: int = 5,
    ) -> list[SemanticMemory]:
        """Find related content from OTHER projects.

        Useful for "you solved something similar in project X" suggestions.
        """
        if not self._vector_store:
            return []

        exclude_id = self._current_project.id if self._current_project else None

        return self._vector_store.find_related_across_projects(
            query=query,
            exclude_project_id=exclude_id,
            limit=limit,
        )

    def get_project_greeting(self, project_path: str) -> str:
        """Generate a context-aware greeting for a project.

        Args:
            project_path: Path to the project

        Returns:
            A personalized greeting based on project history
        """
        project = self._db.get_project_by_path(project_path)

        if project is None:
            # New project
            name = Path(project_path).name
            return f"New project! Welcome to {name}. Let's build something great!"

        # Returning to existing project
        stats = self._db.get_project_stats(project.id)
        sessions = self._db.get_project_sessions(project.id, limit=1)

        name = project.name
        session_count = stats.get("session_count", 0)

        if session_count == 0:
            return f"Back to {name}! Ready to continue?"

        # Get time since last session
        if sessions:
            last_session = sessions[0]
            if last_session.ended_at:
                last_time = datetime.fromisoformat(last_session.ended_at)
                delta = datetime.now() - last_time

                if delta.days > 7:
                    return f"Welcome back to {name}! It's been a while - {delta.days} days."
                elif delta.days > 0:
                    return f"Back to {name}! Last worked on this {delta.days} days ago."

        success_rate = stats.get("success_rate", 0.5)
        if success_rate > 0.7:
            return f"Back to {name}! This project's been going well - {success_rate:.0%} success rate!"
        elif success_rate < 0.3:
            return f"Back to {name}. Let's tackle those issues together!"

        return f"Welcome back to {name}! Session #{session_count + 1}."

    def link_projects(
        self,
        project_a_path: str,
        project_b_path: str,
        link_type: str,
        description: str | None = None,
    ) -> None:
        """Create a relationship between two projects.

        Args:
            project_a_path: First project path
            project_b_path: Second project path
            link_type: Type of relationship (dependency, related, fork, shared_code)
            description: Optional description of the relationship
        """
        project_a = self._db.get_or_create_project(project_a_path)
        project_b = self._db.get_or_create_project(project_b_path)

        if project_a.id and project_b.id:
            self._db.link_projects(
                project_a.id, project_b.id,
                link_type=link_type,
                description=description,
            )

    def get_linked_projects(self, project_path: str | None = None) -> list[tuple[Project, str, str]]:
        """Get projects linked to the specified or current project.

        Returns:
            List of (project, link_type, description) tuples
        """
        if project_path:
            project = self._db.get_project_by_path(project_path)
        else:
            project = self._current_project

        if not project or not project.id:
            return []

        return self._db.get_linked_projects(project.id)

    def get_all_projects(self) -> list[Project]:
        """Get all known projects."""
        return self._db.get_all_projects()

    def get_recent_projects(self, limit: int = 10) -> list[Project]:
        """Get recently accessed projects."""
        return self._db.get_recent_projects(limit=limit)
