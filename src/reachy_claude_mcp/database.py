"""SQLite database for structured project memory."""

import sqlite3
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass

from .config import get_database_path, ensure_directories


@dataclass
class Project:
    """A project that Reachy has worked on."""
    id: int | None
    path: str
    name: str
    description: str | None = None
    first_seen: str | None = None
    last_accessed: str | None = None
    total_sessions: int = 0
    total_successes: int = 0
    total_errors: int = 0


@dataclass
class Session:
    """A coding session within a project."""
    id: int | None
    project_id: int
    started_at: str
    ended_at: str | None = None
    summary: str | None = None
    outcome: str | None = None  # success, error, neutral
    interaction_count: int = 0
    success_count: int = 0
    error_count: int = 0


@dataclass
class ProjectLink:
    """A relationship between two projects."""
    id: int | None
    project_a_id: int
    project_b_id: int
    link_type: str  # dependency, related, fork, shared_code
    description: str | None = None
    created_at: str | None = None


class ProjectDatabase:
    """SQLite database for project memory."""

    def __init__(self, db_path: Path | None = None):
        if db_path is None:
            db_path = get_database_path()
        ensure_directories()
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _get_conn(self):
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        """Initialize the database schema."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    first_seen TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    total_sessions INTEGER DEFAULT 0,
                    total_successes INTEGER DEFAULT 0,
                    total_errors INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    summary TEXT,
                    outcome TEXT,
                    interaction_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                );

                CREATE TABLE IF NOT EXISTS project_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_a_id INTEGER NOT NULL,
                    project_b_id INTEGER NOT NULL,
                    link_type TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (project_a_id) REFERENCES projects(id),
                    FOREIGN KEY (project_b_id) REFERENCES projects(id),
                    UNIQUE(project_a_id, project_b_id, link_type)
                );

                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    sentiment TEXT NOT NULL,
                    output_summary TEXT,
                    reachy_response TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id);
                CREATE INDEX IF NOT EXISTS idx_interactions_session ON interactions(session_id);
                CREATE INDEX IF NOT EXISTS idx_projects_path ON projects(path);
            """)

    # Project methods
    def get_or_create_project(self, path: str, name: str | None = None) -> Project:
        """Get existing project or create new one."""
        now = datetime.now().isoformat()

        if name is None:
            name = Path(path).name

        with self._get_conn() as conn:
            # Try to get existing
            row = conn.execute(
                "SELECT * FROM projects WHERE path = ?", (path,)
            ).fetchone()

            if row:
                # Update last_accessed
                conn.execute(
                    "UPDATE projects SET last_accessed = ? WHERE id = ?",
                    (now, row["id"])
                )
                return Project(
                    id=row["id"],
                    path=row["path"],
                    name=row["name"],
                    description=row["description"],
                    first_seen=row["first_seen"],
                    last_accessed=now,
                    total_sessions=row["total_sessions"],
                    total_successes=row["total_successes"],
                    total_errors=row["total_errors"],
                )

            # Create new
            cursor = conn.execute(
                """INSERT INTO projects (path, name, first_seen, last_accessed)
                   VALUES (?, ?, ?, ?)""",
                (path, name, now, now)
            )
            return Project(
                id=cursor.lastrowid,
                path=path,
                name=name,
                first_seen=now,
                last_accessed=now,
            )

    def update_project_stats(self, project_id: int, successes: int = 0, errors: int = 0):
        """Update project statistics."""
        with self._get_conn() as conn:
            conn.execute(
                """UPDATE projects
                   SET total_successes = total_successes + ?,
                       total_errors = total_errors + ?
                   WHERE id = ?""",
                (successes, errors, project_id)
            )

    def update_project_description(self, project_id: int, description: str):
        """Update project description."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE projects SET description = ? WHERE id = ?",
                (description, project_id)
            )

    def get_project_by_path(self, path: str) -> Project | None:
        """Get a project by its path."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM projects WHERE path = ?", (path,)
            ).fetchone()
            if row:
                return Project(**dict(row))
        return None

    def get_recent_projects(self, limit: int = 10) -> list[Project]:
        """Get recently accessed projects."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM projects
                   ORDER BY last_accessed DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()
            return [Project(**dict(row)) for row in rows]

    def get_all_projects(self) -> list[Project]:
        """Get all projects."""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM projects").fetchall()
            return [Project(**dict(row)) for row in rows]

    # Session methods
    def start_session(self, project_id: int) -> Session:
        """Start a new session for a project."""
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            # Increment session count
            conn.execute(
                "UPDATE projects SET total_sessions = total_sessions + 1 WHERE id = ?",
                (project_id,)
            )
            cursor = conn.execute(
                "INSERT INTO sessions (project_id, started_at) VALUES (?, ?)",
                (project_id, now)
            )
            return Session(
                id=cursor.lastrowid,
                project_id=project_id,
                started_at=now,
            )

    def end_session(self, session_id: int, summary: str | None = None, outcome: str | None = None):
        """End a session."""
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """UPDATE sessions
                   SET ended_at = ?, summary = ?, outcome = ?
                   WHERE id = ?""",
                (now, summary, outcome, session_id)
            )

    def update_session_stats(self, session_id: int, interactions: int = 0,
                            successes: int = 0, errors: int = 0):
        """Update session statistics."""
        with self._get_conn() as conn:
            conn.execute(
                """UPDATE sessions
                   SET interaction_count = interaction_count + ?,
                       success_count = success_count + ?,
                       error_count = error_count + ?
                   WHERE id = ?""",
                (interactions, successes, errors, session_id)
            )

    def get_project_sessions(self, project_id: int, limit: int = 10) -> list[Session]:
        """Get recent sessions for a project."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM sessions
                   WHERE project_id = ?
                   ORDER BY started_at DESC
                   LIMIT ?""",
                (project_id, limit)
            ).fetchall()
            return [Session(**dict(row)) for row in rows]

    # Interaction methods
    def record_interaction(self, session_id: int, sentiment: str,
                          output_summary: str | None = None,
                          reachy_response: str | None = None):
        """Record an interaction within a session."""
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO interactions
                   (session_id, timestamp, sentiment, output_summary, reachy_response)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, now, sentiment, output_summary, reachy_response)
            )

    # Project link methods
    def link_projects(self, project_a_id: int, project_b_id: int,
                     link_type: str, description: str | None = None):
        """Create a link between two projects."""
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            try:
                conn.execute(
                    """INSERT INTO project_links
                       (project_a_id, project_b_id, link_type, description, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (project_a_id, project_b_id, link_type, description, now)
                )
            except sqlite3.IntegrityError:
                # Link already exists, update description
                conn.execute(
                    """UPDATE project_links
                       SET description = ?
                       WHERE project_a_id = ? AND project_b_id = ? AND link_type = ?""",
                    (description, project_a_id, project_b_id, link_type)
                )

    def get_linked_projects(self, project_id: int) -> list[tuple[Project, str, str]]:
        """Get all projects linked to this one.

        Returns list of (project, link_type, description) tuples.
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT p.*, pl.link_type, pl.description as link_description
                   FROM projects p
                   JOIN project_links pl ON (
                       (pl.project_a_id = ? AND pl.project_b_id = p.id) OR
                       (pl.project_b_id = ? AND pl.project_a_id = p.id)
                   )""",
                (project_id, project_id)
            ).fetchall()

            results = []
            for row in rows:
                row_dict = dict(row)
                link_type = row_dict.pop("link_type")
                link_desc = row_dict.pop("link_description")
                project = Project(**row_dict)
                results.append((project, link_type, link_desc))
            return results

    # Query methods
    def search_projects(self, query: str) -> list[Project]:
        """Search projects by name or description."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM projects
                   WHERE name LIKE ? OR description LIKE ? OR path LIKE ?
                   ORDER BY last_accessed DESC""",
                (f"%{query}%", f"%{query}%", f"%{query}%")
            ).fetchall()
            return [Project(**dict(row)) for row in rows]

    def get_project_stats(self, project_id: int) -> dict:
        """Get detailed stats for a project."""
        with self._get_conn() as conn:
            project = conn.execute(
                "SELECT * FROM projects WHERE id = ?", (project_id,)
            ).fetchone()

            if not project:
                return {}

            sessions = conn.execute(
                """SELECT COUNT(*) as count,
                          SUM(interaction_count) as interactions,
                          SUM(success_count) as successes,
                          SUM(error_count) as errors
                   FROM sessions WHERE project_id = ?""",
                (project_id,)
            ).fetchone()

            recent_session = conn.execute(
                """SELECT * FROM sessions
                   WHERE project_id = ?
                   ORDER BY started_at DESC LIMIT 1""",
                (project_id,)
            ).fetchone()

            return {
                "project": Project(**dict(project)),
                "session_count": sessions["count"] or 0,
                "total_interactions": sessions["interactions"] or 0,
                "total_successes": sessions["successes"] or 0,
                "total_errors": sessions["errors"] or 0,
                "success_rate": (
                    (sessions["successes"] or 0) /
                    max(1, (sessions["successes"] or 0) + (sessions["errors"] or 0))
                ),
                "last_session": Session(**dict(recent_session)) if recent_session else None,
            }
