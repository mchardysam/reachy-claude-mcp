"""Qdrant vector store for semantic project memory."""

import hashlib
from datetime import datetime
from dataclasses import dataclass
from typing import Any

from .config import QDRANT_HOST, QDRANT_PORT

# Try to import Qdrant
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None

# Try to import sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None


@dataclass
class SemanticMemory:
    """A semantic memory entry."""
    id: str
    project_id: int
    session_id: int | None
    content: str
    memory_type: str  # interaction, problem, solution, summary
    sentiment: str | None
    timestamp: str
    metadata: dict[str, Any] | None = None
    score: float | None = None  # similarity score when retrieved


# Collection names
INTERACTIONS_COLLECTION = "reachy_interactions"
PROBLEMS_COLLECTION = "reachy_problems"

# Default embedding model (small, fast)
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2


class VectorStore:
    """Qdrant-based vector store for semantic memory."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.host = host or QDRANT_HOST
        self.port = port or QDRANT_PORT
        self._client: QdrantClient | None = None
        self._embedder: SentenceTransformer | None = None
        self._embedding_model = embedding_model
        self._initialized = False

    @property
    def available(self) -> bool:
        """Check if Qdrant and embeddings are available."""
        return QDRANT_AVAILABLE and EMBEDDINGS_AVAILABLE

    def _ensure_initialized(self) -> bool:
        """Initialize client and embedder if needed."""
        if self._initialized:
            return True

        if not self.available:
            return False

        try:
            # Connect to Qdrant
            self._client = QdrantClient(host=self.host, port=self.port)

            # Initialize embedding model
            self._embedder = SentenceTransformer(self._embedding_model)

            # Create collections if they don't exist
            self._ensure_collections()

            self._initialized = True
            return True

        except Exception as e:
            print(f"Failed to initialize vector store: {e}")
            return False

    def _ensure_collections(self):
        """Create collections if they don't exist."""
        collections = [INTERACTIONS_COLLECTION, PROBLEMS_COLLECTION]

        existing = {c.name for c in self._client.get_collections().collections}

        for collection in collections:
            if collection not in existing:
                self._client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIM,
                        distance=Distance.COSINE,
                    ),
                )

    def _generate_id(self, content: str, project_id: int, timestamp: str) -> str:
        """Generate a unique ID for a memory entry."""
        data = f"{content}:{project_id}:{timestamp}"
        return hashlib.md5(data.encode()).hexdigest()

    def _embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        if self._embedder is None:
            raise RuntimeError("Embedder not initialized")
        return self._embedder.encode(text).tolist()

    def store_interaction(
        self,
        project_id: int,
        session_id: int,
        content: str,
        sentiment: str,
        reachy_response: str | None = None,
    ) -> str | None:
        """Store an interaction in the vector store.

        Returns the memory ID if successful, None otherwise.
        """
        if not self._ensure_initialized():
            return None

        timestamp = datetime.now().isoformat()
        memory_id = self._generate_id(content, project_id, timestamp)

        # Create embedding from content
        embedding = self._embed(content)

        # Store in Qdrant
        self._client.upsert(
            collection_name=INTERACTIONS_COLLECTION,
            points=[
                PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload={
                        "project_id": project_id,
                        "session_id": session_id,
                        "content": content,
                        "sentiment": sentiment,
                        "reachy_response": reachy_response,
                        "timestamp": timestamp,
                        "memory_type": "interaction",
                    },
                )
            ],
        )

        return memory_id

    def store_problem_solution(
        self,
        project_id: int,
        session_id: int | None,
        problem: str,
        solution: str,
        tags: list[str] | None = None,
    ) -> str | None:
        """Store a problem-solution pair for future reference.

        Returns the memory ID if successful, None otherwise.
        """
        if not self._ensure_initialized():
            return None

        timestamp = datetime.now().isoformat()
        content = f"Problem: {problem}\nSolution: {solution}"
        memory_id = self._generate_id(content, project_id, timestamp)

        # Create embedding from combined problem+solution
        embedding = self._embed(content)

        # Store in Qdrant
        self._client.upsert(
            collection_name=PROBLEMS_COLLECTION,
            points=[
                PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload={
                        "project_id": project_id,
                        "session_id": session_id,
                        "problem": problem,
                        "solution": solution,
                        "tags": tags or [],
                        "timestamp": timestamp,
                        "memory_type": "problem_solution",
                    },
                )
            ],
        )

        return memory_id

    def search_similar_interactions(
        self,
        query: str,
        project_id: int | None = None,
        limit: int = 5,
    ) -> list[SemanticMemory]:
        """Search for similar past interactions.

        Args:
            query: Text to search for
            project_id: Optional project ID to filter by
            limit: Maximum results to return

        Returns:
            List of similar memories with scores
        """
        if not self._ensure_initialized():
            return []

        # Generate query embedding
        query_embedding = self._embed(query)

        # Build filter
        query_filter = None
        if project_id is not None:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="project_id",
                        match=MatchValue(value=project_id),
                    )
                ]
            )

        # Search
        results = self._client.query_points(
            collection_name=INTERACTIONS_COLLECTION,
            query=query_embedding,
            query_filter=query_filter,
            limit=limit,
        )

        # Convert to SemanticMemory objects
        memories = []
        for result in results.points:
            payload = result.payload
            memories.append(
                SemanticMemory(
                    id=result.id,
                    project_id=payload["project_id"],
                    session_id=payload.get("session_id"),
                    content=payload["content"],
                    memory_type=payload["memory_type"],
                    sentiment=payload.get("sentiment"),
                    timestamp=payload["timestamp"],
                    metadata={"reachy_response": payload.get("reachy_response")},
                    score=result.score,
                )
            )

        return memories

    def search_similar_problems(
        self,
        query: str,
        project_id: int | None = None,
        limit: int = 5,
    ) -> list[SemanticMemory]:
        """Search for similar past problems and their solutions.

        Args:
            query: Problem description to search for
            project_id: Optional project ID to filter by (None = search all)
            limit: Maximum results to return

        Returns:
            List of similar problem-solution pairs with scores
        """
        if not self._ensure_initialized():
            return []

        # Generate query embedding
        query_embedding = self._embed(query)

        # Build filter
        query_filter = None
        if project_id is not None:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="project_id",
                        match=MatchValue(value=project_id),
                    )
                ]
            )

        # Search
        results = self._client.query_points(
            collection_name=PROBLEMS_COLLECTION,
            query=query_embedding,
            query_filter=query_filter,
            limit=limit,
        )

        # Convert to SemanticMemory objects
        memories = []
        for result in results.points:
            payload = result.payload
            memories.append(
                SemanticMemory(
                    id=result.id,
                    project_id=payload["project_id"],
                    session_id=payload.get("session_id"),
                    content=f"Problem: {payload['problem']}\nSolution: {payload['solution']}",
                    memory_type="problem_solution",
                    sentiment=None,
                    timestamp=payload["timestamp"],
                    metadata={
                        "problem": payload["problem"],
                        "solution": payload["solution"],
                        "tags": payload.get("tags", []),
                    },
                    score=result.score,
                )
            )

        return memories

    def find_related_across_projects(
        self,
        query: str,
        exclude_project_id: int | None = None,
        limit: int = 5,
    ) -> list[SemanticMemory]:
        """Find related content across ALL projects.

        Useful for "you solved something similar in another project" suggestions.
        """
        if not self._ensure_initialized():
            return []

        # Generate query embedding
        query_embedding = self._embed(query)

        # Build filter to exclude current project
        query_filter = None
        if exclude_project_id is not None:
            query_filter = Filter(
                must_not=[
                    FieldCondition(
                        key="project_id",
                        match=MatchValue(value=exclude_project_id),
                    )
                ]
            )

        # Search both collections
        interaction_results = self._client.query_points(
            collection_name=INTERACTIONS_COLLECTION,
            query=query_embedding,
            query_filter=query_filter,
            limit=limit,
        )

        problem_results = self._client.query_points(
            collection_name=PROBLEMS_COLLECTION,
            query=query_embedding,
            query_filter=query_filter,
            limit=limit,
        )

        # Combine and sort by score
        all_results = []

        for result in interaction_results.points:
            payload = result.payload
            all_results.append(
                SemanticMemory(
                    id=result.id,
                    project_id=payload["project_id"],
                    session_id=payload.get("session_id"),
                    content=payload["content"],
                    memory_type="interaction",
                    sentiment=payload.get("sentiment"),
                    timestamp=payload["timestamp"],
                    score=result.score,
                )
            )

        for result in problem_results.points:
            payload = result.payload
            all_results.append(
                SemanticMemory(
                    id=result.id,
                    project_id=payload["project_id"],
                    session_id=payload.get("session_id"),
                    content=f"Problem: {payload['problem']}\nSolution: {payload['solution']}",
                    memory_type="problem_solution",
                    sentiment=None,
                    timestamp=payload["timestamp"],
                    metadata={
                        "problem": payload["problem"],
                        "solution": payload["solution"],
                    },
                    score=result.score,
                )
            )

        # Sort by score (highest first) and limit
        all_results.sort(key=lambda x: x.score or 0, reverse=True)
        return all_results[:limit]

    def get_project_summary_embedding(self, project_id: int) -> list[float] | None:
        """Get an aggregate embedding representing a project's content.

        Useful for finding similar projects.
        """
        if not self._ensure_initialized():
            return None

        # Get all interactions for the project
        results = self._client.scroll(
            collection_name=INTERACTIONS_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="project_id",
                        match=MatchValue(value=project_id),
                    )
                ]
            ),
            limit=100,
            with_vectors=True,
        )

        if not results[0]:
            return None

        # Average all vectors
        vectors = [point.vector for point in results[0]]
        if not vectors:
            return None

        avg_vector = [
            sum(v[i] for v in vectors) / len(vectors)
            for i in range(len(vectors[0]))
        ]

        return avg_vector


# Singleton instance
_vector_store: VectorStore | None = None


def get_vector_store(host: str | None = None, port: int | None = None) -> VectorStore:
    """Get or create the vector store singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(host=host, port=port)
    return _vector_store
