"""FAISS vector store for code chunks."""

import json
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from config import EMBEDDING_DIMENSION, INDEXES_DIR
from src.parsing import CodeChunk
from .embedder import CodeEmbedder


class VectorStore:
    """Manages FAISS indexes for code chunks."""

    def __init__(self, embedder: Optional[CodeEmbedder] = None):
        """Initialize the vector store.

        Args:
            embedder: CodeEmbedder instance (creates one if not provided)
        """
        self.embedder = embedder or CodeEmbedder()
        self.dimension = EMBEDDING_DIMENSION

        # FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity

        # Metadata storage (parallel to index)
        self.chunks: list[CodeChunk] = []

    def add_chunks(self, chunks: list[CodeChunk]) -> int:
        """Add code chunks to the index.

        Args:
            chunks: List of CodeChunk objects to index

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        # Convert chunks to text for embedding
        texts = [chunk.to_text() for chunk in chunks]

        # Embed texts
        embeddings = self.embedder.embed(texts)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to index
        self.index.add(embeddings)

        # Store metadata
        self.chunks.extend(chunks)

        return len(chunks)

    def search(
        self,
        query: str,
        k: int = 5,
        repo_filter: Optional[list[str]] = None
    ) -> list[tuple[CodeChunk, float]]:
        """Search for similar code chunks.

        Args:
            query: Search query text
            k: Number of results to return
            repo_filter: Optional list of repo names to filter by

        Returns:
            List of (CodeChunk, score) tuples sorted by relevance
        """
        if self.index.ntotal == 0:
            return []

        # Embed query
        query_embedding = self.embedder.embed_single(query)
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # Search with more results if filtering
        search_k = k * 3 if repo_filter else k
        search_k = min(search_k, self.index.ntotal)

        distances, indices = self.index.search(query_embedding, search_k)

        # Collect results with optional filtering
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0:  # FAISS returns -1 for missing results
                continue

            chunk = self.chunks[idx]

            # Apply repo filter if specified
            if repo_filter and chunk.repo_name not in repo_filter:
                continue

            results.append((chunk, float(score)))

            if len(results) >= k:
                break

        return results

    def remove_repo(self, repo_name: str) -> int:
        """Remove all chunks from a repository.

        Args:
            repo_name: Name of repository to remove

        Returns:
            Number of chunks removed
        """
        # Find chunks to keep
        keep_indices = [
            i for i, chunk in enumerate(self.chunks)
            if chunk.repo_name != repo_name
        ]
        removed_count = len(self.chunks) - len(keep_indices)

        if removed_count == 0:
            return 0

        # Rebuild index with remaining chunks
        remaining_chunks = [self.chunks[i] for i in keep_indices]

        # Reset and re-add
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []

        if remaining_chunks:
            self.add_chunks(remaining_chunks)

        return removed_count

    def save(self, path: Optional[Path] = None) -> None:
        """Save the index and metadata to disk.

        Args:
            path: Directory to save to (defaults to INDEXES_DIR)
        """
        path = path or INDEXES_DIR
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))

        # Save chunk metadata
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, path: Optional[Path] = None) -> bool:
        """Load the index and metadata from disk.

        Args:
            path: Directory to load from (defaults to INDEXES_DIR)

        Returns:
            True if loaded successfully, False if files don't exist
        """
        path = path or INDEXES_DIR

        index_path = path / "index.faiss"
        chunks_path = path / "chunks.pkl"

        if not index_path.exists() or not chunks_path.exists():
            return False

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))

        # Load chunk metadata
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        return True

    @property
    def total_chunks(self) -> int:
        """Get total number of indexed chunks."""
        return len(self.chunks)

    def get_repo_stats(self) -> dict[str, int]:
        """Get chunk counts per repository.

        Returns:
            Dictionary mapping repo names to chunk counts
        """
        stats = {}
        for chunk in self.chunks:
            stats[chunk.repo_name] = stats.get(chunk.repo_name, 0) + 1
        return stats
