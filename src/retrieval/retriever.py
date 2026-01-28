"""Semantic code retrieval."""

from pathlib import Path
from typing import Optional

from src.indexing import CodeEmbedder, VectorStore
from src.parsing import CodeChunk


class CodeRetriever:
    """Combines embedding and vector store for code retrieval."""

    def __init__(self, vector_store: Optional[VectorStore] = None):
        """Initialize the retriever.

        Args:
            vector_store: VectorStore instance (creates one if not provided)
        """
        self.vector_store = vector_store or VectorStore()

    def retrieve(
        self,
        query: str,
        k: int = 5,
        repos: Optional[list[str]] = None
    ) -> list[CodeChunk]:
        """Retrieve relevant code chunks for a query.

        Args:
            query: Natural language query
            k: Number of chunks to retrieve
            repos: Optional list of repo names to search within

        Returns:
            List of relevant CodeChunk objects
        """
        results = self.vector_store.search(query, k=k, repo_filter=repos)
        return [chunk for chunk, score in results]

    def retrieve_with_scores(
        self,
        query: str,
        k: int = 5,
        repos: Optional[list[str]] = None
    ) -> list[tuple[CodeChunk, float]]:
        """Retrieve relevant code chunks with similarity scores.

        Args:
            query: Natural language query
            k: Number of chunks to retrieve
            repos: Optional list of repo names to search within

        Returns:
            List of (CodeChunk, score) tuples
        """
        return self.vector_store.search(query, k=k, repo_filter=repos)

    def format_context(self, chunks: list[CodeChunk]) -> str:
        """Format retrieved chunks as context for the LLM.

        Args:
            chunks: List of CodeChunk objects

        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant code found."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            location = f"{chunk.repo_name}/{chunk.file_path}:{chunk.line_start}-{chunk.line_end}"

            context_parts.append(f"### Code Snippet {i}")
            context_parts.append(f"**Location:** `{location}`")
            context_parts.append(f"**Type:** {chunk.type}")
            context_parts.append(f"**Name:** `{chunk.name}`")

            if chunk.docstring:
                context_parts.append(f"**Docstring:** {chunk.docstring}")

            context_parts.append(f"```python\n{chunk.code}\n```")
            context_parts.append("")  # Empty line between chunks

        return "\n".join(context_parts)

    def add_chunks(self, chunks: list[CodeChunk]) -> int:
        """Add chunks to the vector store.

        Args:
            chunks: List of CodeChunk objects

        Returns:
            Number of chunks added
        """
        return self.vector_store.add_chunks(chunks)

    def save(self, path: Optional[Path] = None) -> None:
        """Save the vector store to disk."""
        self.vector_store.save(path)

    def load(self, path: Optional[Path] = None) -> bool:
        """Load the vector store from disk."""
        return self.vector_store.load(path)
