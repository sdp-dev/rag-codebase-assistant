"""Integration tests for the RAG pipeline."""

import tempfile
from pathlib import Path

import pytest

from src.parsing import CodeChunker, PythonParser
from src.indexing import VectorStore
from src.retrieval import CodeRetriever


@pytest.fixture
def temp_repo():
    """Create a temporary repository with Python files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)

        # Create a simple Python module
        (repo_path / "calculator.py").write_text('''
"""Calculator module."""

def add(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Product of a and b
    """
    return a * b


class Calculator:
    """A calculator that maintains state."""

    def __init__(self):
        """Initialize the calculator with zero."""
        self.result = 0

    def add(self, value: int) -> int:
        """Add to the current result."""
        self.result += value
        return self.result

    def reset(self):
        """Reset the result to zero."""
        self.result = 0
''')

        (repo_path / "utils.py").write_text('''
"""Utility functions."""

def validate_number(x):
    """Check if x is a valid number."""
    return isinstance(x, (int, float))


def format_result(value: float) -> str:
    """Format a number for display."""
    return f"Result: {value:.2f}"
''')

        yield repo_path


def test_full_pipeline(temp_repo):
    """Test the complete ingestion -> indexing -> retrieval pipeline."""
    # Parse repository
    chunker = CodeChunker()
    chunks = list(chunker.chunk_repository(temp_repo, "test-calc"))

    # Should find functions, class, and methods
    assert len(chunks) > 0
    chunk_types = {c.type for c in chunks}
    assert "function" in chunk_types
    assert "class" in chunk_types
    assert "method" in chunk_types

    # Index chunks
    retriever = CodeRetriever()
    retriever.add_chunks(chunks)

    # Test retrieval
    results = retriever.retrieve("how to add numbers", k=3)
    assert len(results) > 0

    # The add functions should be in results
    names = [r.name for r in results]
    assert any("add" in name for name in names)


def test_retrieval_relevance(temp_repo):
    """Test that retrieval returns relevant results."""
    chunker = CodeChunker()
    chunks = list(chunker.chunk_repository(temp_repo, "test-calc"))

    retriever = CodeRetriever()
    retriever.add_chunks(chunks)

    # Query about multiplication
    results = retriever.retrieve("multiply two numbers", k=2)
    top_result = results[0]
    assert "multiply" in top_result.name.lower()

    # Query about formatting
    results = retriever.retrieve("format display output", k=2)
    top_result = results[0]
    assert "format" in top_result.name.lower()


def test_context_formatting(temp_repo):
    """Test that context is properly formatted for LLM."""
    chunker = CodeChunker()
    chunks = list(chunker.chunk_repository(temp_repo, "test-calc"))

    retriever = CodeRetriever()
    retriever.add_chunks(chunks)

    results = retriever.retrieve("calculator", k=2)
    context = retriever.format_context(results)

    # Context should contain markdown formatting
    assert "###" in context
    assert "```python" in context
    assert "test-calc" in context  # repo name


def test_persistence(temp_repo):
    """Test that indexes can be saved and loaded."""
    with tempfile.TemporaryDirectory() as index_dir:
        index_path = Path(index_dir)

        # Create and save index
        chunker = CodeChunker()
        chunks = list(chunker.chunk_repository(temp_repo, "test-calc"))

        retriever = CodeRetriever()
        retriever.add_chunks(chunks)
        retriever.save(index_path)

        # Load in new retriever
        new_retriever = CodeRetriever()
        assert new_retriever.load(index_path)

        # Should be able to search
        results = new_retriever.retrieve("add numbers", k=1)
        assert len(results) == 1
