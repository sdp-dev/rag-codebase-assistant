"""Tests for the vector store."""

import tempfile
from pathlib import Path

import pytest
from src.indexing.vector_store import VectorStore
from src.parsing.python_parser import CodeChunk


@pytest.fixture
def sample_chunks():
    """Create sample code chunks for testing."""
    return [
        CodeChunk(
            type="function",
            name="add",
            code="def add(a, b): return a + b",
            docstring="Add two numbers",
            file_path="math.py",
            line_start=1,
            line_end=1,
            repo_name="test-repo"
        ),
        CodeChunk(
            type="function",
            name="subtract",
            code="def subtract(a, b): return a - b",
            docstring="Subtract two numbers",
            file_path="math.py",
            line_start=3,
            line_end=3,
            repo_name="test-repo"
        ),
        CodeChunk(
            type="class",
            name="Calculator",
            code="class Calculator:\n    pass",
            docstring="A calculator class",
            file_path="calc.py",
            line_start=1,
            line_end=2,
            repo_name="other-repo"
        ),
    ]


@pytest.fixture
def vector_store():
    return VectorStore()


def test_add_chunks(vector_store, sample_chunks):
    """Test adding chunks to the store."""
    added = vector_store.add_chunks(sample_chunks)

    assert added == 3
    assert vector_store.total_chunks == 3


def test_search(vector_store, sample_chunks):
    """Test searching for similar chunks."""
    vector_store.add_chunks(sample_chunks)

    results = vector_store.search("add two numbers together", k=2)

    assert len(results) == 2
    # First result should be the add function
    chunk, score = results[0]
    assert chunk.name == "add"
    assert score > 0


def test_search_with_repo_filter(vector_store, sample_chunks):
    """Test filtering search by repository."""
    vector_store.add_chunks(sample_chunks)

    # Search only in test-repo
    results = vector_store.search("calculator", k=3, repo_filter=["test-repo"])

    # Should not include the Calculator class from other-repo
    assert all(chunk.repo_name == "test-repo" for chunk, _ in results)


def test_remove_repo(vector_store, sample_chunks):
    """Test removing a repository from the store."""
    vector_store.add_chunks(sample_chunks)
    assert vector_store.total_chunks == 3

    removed = vector_store.remove_repo("test-repo")

    assert removed == 2
    assert vector_store.total_chunks == 1


def test_save_and_load(vector_store, sample_chunks):
    """Test persistence."""
    vector_store.add_chunks(sample_chunks)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        vector_store.save(path)

        # Create new store and load
        new_store = VectorStore()
        loaded = new_store.load(path)

        assert loaded
        assert new_store.total_chunks == 3


def test_get_repo_stats(vector_store, sample_chunks):
    """Test getting per-repo statistics."""
    vector_store.add_chunks(sample_chunks)

    stats = vector_store.get_repo_stats()

    assert stats["test-repo"] == 2
    assert stats["other-repo"] == 1
