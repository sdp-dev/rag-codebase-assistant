"""Tests for the code retriever."""

import pytest
from src.retrieval.retriever import CodeRetriever
from src.parsing.python_parser import CodeChunk


@pytest.fixture
def sample_chunks():
    """Create sample code chunks for testing."""
    return [
        CodeChunk(
            type="function",
            name="fetch_data",
            code="def fetch_data(url): return requests.get(url).json()",
            docstring="Fetch JSON data from a URL",
            file_path="api.py",
            line_start=1,
            line_end=1,
            repo_name="test-repo"
        ),
        CodeChunk(
            type="function",
            name="process_data",
            code="def process_data(data): return [x * 2 for x in data]",
            docstring="Process data by doubling values",
            file_path="utils.py",
            line_start=1,
            line_end=1,
            repo_name="test-repo"
        ),
        CodeChunk(
            type="class",
            name="DataProcessor",
            code="class DataProcessor:\n    def run(self): pass",
            docstring="Main data processing class",
            file_path="processor.py",
            line_start=1,
            line_end=2,
            repo_name="test-repo"
        ),
    ]


@pytest.fixture
def retriever(sample_chunks):
    r = CodeRetriever()
    r.add_chunks(sample_chunks)
    return r


def test_retrieve(retriever):
    """Test basic retrieval."""
    chunks = retriever.retrieve("how to fetch data from API", k=2)

    assert len(chunks) == 2
    # fetch_data should be most relevant
    assert any(c.name == "fetch_data" for c in chunks)


def test_retrieve_with_scores(retriever):
    """Test retrieval with scores."""
    results = retriever.retrieve_with_scores("data processing", k=2)

    assert len(results) == 2
    for chunk, score in results:
        assert isinstance(chunk, CodeChunk)
        assert isinstance(score, float)
        assert score > 0


def test_format_context(retriever):
    """Test context formatting."""
    chunks = retriever.retrieve("fetch data", k=1)
    context = retriever.format_context(chunks)

    assert "### Code Snippet 1" in context
    assert "**Location:**" in context
    assert "```python" in context


def test_format_context_empty(retriever):
    """Test formatting empty results."""
    context = retriever.format_context([])

    assert "No relevant code found" in context
