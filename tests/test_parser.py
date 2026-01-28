"""Tests for the Python parser."""

import pytest
from src.parsing.python_parser import PythonParser, CodeChunk


@pytest.fixture
def parser():
    return PythonParser()


def test_parse_function(parser):
    """Test parsing a simple function."""
    source = '''
def hello(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"
'''
    chunks = parser.parse_source(source, "test.py", "test-repo")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.type == "function"
    assert chunk.name == "hello"
    assert chunk.docstring == "Say hello to someone."
    assert chunk.repo_name == "test-repo"


def test_parse_class(parser):
    """Test parsing a class with methods."""
    source = '''
class Calculator:
    """A simple calculator."""

    def add(self, a, b):
        """Add two numbers."""
        return a + b

    def subtract(self, a, b):
        """Subtract two numbers."""
        return a - b
'''
    chunks = parser.parse_source(source, "calc.py", "test-repo")

    # Should get class + 2 methods
    assert len(chunks) == 3

    class_chunk = next(c for c in chunks if c.type == "class")
    assert class_chunk.name == "Calculator"
    assert class_chunk.docstring == "A simple calculator."

    method_chunks = [c for c in chunks if c.type == "method"]
    assert len(method_chunks) == 2
    assert any(c.name == "Calculator.add" for c in method_chunks)
    assert any(c.name == "Calculator.subtract" for c in method_chunks)


def test_parse_no_docstring(parser):
    """Test parsing code without docstrings."""
    source = '''
def no_doc(x):
    return x * 2
'''
    chunks = parser.parse_source(source, "test.py", "test-repo")

    assert len(chunks) == 1
    assert chunks[0].docstring is None


def test_chunk_to_text():
    """Test CodeChunk text conversion."""
    chunk = CodeChunk(
        type="function",
        name="test_func",
        code="def test_func(): pass",
        docstring="A test function",
        file_path="test.py",
        line_start=1,
        line_end=1,
        repo_name="test-repo"
    )

    text = chunk.to_text()
    assert "function: test_func" in text
    assert "Docstring: A test function" in text
    assert "def test_func(): pass" in text
