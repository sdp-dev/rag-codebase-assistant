"""Code chunking for repository files."""

from pathlib import Path
from typing import Iterator

from .python_parser import PythonParser, CodeChunk


# Directories to skip when walking repository
SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "env",
    ".env",
    "node_modules",
    ".tox",
    ".pytest_cache",
    ".mypy_cache",
    "dist",
    "build",
    "egg-info",
    ".eggs",
}

# File patterns to skip
SKIP_PATTERNS = {
    "test_",
    "_test.py",
    "tests.py",
    "conftest.py",
}


class CodeChunker:
    """Walks repository and extracts code chunks."""

    def __init__(self, include_tests: bool = False):
        """Initialize the chunker.

        Args:
            include_tests: Whether to include test files
        """
        self.parser = PythonParser()
        self.include_tests = include_tests

    def chunk_repository(self, repo_path: Path, repo_name: str) -> Iterator[CodeChunk]:
        """Walk a repository and yield code chunks.

        Args:
            repo_path: Path to the repository root
            repo_name: Name of the repository

        Yields:
            CodeChunk objects for each function, class, and method
        """
        for py_file in self._walk_python_files(repo_path):
            # Get relative path for display
            try:
                rel_path = py_file.relative_to(repo_path)
            except ValueError:
                rel_path = py_file

            try:
                chunks = self.parser.parse_file(py_file, repo_name)
                for chunk in chunks:
                    # Update file path to be relative
                    chunk.file_path = str(rel_path)
                    yield chunk
            except Exception as e:
                # Skip files that can't be parsed
                print(f"Warning: Could not parse {py_file}: {e}")
                continue

    def _walk_python_files(self, root: Path) -> Iterator[Path]:
        """Walk directory tree and yield Python files.

        Args:
            root: Root directory to walk

        Yields:
            Paths to Python files
        """
        for item in root.iterdir():
            if item.is_dir():
                # Skip excluded directories
                if item.name in SKIP_DIRS:
                    continue
                if item.name.endswith(".egg-info"):
                    continue

                # Recurse into subdirectory
                yield from self._walk_python_files(item)

            elif item.is_file() and item.suffix == ".py":
                # Skip test files unless configured to include them
                if not self.include_tests and self._is_test_file(item):
                    continue

                yield item

    def _is_test_file(self, path: Path) -> bool:
        """Check if a file is a test file.

        Args:
            path: Path to check

        Returns:
            True if the file appears to be a test file
        """
        name = path.name

        # Check filename patterns
        for pattern in SKIP_PATTERNS:
            if pattern in name:
                return True

        # Check if in tests directory
        parts = path.parts
        if "tests" in parts or "test" in parts:
            return True

        return False

    def count_python_files(self, repo_path: Path) -> int:
        """Count Python files in a repository.

        Args:
            repo_path: Path to the repository

        Returns:
            Number of Python files
        """
        return sum(1 for _ in self._walk_python_files(repo_path))
