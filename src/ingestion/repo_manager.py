"""Repository metadata management."""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from config import METADATA_FILE, REPOS_DIR


@dataclass
class RepoMetadata:
    """Metadata for an indexed repository."""
    url: str
    name: str
    local_path: str
    last_indexed: str
    file_count: int
    chunk_count: int


class RepoManager:
    """Manages indexed repository metadata."""

    def __init__(self, metadata_file: Path = METADATA_FILE):
        self.metadata_file = metadata_file
        self._repos: dict[str, RepoMetadata] = {}
        self._load()

    def _load(self) -> None:
        """Load metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                data = json.load(f)
                self._repos = {
                    name: RepoMetadata(**meta)
                    for name, meta in data.items()
                }

    def _save(self) -> None:
        """Save metadata to disk."""
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_file, "w") as f:
            json.dump(
                {name: asdict(meta) for name, meta in self._repos.items()},
                f,
                indent=2
            )

    def add_repo(
        self,
        url: str,
        name: str,
        local_path: Path,
        file_count: int = 0,
        chunk_count: int = 0
    ) -> RepoMetadata:
        """Add or update a repository in the metadata.

        Args:
            url: GitHub repository URL
            name: Repository name
            local_path: Path to local clone
            file_count: Number of Python files indexed
            chunk_count: Number of code chunks indexed

        Returns:
            The repository metadata
        """
        metadata = RepoMetadata(
            url=url,
            name=name,
            local_path=str(local_path),
            last_indexed=datetime.now().isoformat(),
            file_count=file_count,
            chunk_count=chunk_count
        )
        self._repos[name] = metadata
        self._save()
        return metadata

    def get_repo(self, name: str) -> Optional[RepoMetadata]:
        """Get metadata for a repository by name."""
        return self._repos.get(name)

    def list_repos(self) -> list[RepoMetadata]:
        """List all indexed repositories."""
        return list(self._repos.values())

    def remove_repo(self, name: str) -> bool:
        """Remove a repository from tracking.

        Args:
            name: Repository name

        Returns:
            True if removed, False if not found
        """
        if name in self._repos:
            del self._repos[name]
            self._save()
            return True
        return False

    def update_repo(
        self,
        name: str,
        file_count: Optional[int] = None,
        chunk_count: Optional[int] = None
    ) -> Optional[RepoMetadata]:
        """Update repository metadata.

        Args:
            name: Repository name
            file_count: New file count (optional)
            chunk_count: New chunk count (optional)

        Returns:
            Updated metadata or None if not found
        """
        if name not in self._repos:
            return None

        meta = self._repos[name]
        if file_count is not None:
            meta.file_count = file_count
        if chunk_count is not None:
            meta.chunk_count = chunk_count
        meta.last_indexed = datetime.now().isoformat()

        self._save()
        return meta
