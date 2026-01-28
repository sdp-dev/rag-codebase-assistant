"""GitHub ingestion module."""

from .github_client import clone_repo, pull_repo, get_repo_name
from .repo_manager import RepoManager

__all__ = ["clone_repo", "pull_repo", "get_repo_name", "RepoManager"]
