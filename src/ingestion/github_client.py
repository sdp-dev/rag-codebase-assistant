"""GitHub repository cloning and updating."""

from pathlib import Path
from git import Repo, GitCommandError


def clone_repo(url: str, target_dir: Path) -> Path:
    """Clone a GitHub repository to the target directory.

    Args:
        url: GitHub repository URL (HTTPS format)
        target_dir: Directory to clone into

    Returns:
        Path to the cloned repository

    Raises:
        GitCommandError: If cloning fails
    """
    # Extract repo name from URL
    repo_name = url.rstrip("/").split("/")[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    repo_path = target_dir / repo_name

    if repo_path.exists():
        # Repository already exists, pull instead
        pull_repo(repo_path)
        return repo_path

    # Clone the repository
    Repo.clone_from(url, repo_path)
    return repo_path


def pull_repo(repo_path: Path) -> None:
    """Pull latest changes for an existing repository.

    Args:
        repo_path: Path to the repository

    Raises:
        GitCommandError: If pull fails
    """
    repo = Repo(repo_path)
    origin = repo.remotes.origin
    origin.pull()


def get_repo_name(url: str) -> str:
    """Extract repository name from a GitHub URL.

    Args:
        url: GitHub repository URL

    Returns:
        Repository name
    """
    repo_name = url.rstrip("/").split("/")[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
    return repo_name
