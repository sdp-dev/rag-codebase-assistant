"""Configuration settings for the RAG codebase assistant."""

from pathlib import Path

# Paths
DATA_DIR = Path("data")
REPOS_DIR = DATA_DIR / "repos"
INDEXES_DIR = DATA_DIR / "indexes"
METADATA_FILE = DATA_DIR / "metadata.json"

# Embedding
EMBEDDING_MODEL = "all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768

# Retrieval
DEFAULT_TOP_K = 5

# LLM - using Ollama (local)
LLM_MODEL = "llama3"  # or "mistral", "codellama", etc.
LLM_PROVIDER = "ollama"  # "ollama" or "openai"
