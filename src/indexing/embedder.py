"""Code embedding using sentence-transformers."""

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, EMBEDDING_DIMENSION


class CodeEmbedder:
    """Wraps sentence-transformers for code embedding."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Initialize the embedder.

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = EMBEDDING_DIMENSION

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            Array of embeddings with shape (len(texts), dimension)
        """
        if not texts:
            return np.array([]).reshape(0, self.dimension)

        embeddings = self.model.encode(
            texts,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text.

        Args:
            text: Text string to embed

        Returns:
            Embedding array with shape (dimension,)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)
