"""Tests for the code embedder."""

import numpy as np
import pytest
from src.indexing.embedder import CodeEmbedder
from config import EMBEDDING_DIMENSION


@pytest.fixture
def embedder():
    return CodeEmbedder()


def test_embed_single(embedder):
    """Test embedding a single text."""
    embedding = embedder.embed_single("def hello(): pass")

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (EMBEDDING_DIMENSION,)
    assert embedding.dtype == np.float32


def test_embed_batch(embedder):
    """Test embedding multiple texts."""
    texts = [
        "def hello(): pass",
        "class Calculator: pass",
        "import os"
    ]
    embeddings = embedder.embed(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, EMBEDDING_DIMENSION)
    assert embeddings.dtype == np.float32


def test_embed_empty(embedder):
    """Test embedding empty list."""
    embeddings = embedder.embed([])

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (0, EMBEDDING_DIMENSION)


def test_similar_texts_closer(embedder):
    """Test that similar texts have closer embeddings."""
    func1 = "def add(a, b): return a + b"
    func2 = "def sum(x, y): return x + y"
    unrelated = "class DatabaseConnection: pass"

    emb1 = embedder.embed_single(func1)
    emb2 = embedder.embed_single(func2)
    emb3 = embedder.embed_single(unrelated)

    # Cosine similarity (embeddings already normalized-ish)
    sim_12 = np.dot(emb1, emb2)
    sim_13 = np.dot(emb1, emb3)

    # Similar functions should be more similar to each other
    assert sim_12 > sim_13
