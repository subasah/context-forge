"""
ContextForge Dense Embedding Providers.
Includes SentenceTransformers integration with automatic fallback to
LSA (Latent Semantic Analysis) for air-gapped or zero-download environments.
"""

from __future__ import annotations
import logging
from typing import List, Optional
import numpy as np

from context_forge.embeddings.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors so that dot product is equivalent to cosine similarity."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


class SemanticLSAEmbeddingProvider(BaseEmbeddingProvider):
    """
    Lightweight, deterministic Dense Semantic Embedding Provider using
    TF-IDF and Truncated SVD (Latent Semantic Analysis).
    Requires zero network access, runs in milliseconds, and produces genuine dense vectors.
    """

    def __init__(self, dimension: int = 128):
        self._dim = dimension
        self.fitted = False
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=2048,
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(n_components=self._dim, random_state=42)

    @property
    def dimension(self) -> int:
        return self._dim

    def fit(self, texts: List[str]) -> None:
        if not texts:
            return
        tfidf = self.vectorizer.fit_transform(texts)
        actual_components = min(self._dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        if actual_components < 2:
            self.fitted = False
            return
        self.svd.n_components = actual_components
        self.svd.fit(tfidf)
        self._dim = actual_components
        self.fitted = True

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim))

        if not self.fitted:
            self.fit(texts)

        if not self.fitted:
            # Fallback deterministic bag-of-characters
            dense = np.zeros((len(texts), self._dim))
            for i, t in enumerate(texts):
                for ch in t:
                    dense[i, ord(ch) % self._dim] += 1.0
            return l2_normalize(dense)

        try:
            tfidf = self.vectorizer.transform(texts)
            reduced = self.svd.transform(tfidf)
            return l2_normalize(reduced)
        except Exception:
            return np.ones((len(texts), self._dim)) / np.sqrt(self._dim)

    def embed_query(self, query: str) -> np.ndarray:
        emb = self.embed_documents([query])
        return emb[0]


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """Production Dense Embedding Provider using SentenceTransformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        if hasattr(self.model, "get_embedding_dimension"):
            self._dim = self.model.get_embedding_dimension()
        else:
            self._dim = self.model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim))
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        embeddings = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings[0]


class UnifiedDenseEmbeddingProvider(BaseEmbeddingProvider):
    """
    Unified dense embedding provider that automatically selects SentenceTransformers
    when available, gracefully falling back to Semantic LSA for air-gapped environments.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", fallback_to_lsa: bool = True):
        self.model_name = model_name
        self._provider: BaseEmbeddingProvider

        try:
            self._provider = SentenceTransformerEmbeddingProvider(model_name=model_name)
            logger.info("Initialized SentenceTransformer (%s)", model_name)
        except Exception as e:
            if fallback_to_lsa:
                logger.warning("Could not load %s (%s). Falling back to Semantic LSA.", model_name, e)
                self._provider = SemanticLSAEmbeddingProvider(dimension=128)
            else:
                raise e

    @property
    def dimension(self) -> int:
        return self._provider.dimension

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self._provider.embed_documents(texts)

    def embed_query(self, query: str) -> np.ndarray:
        return self._provider.embed_query(query)
