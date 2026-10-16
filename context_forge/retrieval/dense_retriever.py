"""
ContextForge Dense Vector Retriever.
Computes cosine similarity matrix across dense representations.
"""

from typing import List, Optional
import numpy as np

from context_forge.core.document import Chunk, RetrievalResult
from context_forge.embeddings.base import BaseEmbeddingProvider
from context_forge.embeddings.dense import l2_normalize


class DenseRetriever:
    """Vector retriever using exact inner product / cosine similarity."""

    def __init__(self, embedding_provider: BaseEmbeddingProvider):
        self.embedding_provider = embedding_provider
        self.chunks: List[Chunk] = []
        self.vectors: Optional[np.ndarray] = None

    def index(self, chunks: List[Chunk]) -> None:
        """Indexes chunks by computing and caching their dense vectors."""
        self.chunks = chunks
        if not chunks:
            self.vectors = None
            return

        texts = [c.enriched_text for c in chunks]
        self.vectors = self.embedding_provider.embed_documents(texts)
        # Store vector back in chunk model for persistence/inspection
        for idx, c in enumerate(self.chunks):
            c.dense_vector = self.vectors[idx].tolist()

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if not self.chunks or self.vectors is None or len(self.vectors) == 0:
            return []

        query_vector = self.embedding_provider.embed_query(query)
        # Since vectors are L2 normalized, dot product equals cosine similarity
        scores = np.dot(self.vectors, query_vector)

        # Top-k indices
        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[RetrievalResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.chunks[idx]
            score = float(scores[idx])
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    dense_score=score,
                    retrieval_mode="dense",
                    rank=rank,
                )
            )

        return results
