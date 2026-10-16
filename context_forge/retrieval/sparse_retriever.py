"""
ContextForge Sparse Retriever.
Leverages BM25 Okapi to score and rank lexical matches.
"""

from typing import List
import numpy as np

from context_forge.core.document import Chunk, RetrievalResult
from context_forge.embeddings.sparse import BM25Indexer


class SparseRetriever:
    """Sparse lexical retriever powered by BM25 Okapi."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.indexer = BM25Indexer(k1=k1, b=b)
        self.chunks: List[Chunk] = []

    def index(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        if not chunks:
            return
        texts = [c.enriched_text for c in chunks]
        self.indexer.fit(texts)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if not self.chunks or self.indexer.corpus_size == 0:
            return []

        raw_scores = self.indexer.score_query(query)
        scores_arr = np.array(raw_scores)

        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores_arr)[::-1][:top_k]

        results: List[RetrievalResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores_arr[idx])
            results.append(
                RetrievalResult(
                    chunk=self.chunks[idx],
                    score=score,
                    sparse_score=score,
                    retrieval_mode="sparse",
                    rank=rank,
                )
            )

        return results
