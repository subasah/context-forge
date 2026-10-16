"""
ContextForge Hybrid Retriever.
Combines Dense Semantic Vectors and Sparse BM25 Lexical Scores
using Reciprocal Rank Fusion (RRF) and Min-Max Convex Combination.
"""

from typing import Dict, List, Literal, Optional
from context_forge.core.document import Chunk, RetrievalResult
from context_forge.retrieval.dense_retriever import DenseRetriever
from context_forge.retrieval.sparse_retriever import SparseRetriever


class HybridRetriever:
    """
    Hybrid retriever implementing Reciprocal Rank Fusion (RRF)
    and normalized score convex combination.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        fusion_mode: Literal["rrf", "convex"] = "rrf",
        rrf_k: int = 60,
        alpha: float = 0.5,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion_mode = fusion_mode
        self.rrf_k = rrf_k
        self.alpha = alpha

    def index(self, chunks: List[Chunk]) -> None:
        self.dense_retriever.index(chunks)
        self.sparse_retriever.index(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        pool_factor: int = 2,
    ) -> List[RetrievalResult]:
        """
        Executes hybrid retrieval:
        1. Retrieves top candidate pool from dense (pool_k = top_k * pool_factor)
        2. Retrieves top candidate pool from sparse (pool_k = top_k * pool_factor)
        3. Fuses rankings using RRF or convex combination
        4. Returns top_k fused results
        """
        pool_k = max(top_k * pool_factor, 10)
        dense_results = self.dense_retriever.retrieve(query, top_k=pool_k)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=pool_k)

        if self.fusion_mode == "rrf":
            return self._fuse_rrf(dense_results, sparse_results, top_k)
        else:
            return self._fuse_convex(dense_results, sparse_results, top_k)

    def _fuse_rrf(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion:
        Score(d) = sum_{system} 1 / (k + rank)
        """
        chunk_map: Dict[str, Chunk] = {}
        dense_ranks: Dict[str, int] = {}
        sparse_ranks: Dict[str, int] = {}
        dense_scores: Dict[str, float] = {}
        sparse_scores: Dict[str, float] = {}

        for rank, res in enumerate(dense_results, start=1):
            cid = res.chunk.id
            chunk_map[cid] = res.chunk
            dense_ranks[cid] = rank
            dense_scores[cid] = res.score

        for rank, res in enumerate(sparse_results, start=1):
            cid = res.chunk.id
            chunk_map[cid] = res.chunk
            sparse_ranks[cid] = rank
            sparse_scores[cid] = res.score

        all_cids = set(dense_ranks.keys()).union(sparse_ranks.keys())
        fused_scores: Dict[str, float] = {}

        for cid in all_cids:
            score = 0.0
            if cid in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[cid])
            if cid in sparse_ranks:
                score += 1.0 / (self.rrf_k + sparse_ranks[cid])
            fused_scores[cid] = score

        # Sort descending by RRF score
        sorted_cids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)[:top_k]

        final_results: List[RetrievalResult] = []
        for rank, cid in enumerate(sorted_cids, start=1):
            final_results.append(
                RetrievalResult(
                    chunk=chunk_map[cid],
                    score=fused_scores[cid],
                    retrieval_mode="hybrid_rrf",
                    dense_score=dense_scores.get(cid),
                    sparse_score=sparse_scores.get(cid),
                    rrf_score=fused_scores[cid],
                    rank=rank,
                    metadata={
                        "dense_rank": dense_ranks.get(cid),
                        "sparse_rank": sparse_ranks.get(cid),
                    },
                )
            )

        return final_results

    def _fuse_convex(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        Min-Max Normalized Convex Combination:
        Score(d) = alpha * Norm(Dense) + (1 - alpha) * Norm(Sparse)
        """
        def min_max_normalize(results: List[RetrievalResult]) -> Dict[str, float]:
            if not results:
                return {}
            scores = [r.score for r in results]
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return {r.chunk.id: 1.0 for r in results}
            return {r.chunk.id: (r.score - min_s) / (max_s - min_s) for r in results}

        norm_dense = min_max_normalize(dense_results)
        norm_sparse = min_max_normalize(sparse_results)

        chunk_map: Dict[str, Chunk] = {r.chunk.id: r.chunk for r in dense_results + sparse_results}
        raw_dense_map = {r.chunk.id: r.score for r in dense_results}
        raw_sparse_map = {r.chunk.id: r.score for r in sparse_results}

        all_cids = set(norm_dense.keys()).union(norm_sparse.keys())
        combined_scores: Dict[str, float] = {}

        for cid in all_cids:
            d_val = norm_dense.get(cid, 0.0)
            s_val = norm_sparse.get(cid, 0.0)
            combined_scores[cid] = self.alpha * d_val + (1.0 - self.alpha) * s_val

        sorted_cids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:top_k]

        final_results: List[RetrievalResult] = []
        for rank, cid in enumerate(sorted_cids, start=1):
            final_results.append(
                RetrievalResult(
                    chunk=chunk_map[cid],
                    score=combined_scores[cid],
                    retrieval_mode="hybrid_alpha",
                    dense_score=raw_dense_map.get(cid),
                    sparse_score=raw_sparse_map.get(cid),
                    rank=rank,
                )
            )

        return final_results
