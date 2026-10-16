"""
ContextForge Reranker Engine.
Provides neural Cross-Encoder reranking and high-throughput heuristic cross-attention.
Reranks candidate pools to push the most contextually relevant chunks to the top.
"""

from __future__ import annotations
import logging
import re
from typing import List, Optional
import numpy as np

from context_forge.core.document import RetrievalResult

logger = logging.getLogger(__name__)


class HeuristicCrossAttentionReranker:
    """
    Lightweight, deterministic cross-attention heuristic reranker.
    Scores query-chunk pairs on exact phrase match, token co-occurrence proximity,
    and title relevance. Zero GPU/neural download needed.
    """

    def rerank(self, query: str, candidates: List[RetrievalResult], top_n: int = 3) -> List[RetrievalResult]:
        if not candidates:
            return []

        query_lower = query.lower()
        query_terms = [t for t in re.findall(r"\w+", query_lower) if len(t) > 2]

        scored_results: List[RetrievalResult] = []
        for cand in candidates:
            text_lower = cand.chunk.enriched_text.lower()

            # Exact query substring match bonus
            exact_match_score = 1.0 if query_lower in text_lower else 0.0

            # Term overlap coverage
            matched_terms = [t for t in query_terms if t in text_lower]
            term_coverage = len(matched_terms) / len(query_terms) if query_terms else 0.0

            # Proximity heuristic: distance between matched terms
            proximity_score = 0.0
            if len(matched_terms) >= 2:
                positions = [text_lower.find(t) for t in matched_terms if text_lower.find(t) != -1]
                if positions:
                    span = max(positions) - min(positions)
                    proximity_score = max(0.0, 1.0 - (span / max(len(text_lower), 1)))

            # Base score contribution
            base_score = cand.score
            # Final composite rerank score
            rerank_score = (
                0.35 * base_score
                + 0.35 * term_coverage
                + 0.20 * exact_match_score
                + 0.10 * proximity_score
            )

            res = cand.model_copy(deep=True)
            res.rerank_score = round(float(rerank_score), 4)
            res.score = res.rerank_score
            res.retrieval_mode = "reranked"
            scored_results.append(res)

        scored_results.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
        top_results = scored_results[:top_n]
        for rank, res in enumerate(top_results, start=1):
            res.rank = rank

        return top_results


class NeuralCrossEncoderReranker:
    """Production Cross-Encoder reranker using HuggingFace / SentenceTransformers."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[RetrievalResult], top_n: int = 3) -> List[RetrievalResult]:
        if not candidates:
            return []

        pairs = [[query, cand.chunk.enriched_text] for cand in candidates]
        scores = self.model.predict(pairs)

        scored_results: List[RetrievalResult] = []
        for cand, score in zip(candidates, scores):
            res = cand.model_copy(deep=True)
            # Sigmoid / normalize score
            norm_score = 1.0 / (1.0 + np.exp(-float(score)))
            res.rerank_score = round(float(norm_score), 4)
            res.score = res.rerank_score
            res.retrieval_mode = "reranked"
            scored_results.append(res)

        scored_results.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
        top_results = scored_results[:top_n]
        for rank, res in enumerate(top_results, start=1):
            res.rank = rank

        return top_results


class UnifiedReranker:
    """Unified reranker that prioritizes Neural CrossEncoder and falls back gracefully."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", prefer_neural: bool = False):
        self.prefer_neural = prefer_neural
        self._reranker = None

        if prefer_neural:
            try:
                self._reranker = NeuralCrossEncoderReranker(model_name=model_name)
                logger.info("Initialized NeuralCrossEncoderReranker (%s)", model_name)
            except Exception as e:
                logger.warning("Neural cross-encoder unavailable (%s). Falling back to heuristic reranker.", e)
                self._reranker = HeuristicCrossAttentionReranker()
        else:
            self._reranker = HeuristicCrossAttentionReranker()

    def rerank(self, query: str, candidates: List[RetrievalResult], top_n: int = 3) -> List[RetrievalResult]:
        return self._reranker.rerank(query, candidates, top_n=top_n)
