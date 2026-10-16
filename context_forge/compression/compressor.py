"""
ContextForge Contextual Compressor.
Prunes irrelevant sentences and applies MMR to eliminate redundant passages.
"""

from typing import List, Optional
import re
import numpy as np

from context_forge.core.document import RetrievalResult
from context_forge.embeddings.base import BaseEmbeddingProvider
from context_forge.compression.mmr import maximal_marginal_relevance, compute_cosine_similarity
from context_forge.ingestion.cleaner import TextCleaner


class ContextualCompressor:
    """
    Contextual sentence compressor and redundancy filter.
    Drastically compresses token footprint by extracting query-relevant sentences.
    """

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        similarity_threshold: float = 0.55,
        min_sentences_per_chunk: int = 1,
    ):
        self.embedding_provider = embedding_provider
        self.similarity_threshold = similarity_threshold
        self.min_sentences_per_chunk = min_sentences_per_chunk

    def compress_chunk_sentences(self, query: str, candidate: RetrievalResult) -> RetrievalResult:
        """Extracts only query-relevant sentences from a chunk."""
        text = candidate.chunk.text
        # Regex split on sentence boundaries
        raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(raw_sentences) <= 1:
            return candidate

        query_vec = self.embedding_provider.embed_query(query)
        sentence_vecs = self.embedding_provider.embed_documents(raw_sentences)

        relevance_scores = [
            compute_cosine_similarity(query_vec, sentence_vecs[i])
            for i in range(len(raw_sentences))
        ]

        # Select sentences exceeding threshold, or top N sentences
        retained_sentences: List[str] = []
        for i, (sent, score) in enumerate(zip(raw_sentences, relevance_scores)):
            if score >= self.similarity_threshold:
                retained_sentences.append(sent)

        # Fallback to top sentence if none crossed threshold
        if len(retained_sentences) < self.min_sentences_per_chunk:
            top_sent_idx = int(np.argmax(relevance_scores))
            retained_sentences = [raw_sentences[top_sent_idx]]

        compressed_text = " ".join(retained_sentences)
        orig_tokens = candidate.chunk.token_count
        new_tokens = TextCleaner.count_tokens(compressed_text)

        new_cand = candidate.model_copy(deep=True)
        new_cand.chunk.text = compressed_text
        new_cand.chunk.token_count = new_tokens
        new_cand.metadata.update({
            "original_tokens": orig_tokens,
            "compressed_tokens": new_tokens,
            "compression_ratio": round(1.0 - (new_tokens / max(orig_tokens, 1)), 2),
            "retained_sentence_count": len(retained_sentences),
            "total_sentence_count": len(raw_sentences),
        })
        new_cand.retrieval_mode = "compressed"
        return new_cand

    def compress_results(
        self,
        query: str,
        candidates: List[RetrievalResult],
        apply_sentence_filtering: bool = True,
        apply_mmr: bool = True,
        top_n: int = 3,
        mmr_lambda: float = 0.7,
    ) -> List[RetrievalResult]:
        """Applies sentence filtering and MMR redundancy reduction."""
        if not candidates:
            return []

        processed = candidates
        if apply_sentence_filtering:
            processed = [self.compress_chunk_sentences(query, c) for c in candidates]

        if apply_mmr and len(processed) > top_n:
            query_vec = self.embedding_provider.embed_query(query)
            cand_vecs = self.embedding_provider.embed_documents([c.chunk.text for c in processed])
            processed = maximal_marginal_relevance(
                query_vector=query_vec,
                candidate_vectors=cand_vecs,
                candidates=processed,
                top_n=top_n,
                lambda_param=mmr_lambda,
            )

        return processed[:top_n]
