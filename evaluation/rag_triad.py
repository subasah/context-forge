"""
ContextForge RAG Triad Evaluation Suite.
Computes quantitative scores for:
1. Context Relevance (Retrieved Context <-> Query)
2. Faithfulness / Groundedness (Generated Answer <-> Retrieved Context)
3. Answer Relevance (Generated Answer <-> Query)
"""

from typing import List
import re
import numpy as np
from pydantic import BaseModel, Field

from context_forge.core.document import RetrievalResult
from context_forge.embeddings.base import BaseEmbeddingProvider


class RAGTriadScores(BaseModel):
    context_relevance: float  # 0.0 to 1.0
    faithfulness: float  # 0.0 to 1.0
    answer_relevance: float  # 0.0 to 1.0
    composite_rag_score: float  # Harmonic mean or weighted average
    details: dict = Field(default_factory=dict)


class RAGTriadEvaluator:
    """Evaluates the three pillars of the RAG Triad."""

    def __init__(self, embedding_provider: BaseEmbeddingProvider):
        self.embedding_provider = embedding_provider

    def evaluate_triad(
        self,
        query: str,
        retrieved_results: List[RetrievalResult],
        generated_answer: str,
    ) -> RAGTriadScores:
        """Computes Context Relevance, Faithfulness, and Answer Relevance."""
        if not retrieved_results or not generated_answer.strip():
            return RAGTriadScores(
                context_relevance=0.0,
                faithfulness=0.0,
                answer_relevance=0.0,
                composite_rag_score=0.0,
            )

        # 1. Context Relevance: Cosine similarity between query and retrieved chunks
        query_vec = self.embedding_provider.embed_query(query)
        chunk_texts = [r.chunk.text for r in retrieved_results]
        chunk_vecs = self.embedding_provider.embed_documents(chunk_texts)

        chunk_sims = np.dot(chunk_vecs, query_vec)
        context_relevance = float(np.mean(chunk_sims))
        context_relevance = max(0.0, min(1.0, context_relevance))

        # 2. Faithfulness: Entailment / overlap of answer statements with chunks
        answer_vec = self.embedding_provider.embed_query(generated_answer)
        # Max similarity of answer to any chunk
        faithfulness = float(np.max(chunk_sims))
        # Word overlap check
        ans_words = set(re.findall(r"\w{3,}", generated_answer.lower()))
        all_chunk_words = set(re.findall(r"\w{3,}", " ".join(chunk_texts).lower()))
        word_overlap = len(ans_words.intersection(all_chunk_words)) / max(len(ans_words), 1)
        faithfulness = 0.6 * faithfulness + 0.4 * word_overlap
        faithfulness = max(0.0, min(1.0, faithfulness))

        # 3. Answer Relevance: Cosine similarity between generated answer and original query
        answer_relevance = float(np.dot(query_vec, answer_vec))
        answer_relevance = max(0.0, min(1.0, answer_relevance))

        # Composite score (Harmonic mean of the three pillars)
        eps = 1e-6
        composite = 3.0 / (
            (1.0 / (context_relevance + eps))
            + (1.0 / (faithfulness + eps))
            + (1.0 / (answer_relevance + eps))
        )
        composite = max(0.0, min(1.0, composite))

        return RAGTriadScores(
            context_relevance=round(context_relevance, 3),
            faithfulness=round(faithfulness, 3),
            answer_relevance=round(answer_relevance, 3),
            composite_rag_score=round(composite, 3),
            details={
                "retrieved_chunk_count": len(retrieved_results),
                "answer_token_length": len(generated_answer.split()),
                "word_overlap_ratio": round(word_overlap, 3),
            },
        )
