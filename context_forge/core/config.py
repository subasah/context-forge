"""
ContextForge Configuration.
Centralized configuration with sensible defaults and environment variable overrides.
"""

from typing import Literal
from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    strategy: Literal["fixed", "recursive", "semantic", "parent_child", "contextual"] = "recursive"
    chunk_size: int = 400  # in characters or tokens
    chunk_overlap: int = 80
    parent_chunk_size: int = 1200
    child_chunk_size: int = 300
    semantic_similarity_threshold: float = 0.75
    contextual_summary_max_tokens: int = 50


class RetrievalConfig(BaseModel):
    retrieval_mode: Literal["dense", "sparse", "hybrid_rrf", "hybrid_alpha"] = "hybrid_rrf"
    top_k: int = 5
    rrf_k: int = 60  # Standard Reciprocal Rank Fusion constant
    alpha: float = 0.5  # Convex combination weight: alpha*Dense + (1-alpha)*Sparse
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    enable_query_expansion: bool = False
    enable_hyde: bool = False


class RerankingConfig(BaseModel):
    enabled: bool = True
    top_n: int = 3  # How many chunks to retain after reranking
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_neural_reranker: bool = True  # Can fall back to cross-attention token overlap reranker if offline


class CompressionConfig(BaseModel):
    enabled: bool = False
    method: Literal["mmr", "sentence_filter"] = "mmr"
    mmr_lambda: float = 0.7  # Relevance vs Diversity trade-off
    similarity_threshold: float = 0.65


class GroundingConfig(BaseModel):
    enabled: bool = True
    min_faithfulness_score: float = 0.70
    min_retrieval_confidence: float = 0.35
    strict_citation_validation: bool = True
    refusal_message: str = (
        "I cannot answer this question with high confidence based strictly on the provided documents. "
        "The retrieved evidence does not sufficiently ground the answer."
    )


class MemoryConfig(BaseModel):
    max_context_budget: int = 4096
    reserved_generation_budget: int = 1024
    system_prompt_budget: int = 512
    short_term_window_turns: int = 4
    enable_summary_memory: bool = True
    enable_persistent_memory: bool = True
    mitigate_lost_in_the_middle: bool = True


class PipelineConfig(BaseModel):
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    grounding: GroundingConfig = Field(default_factory=GroundingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    embedding_model: str = "all-MiniLM-L6-v2"
    fallback_to_tfidf: bool = True  # Ensures 100% reliability offline
