"""
ContextForge Core Data Models.
Defines strongly typed entities for documents, chunks, retrieval results,
grounding audits, and context window budgets.
"""

from __future__ import annotations
import uuid
import time
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Raw ingested document before chunking."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def title(self) -> str:
        return self.metadata.get("title", self.source)


class Chunk(BaseModel):
    """Segment of a document produced by a chunking strategy."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    text: str
    chunk_index: int
    token_count: int = 0
    start_char: int = 0
    end_char: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parent_id: Optional[str] = None
    parent_text: Optional[str] = None
    contextual_header: Optional[str] = None
    dense_vector: Optional[List[float]] = None

    @property
    def enriched_text(self) -> str:
        """Text prepended with contextual header (Anthropic-style) if available."""
        if self.contextual_header:
            return f"[{self.contextual_header}]\n{self.text}"
        return self.text


class RetrievalResult(BaseModel):
    """Result of retrieving and scoring a chunk."""
    chunk: Chunk
    score: float
    retrieval_mode: Literal["dense", "sparse", "hybrid_rrf", "hybrid_alpha", "reranked", "compressed"] = "dense"
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    rank: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Inline source citation mapped to a retrieved chunk."""
    citation_id: int
    chunk_id: str
    document_id: str
    source_title: str
    exact_quote: str
    char_match_ratio: float = 1.0
    is_valid: bool = True


class ClaimVerification(BaseModel):
    """Entailment / grounding verification for a single sentence claim."""
    claim_text: str
    is_supported: bool
    confidence: float
    evidence_chunk_ids: List[str] = Field(default_factory=list)
    evidence_quotes: List[str] = Field(default_factory=list)
    reasoning: str = ""


class GroundingReport(BaseModel):
    """Audit report assessing response groundedness, citations, and hallucination risk."""
    is_fully_grounded: bool
    grounding_score: float  # 0.0 to 1.0 (faithfulness)
    citation_coverage: float  # fraction of claims with verified citations
    claims: List[ClaimVerification] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    hallucination_detected: bool = False
    refusal_recommended: bool = False
    refusal_reason: Optional[str] = None


class TokenAllocation(BaseModel):
    """Breakdown of token budget consumption."""
    system_prompt_tokens: int = 0
    instruction_tokens: int = 0
    memory_history_tokens: int = 0
    retrieved_context_tokens: int = 0
    reserved_generation_tokens: int = 0
    total_tokens: int = 0
    budget_limit: int = 4096


class AssembledContext(BaseModel):
    """The final token-budgeted prompt ready for LLM inference."""
    system_prompt: str
    assembled_prompt: str
    retrieved_chunks: List[RetrievalResult] = Field(default_factory=list)
    token_allocation: TokenAllocation
    lost_in_the_middle_applied: bool = False
    ordering: List[str] = Field(default_factory=list)  # chunk ids in final order
