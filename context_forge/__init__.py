"""
ContextForge: Enterprise-Grade RAG, Grounding & Context Engineering Framework.
"""

from context_forge.pipeline import RAGPipeline
from context_forge.core.document import (
    Document,
    Chunk,
    RetrievalResult,
    Citation,
    ClaimVerification,
    GroundingReport,
    AssembledContext,
    TokenAllocation,
)
from context_forge.core.config import PipelineConfig

__version__ = "0.1.0"
__all__ = [
    "RAGPipeline",
    "PipelineConfig",
    "Document",
    "Chunk",
    "RetrievalResult",
    "Citation",
    "ClaimVerification",
    "GroundingReport",
    "AssembledContext",
    "TokenAllocation",
]
