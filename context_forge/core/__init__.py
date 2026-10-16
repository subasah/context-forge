"""ContextForge Core Package."""

from context_forge.core.document import (
    Document,
    Chunk,
    RetrievalResult,
    Citation,
    ClaimVerification,
    GroundingReport,
    TokenAllocation,
    AssembledContext,
)
from context_forge.core.config import (
    PipelineConfig,
    ChunkingConfig,
    RetrievalConfig,
    RerankingConfig,
    CompressionConfig,
    GroundingConfig,
    MemoryConfig,
)
from context_forge.core.trace import ExecutionTrace, TraceStep, TraceContext

__all__ = [
    "Document",
    "Chunk",
    "RetrievalResult",
    "Citation",
    "ClaimVerification",
    "GroundingReport",
    "TokenAllocation",
    "AssembledContext",
    "PipelineConfig",
    "ChunkingConfig",
    "RetrievalConfig",
    "RerankingConfig",
    "CompressionConfig",
    "GroundingConfig",
    "MemoryConfig",
    "ExecutionTrace",
    "TraceStep",
    "TraceContext",
]
