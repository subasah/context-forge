"""ContextForge Embeddings Package."""

from context_forge.embeddings.base import BaseEmbeddingProvider
from context_forge.embeddings.dense import (
    UnifiedDenseEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    SemanticLSAEmbeddingProvider,
    l2_normalize,
)
from context_forge.embeddings.sparse import BM25Indexer

__all__ = [
    "BaseEmbeddingProvider",
    "UnifiedDenseEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "SemanticLSAEmbeddingProvider",
    "BM25Indexer",
    "l2_normalize",
]
