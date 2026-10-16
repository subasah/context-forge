"""Tests for Dense, Sparse, Hybrid, and Reranking retrievers."""

import pytest
from context_forge.core.document import Document
from context_forge.chunking.recursive import RecursiveCharacterChunker
from context_forge.embeddings.dense import SemanticLSAEmbeddingProvider
from context_forge.retrieval.dense_retriever import DenseRetriever
from context_forge.retrieval.sparse_retriever import SparseRetriever
from context_forge.retrieval.hybrid_retriever import HybridRetriever
from context_forge.retrieval.reranker import HeuristicCrossAttentionReranker


@pytest.fixture
def indexed_retrievers():
    texts = [
        "Python is a versatile high-level programming language used in machine learning and data science.",
        "PostgreSQL is a powerful open-source object-relational database management system with ACID guarantees.",
        "Reciprocal Rank Fusion RRF is an algorithm for combining multiple search ranking results into a single list.",
        "Model Context Protocol MCP defines standard interfaces for AI agents to retrieve context on demand.",
    ]
    docs = [Document(text=t, metadata={"title": f"Doc {i}"}) for i, t in enumerate(texts)]
    chunker = RecursiveCharacterChunker(chunk_size=300, chunk_overlap=50)
    all_chunks = []
    for d in docs:
        all_chunks.extend(chunker.chunk(d))

    embed_provider = SemanticLSAEmbeddingProvider(dimension=32)
    dense = DenseRetriever(embed_provider)
    sparse = SparseRetriever(k1=1.5, b=0.75)
    hybrid = HybridRetriever(dense, sparse, fusion_mode="rrf", rrf_k=60)

    dense.index(all_chunks)
    sparse.index(all_chunks)
    hybrid.index(all_chunks)

    return dense, sparse, hybrid, all_chunks


def test_dense_retrieval(indexed_retrievers):
    dense, _, _, _ = indexed_retrievers
    results = dense.retrieve("programming language for data science", top_k=2)
    assert len(results) == 2
    assert "Python" in results[0].chunk.text


def test_sparse_bm25_retrieval(indexed_retrievers):
    _, sparse, _, _ = indexed_retrievers
    results = sparse.retrieve("ACID relational database PostgreSQL", top_k=2)
    assert len(results) == 2
    assert "PostgreSQL" in results[0].chunk.text
    assert results[0].sparse_score is not None and results[0].sparse_score > 0.0


def test_hybrid_rrf_retrieval(indexed_retrievers):
    _, _, hybrid, _ = indexed_retrievers
    results = hybrid.retrieve("Reciprocal Rank Fusion algorithm", top_k=3)
    assert len(results) >= 1
    top = results[0]
    assert "Reciprocal Rank Fusion" in top.chunk.text
    assert top.rrf_score is not None
    assert top.retrieval_mode == "hybrid_rrf"


def test_reranker(indexed_retrievers):
    _, _, hybrid, _ = indexed_retrievers
    candidates = hybrid.retrieve("Model Context Protocol for AI agents", top_k=4)
    reranker = HeuristicCrossAttentionReranker()
    reranked = reranker.rerank("Model Context Protocol for AI agents", candidates, top_n=2)
    assert len(reranked) == 2
    assert "Model Context Protocol" in reranked[0].chunk.text
    assert reranked[0].rank == 1
