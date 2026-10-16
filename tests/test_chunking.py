"""Tests for ContextForge chunking strategies."""

import pytest
from context_forge.core.document import Document
from context_forge.chunking.fixed_window import FixedWindowChunker
from context_forge.chunking.recursive import RecursiveCharacterChunker
from context_forge.chunking.semantic import SemanticChunker
from context_forge.chunking.parent_child import ParentChildChunker
from context_forge.chunking.contextual import ContextualChunker


@pytest.fixture
def sample_doc():
    text = (
        "# Architecture Overview\n\n"
        "Retrieval-Augmented Generation (RAG) grounds language models in external knowledge.\n\n"
        "## Chunking Mechanics\n\n"
        "Recursive chunking splits text along hierarchical boundaries such as paragraphs and sentences. "
        "It prevents fragments from cutting through individual words or mid-sentence concepts.\n\n"
        "## Vector Embeddings\n\n"
        "Dense vectors map semantic relationships in high-dimensional continuous space."
    )
    return Document(text=text, metadata={"title": "RAG Architecture", "source": "test_doc.md"})


def test_fixed_window_chunker(sample_doc):
    chunker = FixedWindowChunker(chunk_size=120, chunk_overlap=30)
    chunks = chunker.chunk(sample_doc)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 120
        assert c.document_id == sample_doc.id


def test_recursive_character_chunker(sample_doc):
    chunker = RecursiveCharacterChunker(chunk_size=150, chunk_overlap=30)
    chunks = chunker.chunk(sample_doc)
    assert len(chunks) >= 2
    # Check that chunks preserve semantic boundaries
    combined = " ".join(c.text for c in chunks)
    assert "Retrieval-Augmented Generation" in combined
    assert "Vector Embeddings" in combined


def test_semantic_chunker(sample_doc):
    chunker = SemanticChunker(max_chunk_size=300, min_chunk_size=50)
    chunks = chunker.chunk(sample_doc)
    assert len(chunks) >= 1
    assert all(c.token_count > 0 for c in chunks)


def test_parent_child_chunker(sample_doc):
    chunker = ParentChildChunker(parent_chunk_size=250, child_chunk_size=100)
    children = chunker.chunk(sample_doc)
    assert len(children) >= 1
    for child in children:
        assert child.parent_id is not None
        assert child.parent_text is not None
        assert len(child.parent_text) >= len(child.text)


def test_contextual_chunker(sample_doc):
    chunker = ContextualChunker()
    chunks = chunker.chunk(sample_doc)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.contextual_header is not None
        assert "Doc: RAG Architecture" in c.contextual_header
        assert c.enriched_text.startswith(f"[{c.contextual_header}]")
