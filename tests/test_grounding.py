"""Tests for Grounding, Citation Verification, and Hallucination Mitigation."""

import pytest
from context_forge.core.document import Chunk, RetrievalResult
from context_forge.embeddings.dense import SemanticLSAEmbeddingProvider
from context_forge.grounding.citation_engine import CitationEngine
from context_forge.grounding.verifier import GroundingVerifier
from context_forge.grounding.guardrail import RetrievalGuardrail


@pytest.fixture
def sample_results():
    c1 = Chunk(
        document_id="doc_1",
        text="The GDPR Article 17 right to erasure requires data removal within 30 days.",
        chunk_index=0,
        metadata={"title": "GDPR Compliance"},
    )
    c2 = Chunk(
        document_id="doc_2",
        text="Administrative privileges expire automatically after 4 hours of inactivity.",
        chunk_index=1,
        metadata={"title": "Access Control"},
    )
    return [
        RetrievalResult(chunk=c1, score=0.92, rank=1),
        RetrievalResult(chunk=c2, score=0.85, rank=2),
    ]


def test_citation_extraction_and_validation(sample_results):
    answer = "Data must be removed within 30 days under GDPR [Doc 1]. Privileges expire after 4 hours [Doc 2]."
    citations, coverage = CitationEngine.extract_and_validate_citations(answer, sample_results)

    assert len(citations) == 2
    assert citations[0].is_valid is True
    assert citations[0].citation_id == 1
    assert citations[1].is_valid is True
    assert citations[1].citation_id == 2
    assert coverage == 1.0


def test_faithfulness_and_hallucination_detection(sample_results):
    embed_provider = SemanticLSAEmbeddingProvider(dimension=32)
    verifier = GroundingVerifier(embedding_provider=embed_provider)

    # 1. Grounded answer
    faithful_answer = "Data must be removed within 30 days under GDPR [Doc 1]."
    rep1 = verifier.verify_grounding(faithful_answer, sample_results)
    assert rep1.grounding_score >= 0.70
    assert rep1.hallucination_detected is False

    # 2. Hallucinated answer
    hallucinated_answer = "Employees receive free chocolate every Friday from the cafeteria [Doc 1]."
    rep2 = verifier.verify_grounding(hallucinated_answer, sample_results)
    assert rep2.hallucination_detected is True
    assert rep2.refusal_recommended is True


def test_retrieval_guardrail(sample_results):
    guardrail = RetrievalGuardrail()
    is_suff, reason = guardrail.check_retrieval_sufficiency("data erasure rules", sample_results)
    assert is_suff is True

    # Empty results
    is_suff_empty, reason_empty = guardrail.check_retrieval_sufficiency("data erasure", [])
    assert is_suff_empty is False
    assert "No relevant documents" in reason_empty
