"""Tests for Unified RAGPipeline orchestration."""

import pytest
from context_forge import RAGPipeline, PipelineConfig


def test_pipeline_end_to_end():
    config = PipelineConfig()
    config.chunking.chunk_size = 120
    config.chunking.chunk_overlap = 20
    pipeline = RAGPipeline(config=config)

    # Ingest sample text
    text = (
        "# Security Key Specification\n\n"
        "Production environments mandate FIDO2 hardware authentication keys. "
        "Session tokens expire after exactly 60 minutes.\n\n"
        "# Incident Escalation\n\n"
        "Severity 1 security events must be escalated to the Incident Response Director within 15 minutes."
    )
    chunks = pipeline.ingest_document(text, title="Security Specs", chunk_strategy="recursive")
    assert len(chunks) >= 2

    # Run query
    query = "What hardware keys are mandated for production?"
    answer, grounding_report, assembled_ctx, trace = pipeline.query(query=query)

    assert len(assembled_ctx.retrieved_chunks) >= 1
    assert "fido2" in answer.lower()
    assert "[Doc 1]" in answer
    assert grounding_report.is_fully_grounded is True
    assert grounding_report.grounding_score >= 0.7
    assert trace.total_duration_ms > 0
    assert len(trace.steps) >= 4
