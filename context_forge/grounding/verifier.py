"""
ContextForge Faithfulness & Grounding Verifier.
Decomposes generated output into atomic claims and verifies entailment
against retrieved context to detect hallucinations.
"""

from typing import List, Optional
import re
import numpy as np

from context_forge.core.document import (
    ClaimVerification,
    GroundingReport,
    RetrievalResult,
)
from context_forge.embeddings.base import BaseEmbeddingProvider
from context_forge.grounding.citation_engine import CitationEngine


class GroundingVerifier:
    """
    Evaluates response faithfulness against retrieved source context.
    Detects ungrounded assertions and flags hallucinations.
    """

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        entailment_threshold: float = 0.55,
        min_faithfulness_score: float = 0.70,
    ):
        self.embedding_provider = embedding_provider
        self.entailment_threshold = entailment_threshold
        self.min_faithfulness_score = min_faithfulness_score

    def _extract_atomic_claims(self, text: str) -> List[str]:
        """Splits answer text into verifiable atomic assertions."""
        # Strip conversational pleasantries
        cleaned = re.sub(r"^(sure|here is|based on the documents,|according to the context,)\s*", "", text, flags=re.I)
        # Split on sentence boundaries
        raw_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
        # Strip citation annotations for semantic verification
        return [re.sub(r"\[(?:Doc\s*#?|Source\s*#?)?\d+\]", "", s).strip() for s in raw_sents if len(s.split()) >= 3]

    def verify_grounding(
        self,
        generated_answer: str,
        retrieved_results: List[RetrievalResult],
    ) -> GroundingReport:
        """Audits answer against retrieved evidence."""
        if not retrieved_results:
            return GroundingReport(
                is_fully_grounded=False,
                grounding_score=0.0,
                citation_coverage=0.0,
                hallucination_detected=True,
                refusal_recommended=True,
                refusal_reason="No retrieval context provided for verification.",
            )

        # 1. Validate Citations
        citations, citation_coverage = CitationEngine.extract_and_validate_citations(
            generated_answer, retrieved_results
        )

        # 2. Extract Atomic Claims
        claims = self._extract_atomic_claims(generated_answer)
        if not claims:
            return GroundingReport(
                is_fully_grounded=True,
                grounding_score=1.0,
                citation_coverage=citation_coverage,
                citations=citations,
            )

        # 3. Check Entailment for Each Claim against retrieved chunks
        chunk_texts = [r.chunk.text for r in retrieved_results]
        chunk_vecs = self.embedding_provider.embed_documents(chunk_texts)
        claim_vecs = self.embedding_provider.embed_documents(claims)

        claim_verifications: List[ClaimVerification] = []
        supported_count = 0

        for c_idx, claim in enumerate(claims):
            c_vec = claim_vecs[c_idx]
            # Compute similarities against all retrieved chunks
            sims = np.dot(chunk_vecs, c_vec)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            # Also check lexical word overlap to avoid pure embedding false-positives
            claim_words = set(re.findall(r"\w{3,}", claim.lower()))
            chunk_words = set(re.findall(r"\w{3,}", chunk_texts[best_idx].lower()))
            word_overlap = len(claim_words.intersection(chunk_words)) / max(len(claim_words), 1)

            # Composite entailment metric: require both semantic alignment and lexical grounding
            composite_entailment = (0.65 * best_sim) + (0.35 * word_overlap)
            is_supported = (composite_entailment >= self.entailment_threshold) and (word_overlap >= 0.15)

            if is_supported:
                supported_count += 1
                reasoning = f"Supported by Chunk #{best_idx + 1} with similarity {best_sim:.2f}."
            else:
                reasoning = (
                    f"Ungrounded: Best evidence similarity {best_sim:.2f} "
                    f"below threshold {self.entailment_threshold:.2f}."
                )

            claim_verifications.append(
                ClaimVerification(
                    claim_text=claim,
                    is_supported=is_supported,
                    confidence=round(composite_entailment, 3),
                    evidence_chunk_ids=[retrieved_results[best_idx].chunk.id],
                    evidence_quotes=[chunk_texts[best_idx][:150] + "..."],
                    reasoning=reasoning,
                )
            )

        grounding_score = supported_count / len(claims) if claims else 1.0
        hallucination_detected = grounding_score < self.min_faithfulness_score

        return GroundingReport(
            is_fully_grounded=(grounding_score >= 0.95),
            grounding_score=round(grounding_score, 2),
            citation_coverage=citation_coverage,
            claims=claim_verifications,
            citations=citations,
            hallucination_detected=hallucination_detected,
            refusal_recommended=hallucination_detected,
            refusal_reason="Confidence below safety threshold." if hallucination_detected else None,
        )
