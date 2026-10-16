"""
ContextForge Citation Engine.
Extracts citations from generated responses, validates verbatim quotes,
and maps claims back to source document chunks.
"""

from __future__ import annotations
import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple
from context_forge.core.document import Citation, RetrievalResult


class CitationEngine:
    """Production source attribution and citation validator."""

    CITATION_PATTERN = re.compile(r"\[(?:Doc\s*#?|Source\s*#?)?(\d+)\]")

    @classmethod
    def extract_and_validate_citations(
        cls,
        generated_text: str,
        retrieved_chunks: List[RetrievalResult],
        min_match_ratio: float = 0.50,
    ) -> Tuple[List[Citation], float]:
        """
        Parses inline citations (e.g. [1], [Doc 2]), extracts the preceding sentence,
        and verifies if the evidence exists in the cited chunk.
        Returns: (List of Citation objects, citation_coverage_ratio)
        """
        if not retrieved_chunks:
            return [], 0.0

        # Create 1-based lookup map for retrieved chunks
        chunk_map: Dict[int, RetrievalResult] = {
            rank: r for rank, r in enumerate(retrieved_chunks, start=1)
        }

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", generated_text) if s.strip()]
        citations: List[Citation] = []
        cited_sentence_count = 0

        for sent_idx, sentence in enumerate(sentences):
            matches = list(cls.CITATION_PATTERN.finditer(sentence))
            if not matches:
                continue

            cited_sentence_count += 1
            # Strip citation tags from sentence to isolate the clean assertion
            clean_assertion = cls.CITATION_PATTERN.sub("", sentence).strip()

            for match in matches:
                ref_id = int(match.group(1))
                if ref_id in chunk_map:
                    res = chunk_map[ref_id]
                    source_text = res.chunk.text.lower()
                    assertion_lower = clean_assertion.lower()

                    # Check verbatim or fuzzy sequence match
                    best_ratio = cls._fuzzy_find_match(assertion_lower, source_text)

                    citations.append(
                        Citation(
                            citation_id=ref_id,
                            chunk_id=res.chunk.id,
                            document_id=res.chunk.document_id,
                            source_title=res.chunk.metadata.get("title", f"Doc {ref_id}"),
                            exact_quote=clean_assertion,
                            char_match_ratio=round(best_ratio, 3),
                            is_valid=(best_ratio >= min_match_ratio),
                        )
                    )

        coverage = (
            cited_sentence_count / len(sentences) if len(sentences) > 0 else 0.0
        )
        return citations, round(coverage, 2)

    @staticmethod
    def _fuzzy_find_match(needle: str, haystack: str) -> float:
        """Finds the maximum sequence matcher ratio for needle within sliding windows of haystack."""
        if not needle or not haystack:
            return 0.0

        if needle in haystack:
            return 1.0

        # Sub-token overlap ratio
        needle_words = set(re.findall(r"\w+", needle))
        haystack_words = set(re.findall(r"\w+", haystack))
        if not needle_words:
            return 0.0

        overlap = len(needle_words.intersection(haystack_words)) / len(needle_words)
        return float(overlap)
