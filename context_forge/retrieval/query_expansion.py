"""
ContextForge Query Expansion & HyDE (Hypothetical Document Embeddings).
Bridges vocabulary gaps and aligns question-passage manifold distributions.
"""

from typing import List, Optional, Callable
import re


class QueryExpander:
    """Expands queries using linguistic variation and multi-query decomposition."""

    def __init__(self, expansion_fn: Optional[Callable[[str], List[str]]] = None):
        self.expansion_fn = expansion_fn

    def expand(self, query: str) -> List[str]:
        """Returns original query plus generated variants."""
        if self.expansion_fn:
            try:
                variants = self.expansion_fn(query)
                return [query] + [v for v in variants if v.strip() and v.strip() != query]
            except Exception:
                pass

        # Deterministic heuristic expansion: strip question words, isolate noun phrases
        cleaned = re.sub(r"^(what|how|why|when|where|who|explain|describe)\s+(is|are|does|do|can)\s+", "", query, flags=re.I)
        variants = [query]
        if cleaned.strip() and cleaned.strip() != query:
            variants.append(cleaned.strip())
        return variants


class HyDEGenerator:
    """
    Hypothetical Document Embeddings (HyDE) Generator.
    Gao et al., 2022: Instead of embedding a question, synthesizes a hypothetical
    passage that answers the question, and embeds the passage.
    """

    def __init__(self, generator_fn: Optional[Callable[[str], str]] = None):
        self.generator_fn = generator_fn

    def generate_hypothetical_document(self, query: str) -> str:
        """Generates a hypothetical document passage matching the query topic."""
        if self.generator_fn:
            try:
                return self.generator_fn(query)
            except Exception:
                pass

        # Zero-shot heuristic template for offline/demo mode
        return (
            f"Regarding {query}: In standard enterprise systems, this is implemented "
            f"through specialized architectural mechanisms that guarantee high performance, "
            f"accuracy, and fault tolerance. The core specification defines key protocols, "
            f"data structures, and algorithmic configurations to address {query} directly."
        )
