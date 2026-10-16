"""
ContextForge Safety & Out-of-Domain Guardrails.
Prevents ungrounded hallucinations by intercepting low-confidence retrievals
and enforcing domain adherence.
"""

from typing import List, Optional
from context_forge.core.document import RetrievalResult
from context_forge.core.config import GroundingConfig


class RetrievalGuardrail:
    """Out-of-domain and low-confidence interception guardrail."""

    def __init__(self, config: Optional[GroundingConfig] = None):
        self.config = config or GroundingConfig()

    def check_retrieval_sufficiency(
        self,
        query: str,
        results: List[RetrievalResult],
    ) -> tuple[bool, Optional[str]]:
        """
        Evaluates whether retrieved results contain sufficient signal to answer the query.
        Returns: (is_sufficient: bool, refusal_reason: Optional[str])
        """
        if not results:
            return False, "No relevant documents found in knowledge base."

        # Compute max retrieval confidence
        top_score = results[0].score

        # Check against minimum confidence threshold
        if top_score < self.config.min_retrieval_confidence:
            return False, (
                f"Retrieval confidence ({top_score:.3f}) below safety threshold "
                f"({self.config.min_retrieval_confidence:.3f})."
            )

        return True, None

    def get_refusal_response(self, reason: Optional[str] = None) -> str:
        """Standardized defensive refusal string."""
        if reason:
            return f"{self.config.refusal_message}\n(Diagnostic: {reason})"
        return self.config.refusal_message
