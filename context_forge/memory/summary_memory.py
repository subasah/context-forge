"""
ContextForge Hierarchical Summary Memory.
Compresses older conversational turns into persistent key-fact summaries.
"""

from typing import Callable, List, Optional
from context_forge.memory.buffer_memory import DialogueTurn
from context_forge.ingestion.cleaner import TextCleaner


class SummaryMemory:
    """Hierarchical summary memory to maintain bounded token consumption."""

    def __init__(self, summarizer_fn: Optional[Callable[[str, str], str]] = None):
        self.running_summary: str = ""
        self.summarizer_fn = summarizer_fn

    def update_summary(self, expired_turns: List[DialogueTurn]) -> str:
        """Incorporates newly expired turns into the running summary."""
        if not expired_turns:
            return self.running_summary

        turns_text = "\n".join(f"{t.role}: {t.content}" for t in expired_turns)

        if self.summarizer_fn:
            try:
                self.running_summary = self.summarizer_fn(self.running_summary, turns_text)
                return self.running_summary
            except Exception:
                pass

        # Deterministic extraction of user goals and questions
        extracted_facts: List[str] = []
        for turn in expired_turns:
            if turn.role.lower() == "user":
                extracted_facts.append(f"User inquired: '{turn.content[:80]}...'")
            elif turn.role.lower() == "assistant":
                extracted_facts.append(f"Assistant provided guidance on topic.")

        new_segment = "; ".join(extracted_facts)
        if self.running_summary:
            self.running_summary = f"{self.running_summary}\n- Later: {new_segment}"
        else:
            self.running_summary = f"- Earlier context: {new_segment}"

        return self.running_summary

    @property
    def token_count(self) -> int:
        return TextCleaner.count_tokens(self.running_summary)
