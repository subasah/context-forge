"""
ContextForge Short-Term Buffer Memory.
Manages multi-turn conversational dialogue state within a sliding window.
"""

from typing import Any, Dict, List, Optional
import time
from pydantic import BaseModel, Field
from context_forge.ingestion.cleaner import TextCleaner


class DialogueTurn(BaseModel):
    role: str  # 'user' | 'assistant' | 'system'
    content: str
    timestamp: float = Field(default_factory=time.time)
    token_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BufferMemory:
    """Sliding window short-term conversational buffer."""

    def __init__(self, window_size: int = 6):
        self.window_size = window_size
        self.history: List[DialogueTurn] = []

    def add_turn(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        tokens = TextCleaner.count_tokens(content)
        turn = DialogueTurn(
            role=role,
            content=content,
            token_count=tokens,
            metadata=metadata or {},
        )
        self.history.append(turn)

    def get_recent_turns(self, n: Optional[int] = None) -> List[DialogueTurn]:
        k = n if n is not None else self.window_size
        return self.history[-k:]

    def get_formatted_history(self, n: Optional[int] = None) -> str:
        turns = self.get_recent_turns(n)
        if not turns:
            return ""
        return "\n".join(f"{turn.role.upper()}: {turn.content}" for turn in turns)

    def total_tokens(self, n: Optional[int] = None) -> int:
        turns = self.get_recent_turns(n)
        return sum(t.token_count for t in turns)

    def clear(self) -> None:
        self.history = []
