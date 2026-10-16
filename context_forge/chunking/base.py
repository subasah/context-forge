"""
ContextForge Base Chunker Interface.
"""

from abc import ABC, abstractmethod
from typing import List
from context_forge.core.document import Document, Chunk


class BaseChunker(ABC):
    """Abstract base class for chunking implementations."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Split a document into a sequence of Chunk objects."""
        pass
