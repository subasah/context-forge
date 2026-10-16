"""
ContextForge Base Embedding Provider Interface.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseEmbeddingProvider(ABC):
    """Abstract interface for dense embedding generation."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Generates normalized (N, D) embeddings for a list of document chunks."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Generates a normalized (D,) embedding vector for a search query."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        pass
