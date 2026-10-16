"""
ContextForge Fixed Window Sliding Chunker.
Splits text strictly based on token or character windows with configurable overlap.
"""

from typing import List
from context_forge.chunking.base import BaseChunker
from context_forge.core.document import Document, Chunk
from context_forge.ingestion.cleaner import TextCleaner


class FixedWindowChunker(BaseChunker):
    """
    Fixed-window chunking with sliding overlap.
    Guarantees strict upper bound on chunk size, ideal for small embedding context windows.
    """

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80, mode: str = "character"):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.mode = mode
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

    def chunk(self, document: Document) -> List[Chunk]:
        text = document.text
        if not text.strip():
            return []

        chunks: List[Chunk] = []
        step = self.chunk_size - self.chunk_overlap

        if self.mode == "character":
            start = 0
            chunk_idx = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end]
                chunks.append(
                    Chunk(
                        document_id=document.id,
                        text=chunk_text,
                        chunk_index=chunk_idx,
                        token_count=TextCleaner.count_tokens(chunk_text),
                        start_char=start,
                        end_char=end,
                        metadata={
                            **document.metadata,
                            "strategy": "fixed_window",
                            "mode": "character",
                        },
                    )
                )
                chunk_idx += 1
                if end == len(text):
                    break
                start += step

        else:
            # Word/token window
            words = text.split()
            start = 0
            chunk_idx = 0
            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                chunk_words = words[start:end]
                chunk_text = " ".join(chunk_words)
                chunks.append(
                    Chunk(
                        document_id=document.id,
                        text=chunk_text,
                        chunk_index=chunk_idx,
                        token_count=TextCleaner.count_tokens(chunk_text),
                        metadata={
                            **document.metadata,
                            "strategy": "fixed_window",
                            "mode": "word",
                        },
                    )
                )
                chunk_idx += 1
                if end == len(words):
                    break
                start += step

        return chunks
