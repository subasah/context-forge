"""
ContextForge Recursive Hierarchical Chunker.
Recursively splits text on semantic boundaries: paragraphs -> sentences -> clauses -> words.
"""

import re
from typing import List, Optional
from context_forge.chunking.base import BaseChunker
from context_forge.core.document import Document, Chunk
from context_forge.ingestion.cleaner import TextCleaner


class RecursiveCharacterChunker(BaseChunker):
    """
    Industry-standard recursive chunking.
    Attempts to preserve structural context by prioritizing natural boundary splits.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, document: Document) -> List[Chunk]:
        raw_text = document.text
        if not raw_text.strip():
            return []

        raw_splits = self._split_text(raw_text, self.separators)
        chunks: List[Chunk] = []

        start_char = 0
        for idx, text_segment in enumerate(raw_splits):
            chunk_text = text_segment.strip()
            if not chunk_text:
                continue

            # Approximate char offsets
            found_start = raw_text.find(chunk_text, start_char)
            if found_start != -1:
                start_char = found_start
                end_char = start_char + len(chunk_text)
            else:
                end_char = start_char + len(chunk_text)

            chunks.append(
                Chunk(
                    document_id=document.id,
                    text=chunk_text,
                    chunk_index=idx,
                    token_count=TextCleaner.count_tokens(chunk_text),
                    start_char=start_char,
                    end_char=end_char,
                    metadata={
                        **document.metadata,
                        "strategy": "recursive",
                    },
                )
            )

        return chunks

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    sub_splits = self._split_text(s, new_separators)
                    good_splits.extend(sub_splits)
                else:
                    good_splits.append(s)

        # Merge good_splits respecting chunk_size and chunk_overlap
        current_chunk: List[str] = []
        current_length = 0

        for piece in good_splits:
            piece_len = len(piece) + (len(separator) if current_chunk else 0)
            if current_length + piece_len <= self.chunk_size:
                current_chunk.append(piece)
                current_length += piece_len
            else:
                if current_chunk:
                    merged = separator.join(current_chunk).strip()
                    if merged:
                        final_chunks.append(merged)

                # Reset with overlap
                overlap_buffer: List[str] = []
                overlap_len = 0
                for prev in reversed(current_chunk):
                    if overlap_len + len(prev) <= self.chunk_overlap:
                        overlap_buffer.insert(0, prev)
                        overlap_len += len(prev)
                    else:
                        break

                current_chunk = overlap_buffer + [piece]
                current_length = sum(len(p) for p in current_chunk) + (
                    len(separator) * (len(current_chunk) - 1)
                )

        if current_chunk:
            merged = separator.join(current_chunk).strip()
            if merged:
                final_chunks.append(merged)

        return final_chunks
