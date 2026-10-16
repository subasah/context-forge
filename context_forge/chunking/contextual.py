"""
ContextForge Contextual Chunker (Anthropic-style Contextual Retrieval).
Enriches chunks with document-level and section-level situational headers
to eliminate orphan context loss during dense and sparse retrieval.
"""

import re
from typing import List, Optional, Callable
from context_forge.chunking.base import BaseChunker
from context_forge.chunking.recursive import RecursiveCharacterChunker
from context_forge.core.document import Document, Chunk
from context_forge.ingestion.cleaner import TextCleaner


class ContextualChunker(BaseChunker):
    """
    Anthropic Contextual Retrieval pattern:
    Prepends document title and hierarchical section context to each chunk.
    Enhances both Dense vector semantic affinity and BM25 lexical keyword matching.
    """

    def __init__(
        self,
        base_chunker: Optional[BaseChunker] = None,
        context_generator: Optional[Callable[[str, str], str]] = None,
    ):
        self.base_chunker = base_chunker or RecursiveCharacterChunker(chunk_size=400, chunk_overlap=60)
        self.context_generator = context_generator
        super().__init__(chunk_size=self.base_chunker.chunk_size, chunk_overlap=self.base_chunker.chunk_overlap)

    def _extract_markdown_headers(self, text: str) -> List[tuple[int, str, int]]:
        """Finds markdown headers with (level, title, char_offset)."""
        headers = []
        for match in re.finditer(r"^(#{1,6})\s+(.+)$", text, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            headers.append((level, title, match.start()))
        return headers

    def _get_active_header_context(self, headers: List[tuple[int, str, int]], char_pos: int) -> str:
        """Determines the active hierarchy of headers at a given character position."""
        if not headers:
            return ""

        active_stack: List[tuple[int, str]] = []
        for level, title, offset in headers:
            if offset <= char_pos:
                while active_stack and active_stack[-1][0] >= level:
                    active_stack.pop()
                active_stack.append((level, title))
            else:
                break

        if not active_stack:
            return ""
        return " > ".join(t for _, t in active_stack)

    def chunk(self, document: Document) -> List[Chunk]:
        raw_chunks = self.base_chunker.chunk(document)
        doc_title = document.metadata.get("title", "Document")
        headers = self._extract_markdown_headers(document.text)

        enriched_chunks: List[Chunk] = []
        for chunk in raw_chunks:
            section_path = self._get_active_header_context(headers, chunk.start_char)

            if self.context_generator:
                try:
                    situational_context = self.context_generator(document.text, chunk.text)
                except Exception:
                    situational_context = f"Document: {doc_title}"
            else:
                if section_path:
                    situational_context = f"Doc: {doc_title} | Section: {section_path}"
                else:
                    situational_context = f"Doc: {doc_title}"

            chunk.contextual_header = situational_context
            chunk.metadata.update({
                "strategy": "contextual_retrieval",
                "contextual_header": situational_context,
                "section_path": section_path,
            })
            # Recompute token count to include enriched header
            chunk.token_count = TextCleaner.count_tokens(chunk.enriched_text)
            enriched_chunks.append(chunk)

        return enriched_chunks
