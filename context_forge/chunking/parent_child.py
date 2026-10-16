"""
ContextForge Parent-Child Hierarchical Chunker.
Separates the retrieval representation (small focused child chunks)
from the generation context (large comprehensive parent chunks).
"""

import uuid
from typing import List, Tuple
from context_forge.chunking.base import BaseChunker
from context_forge.chunking.recursive import RecursiveCharacterChunker
from context_forge.core.document import Document, Chunk
from context_forge.ingestion.cleaner import TextCleaner


class ParentChildChunker(BaseChunker):
    """
    Hierarchical chunker producing linked Parent-Child chunk pairs.
    Child chunks provide sharp vector embeddings for retrieval;
    Parent chunks provide rich contextual windows for LLM generation.
    """

    def __init__(
        self,
        parent_chunk_size: int = 1200,
        parent_overlap: int = 150,
        child_chunk_size: int = 300,
        child_overlap: int = 50,
    ):
        super().__init__(chunk_size=child_chunk_size, chunk_overlap=child_overlap)
        self.parent_chunker = RecursiveCharacterChunker(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap,
        )
        self.child_chunker = RecursiveCharacterChunker(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
        )

    def chunk(self, document: Document) -> List[Chunk]:
        """Returns child chunks, each with parent_id and parent_text populated."""
        parent_chunks = self.parent_chunker.chunk(document)
        all_child_chunks: List[Chunk] = []

        global_child_idx = 0
        for p_idx, p_chunk in enumerate(parent_chunks):
            p_id = f"parent_{document.id}_{p_idx}"
            # Create a pseudo-doc for child chunking
            pseudo_doc = Document(
                id=p_id,
                text=p_chunk.text,
                metadata={
                    **document.metadata,
                    "parent_chunk_index": p_idx,
                },
            )
            children = self.child_chunker.chunk(pseudo_doc)

            for c_idx, c_chunk in enumerate(children):
                c_chunk.id = f"child_{p_id}_{c_idx}"
                c_chunk.parent_id = p_id
                c_chunk.parent_text = p_chunk.text
                c_chunk.chunk_index = global_child_idx
                c_chunk.metadata.update({
                    "strategy": "parent_child",
                    "parent_id": p_id,
                    "parent_index": p_idx,
                    "parent_token_count": p_chunk.token_count,
                    "is_child": True,
                })
                all_child_chunks.append(c_chunk)
                global_child_idx += 1

        return all_child_chunks

    def chunk_with_parents(self, document: Document) -> Tuple[List[Chunk], List[Chunk]]:
        """Returns both parent chunks and child chunks explicitly."""
        parent_chunks = self.parent_chunker.chunk(document)
        for idx, p in enumerate(parent_chunks):
            p.id = f"parent_{document.id}_{idx}"
            p.metadata.update({"strategy": "parent_child", "is_parent": True})

        children = self.chunk(document)
        return parent_chunks, children
