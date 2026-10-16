"""
ContextForge Semantic Chunker.
Splits text dynamically based on embedding distance transitions between sentences.
Avoids mid-concept cuts by detecting thematic boundaries.
"""

import re
import numpy as np
from typing import List, Optional, Callable
from context_forge.chunking.base import BaseChunker
from context_forge.core.document import Document, Chunk
from context_forge.ingestion.cleaner import TextCleaner


class SemanticChunker(BaseChunker):
    """
    Semantic chunker based on embedding distance gradients.
    Groups adjacent sentences until the cosine distance between them exceeds
    a percentile threshold of semantic divergence.
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
        breakpoint_percentile_threshold: float = 80.0,
        max_chunk_size: int = 800,
        min_chunk_size: int = 150,
    ):
        super().__init__(chunk_size=max_chunk_size, chunk_overlap=0)
        self.embed_fn = embed_fn
        self.breakpoint_percentile = breakpoint_percentile_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def _split_into_sentences(self, text: str) -> List[str]:
        # Regex split preserving sentence enders
        raw_sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences

    def _default_embed(self, texts: List[str]) -> np.ndarray:
        """Lightweight bag-of-words / character ngram TF-IDF embedding fallback."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=256, stop_words="english")
        try:
            tfidf = vectorizer.fit_transform(texts).toarray()
            # Normalize
            norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return tfidf / norms
        except Exception:
            # Fallback random deterministic unit vectors if texts are too short
            return np.ones((len(texts), 16)) / np.sqrt(16)

    def chunk(self, document: Document) -> List[Chunk]:
        text = document.text
        if not text.strip():
            return []

        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return [
                Chunk(
                    document_id=document.id,
                    text=text.strip(),
                    chunk_index=0,
                    token_count=TextCleaner.count_tokens(text),
                    metadata={**document.metadata, "strategy": "semantic"},
                )
            ]

        # Generate embeddings for sentences
        if self.embed_fn is not None:
            embeddings = self.embed_fn(sentences)
        else:
            embeddings = self._default_embed(sentences)

        # Compute cosine distances between adjacent sentences: 1 - cos_sim
        distances: List[float] = []
        for i in range(len(embeddings) - 1):
            v1 = embeddings[i]
            v2 = embeddings[i + 1]
            denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_sim = np.dot(v1, v2) / denom if denom > 0 else 1.0
            distance = 1.0 - float(cos_sim)
            distances.append(distance)

        # Determine threshold
        if distances:
            threshold = float(np.percentile(distances, self.breakpoint_percentile))
        else:
            threshold = 0.5

        # Group sentences into chunks based on threshold and size constraints
        chunks: List[Chunk] = []
        current_sentences: List[str] = [sentences[0]]
        current_length = len(sentences[0])
        chunk_idx = 0

        for i, dist in enumerate(distances):
            next_sentence = sentences[i + 1]
            next_len = len(next_sentence)

            # Split condition: distance > threshold AND current size is at least min_size,
            # OR adding next sentence exceeds max_chunk_size
            should_split = (
                (dist >= threshold and current_length >= self.min_chunk_size)
                or (current_length + next_len > self.max_chunk_size)
            )

            if should_split:
                chunk_str = " ".join(current_sentences).strip()
                if chunk_str:
                    chunks.append(
                        Chunk(
                            document_id=document.id,
                            text=chunk_str,
                            chunk_index=chunk_idx,
                            token_count=TextCleaner.count_tokens(chunk_str),
                            metadata={
                                **document.metadata,
                                "strategy": "semantic",
                                "split_distance": round(dist, 4),
                                "breakpoint_threshold": round(threshold, 4),
                            },
                        )
                    )
                    chunk_idx += 1
                current_sentences = [next_sentence]
                current_length = next_len
            else:
                current_sentences.append(next_sentence)
                current_length += next_len + 1

        if current_sentences:
            chunk_str = " ".join(current_sentences).strip()
            if chunk_str:
                chunks.append(
                    Chunk(
                        document_id=document.id,
                        text=chunk_str,
                        chunk_index=chunk_idx,
                        token_count=TextCleaner.count_tokens(chunk_str),
                        metadata={
                            **document.metadata,
                            "strategy": "semantic",
                            "breakpoint_threshold": round(threshold, 4),
                        },
                    )
                )

        return chunks
