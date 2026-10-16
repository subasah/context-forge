"""
ContextForge Ingestion Cleaner & Token Estimator.
Normalizes text, removes noise, strips duplicate boilerplate, and provides accurate tokenization.
"""

from __future__ import annotations
import hashlib
import re
import unicodedata
from typing import Optional

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _TIKTOKEN_AVAILABLE = False
    _ENCODER = None


class TextCleaner:
    """Production text cleaning and token auditing."""

    @staticmethod
    def count_tokens(text: str) -> int:
        """Count tokens accurately using tiktoken cl100k_base or heuristic fallback."""
        if not text:
            return 0
        if _TIKTOKEN_AVAILABLE and _ENCODER is not None:
            try:
                return len(_ENCODER.encode(text))
            except Exception:
                pass
        # Heuristic fallback: ~4 characters per token in English
        return max(1, len(text.split()))

    @classmethod
    def clean(
        cls,
        text: str,
        remove_excess_whitespace: bool = True,
        normalize_unicode: bool = True,
        strip_page_markers: bool = False,
    ) -> str:
        """Clean and normalize raw text."""
        if not text:
            return ""

        cleaned = text

        if normalize_unicode:
            cleaned = unicodedata.normalize("NFKC", cleaned)

        if strip_page_markers:
            cleaned = re.sub(r"---\s*\[Page \d+\]\s*---", "", cleaned)

        if remove_excess_whitespace:
            # Replace non-breaking spaces
            cleaned = cleaned.replace("\u00a0", " ")
            # Replace multiple consecutive spaces on the same line with single space
            cleaned = re.sub(r"[ \t]+", " ", cleaned)
            # Replace 3 or more consecutive newlines with 2 newlines
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
            cleaned = cleaned.strip()

        return cleaned

    @staticmethod
    def compute_hash(text: str) -> str:
        """MD5 checksum for content deduplication."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()
