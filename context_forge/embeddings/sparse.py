"""
ContextForge Sparse BM25 Indexer.
Mathematically rigorous Okapi BM25 implementation exposing IDF, TF, and length penalties.
"""

from __future__ import annotations
import math
import re
from collections import Counter
from typing import Dict, List, Set, Tuple


class BM25Indexer:
    """
    Okapi BM25 sparse lexical indexer.
    Implements standard Lucene/Robertson Okapi BM25:
      IDF(q) = ln((N - n(q) + 0.5) / (n(q) + 0.5) + 1)
      score(D, Q) = sum_q [ IDF(q) * (f(q, D) * (k1 + 1)) / (f(q, D) + k1 * (1 - b + b * (|D| / avgdl))) ]
    """

    STOPWORDS: Set[str] = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
        "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
        "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
        "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
        "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
        "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
        "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more",
        "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
        "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
        "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
        "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
        "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
        "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
        "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
        "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
        "your", "yours", "yourself", "yourselves"
    }

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens: List[int] = []
        self.doc_frequencies: Dict[str, int] = {}
        self.term_frequencies: List[Dict[str, int]] = []
        self.idf_cache: Dict[str, float] = {}

    @classmethod
    def tokenize(cls, text: str, remove_stopwords: bool = True) -> List[str]:
        """Case-folding, alphanumeric tokenization and stopword removal."""
        tokens = re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())
        if remove_stopwords:
            tokens = [t for t in tokens if t not in cls.STOPWORDS and len(t) > 1]
        return tokens

    def fit(self, documents: List[str]) -> None:
        """Indexes corpus texts and precomputes IDFs and length statistics."""
        self.corpus_size = len(documents)
        if self.corpus_size == 0:
            return

        self.doc_lens = []
        self.term_frequencies = []
        self.doc_frequencies = {}

        total_tokens = 0
        for doc in documents:
            tokens = self.tokenize(doc)
            length = len(tokens)
            self.doc_lens.append(length)
            total_tokens += length

            tf = Counter(tokens)
            self.term_frequencies.append(dict(tf))

            for term in tf.keys():
                self.doc_frequencies[term] = self.doc_frequencies.get(term, 0) + 1

        self.avg_doc_len = total_tokens / self.corpus_size if self.corpus_size > 0 else 0.0

        # Precompute Robertson-Spärck Jones IDF
        self.idf_cache = {}
        for term, df in self.doc_frequencies.items():
            # Standard Lucene/BM25 IDF formula
            numerator = self.corpus_size - df + 0.5
            denominator = df + 0.5
            self.idf_cache[term] = math.log((numerator / denominator) + 1.0)

    def score_query(self, query: str) -> List[float]:
        """Returns BM25 scores for all documents in the corpus."""
        if self.corpus_size == 0:
            return []

        query_tokens = self.tokenize(query)
        scores = [0.0] * self.corpus_size

        for term in query_tokens:
            if term not in self.idf_cache:
                continue

            idf = self.idf_cache[term]

            for doc_idx in range(self.corpus_size):
                tf = self.term_frequencies[doc_idx].get(term, 0)
                if tf == 0:
                    continue

                doc_len = self.doc_lens[doc_idx]
                len_norm = 1.0 - self.b + self.b * (doc_len / self.avg_doc_len) if self.avg_doc_len > 0 else 1.0
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * len_norm

                scores[doc_idx] += idf * (numerator / denominator)

        return scores

    def explain_score(self, query: str, doc_idx: int) -> List[Dict[str, float]]:
        """Returns breakdown of IDF and TF contribution per query term for a given document."""
        if doc_idx >= self.corpus_size:
            return []

        query_tokens = self.tokenize(query)
        breakdown = []
        doc_len = self.doc_lens[doc_idx]
        len_norm = 1.0 - self.b + self.b * (doc_len / self.avg_doc_len) if self.avg_doc_len > 0 else 1.0

        for term in query_tokens:
            idf = self.idf_cache.get(term, 0.0)
            tf = self.term_frequencies[doc_idx].get(term, 0)
            if tf > 0:
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * len_norm
                term_score = idf * (numerator / denominator)
            else:
                term_score = 0.0

            breakdown.append({
                "term": term,
                "tf": tf,
                "idf": round(idf, 4),
                "contribution": round(term_score, 4),
            })

        return breakdown
