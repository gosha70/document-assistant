"""Sparse encoder interface and BM25 implementation for hybrid search."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SparseVector:
    """Sparse vector representation: parallel arrays of indices and values."""

    indices: list[int]
    values: list[float]


class SparseEncoder(ABC):
    """Abstract interface for sparse vector encoders."""

    @abstractmethod
    def encode(self, texts: list[str]) -> list[SparseVector]:
        """Encode texts into sparse vectors (token_id -> weight mappings)."""

    @abstractmethod
    def encode_query(self, text: str) -> SparseVector:
        """Encode a single query into a sparse vector."""


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Whitespace + lowercasing + punctuation stripping tokenizer."""
    return _TOKEN_RE.findall(text.lower())


class TokenEncoder:
    """Lightweight tokenizer -> token-index vocabulary for producing sparse TF vectors.

    Used by QdrantBackend for server-side IDF sparse indexing.
    The vocabulary is append-only: new tokens get new indices, existing
    mappings never change. This makes concurrent appends safe.
    """

    def __init__(self, vocab: dict[str, int] | None = None):
        self._vocab: dict[str, int] = dict(vocab) if vocab else {}
        self._next_id: int = max(self._vocab.values(), default=-1) + 1

    @property
    def vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode_tf(self, text: str) -> SparseVector:
        """Encode text into a term-frequency sparse vector.

        Unknown tokens are added to the vocabulary (append-only).
        """
        tokens = tokenize(text)
        tf: dict[int, float] = {}
        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = self._next_id
                self._next_id += 1
            idx = self._vocab[token]
            tf[idx] = tf.get(idx, 0) + 1.0

        indices = sorted(tf.keys())
        values = [tf[i] for i in indices]
        return SparseVector(indices=indices, values=values)

    def encode_query_tf(self, text: str) -> SparseVector:
        """Encode query into TF vector using only known vocabulary.

        Unknown tokens are silently ignored (they can't match any stored vectors).
        """
        tokens = tokenize(text)
        tf: dict[int, float] = {}
        for token in tokens:
            idx = self._vocab.get(token)
            if idx is not None:
                tf[idx] = tf.get(idx, 0) + 1.0

        indices = sorted(tf.keys())
        values = [tf[i] for i in indices]
        return SparseVector(indices=indices, values=values)


class BM25SparseEncoder(SparseEncoder):
    """Client-side BM25 sparse vector encoder.

    Uses rank-bm25 for scoring. Builds a token vocabulary from the corpus.
    Used only for Chroma's BM25 side-index (Increment 4), where we control
    the full scoring pipeline client-side.
    """

    def __init__(self):
        self._bm25 = None
        self._corpus_tokens: list[list[str]] = []

    def fit(self, texts: list[str]) -> None:
        """Fit BM25 model on corpus texts."""
        from rank_bm25 import BM25Okapi

        self._corpus_tokens = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def encode(self, texts: list[str]) -> list[SparseVector]:
        """Not used for BM25 — scoring is done via get_scores."""
        raise NotImplementedError("BM25SparseEncoder uses get_scores() for ranking, not encode()")

    def encode_query(self, text: str) -> SparseVector:
        """Not used for BM25 — scoring is done via get_scores."""
        raise NotImplementedError("BM25SparseEncoder uses get_scores() for ranking, not encode_query()")

    def get_scores(self, query: str) -> list[float]:
        """Return BM25 scores for all corpus documents given a query."""
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        return self._bm25.get_scores(tokens).tolist()
