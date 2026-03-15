"""BM25 side-index for Chroma hybrid search.

Operates in two modes based on persist_path:
- persist_path is not None: disk-backed, saves/loads pickle file
- persist_path is None: in-memory only, state lives for process lifetime
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

from src.rag.sparse import BM25SparseEncoder

logger = logging.getLogger(__name__)


class BM25Index:
    """BM25 side-index keyed by Chroma document ID."""

    def __init__(self, persist_path: Optional[str] = None):
        self._persist_path = persist_path
        self._doc_ids: list[str] = []
        self._doc_texts: list[str] = []
        self._encoder = BM25SparseEncoder()
        self._fitted = False

    @property
    def size(self) -> int:
        return len(self._doc_ids)

    def add(self, ids: list[str], texts: list[str]) -> None:
        """Add documents to the index. Rebuilds BM25 model."""
        self._doc_ids.extend(ids)
        self._doc_texts.extend(texts)
        self._rebuild_model()

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return top-k (doc_id, score) pairs for the query."""
        if not self._fitted or not self._doc_ids:
            return []

        scores = self._encoder.get_scores(query)
        scored = list(zip(self._doc_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def delete(self, ids: list[str]) -> None:
        """Remove documents by ID and rebuild."""
        id_set = set(ids)
        pairs = [(did, txt) for did, txt in zip(self._doc_ids, self._doc_texts) if did not in id_set]
        if pairs:
            self._doc_ids, self._doc_texts = list(zip(*pairs))
            self._doc_ids = list(self._doc_ids)
            self._doc_texts = list(self._doc_texts)
        else:
            self._doc_ids = []
            self._doc_texts = []
        self._rebuild_model()

    def save(self) -> None:
        """Persist to disk. No-op when persist_path is None."""
        if self._persist_path is None:
            return
        data = {"doc_ids": self._doc_ids, "doc_texts": self._doc_texts}
        Path(self._persist_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, "wb") as f:
            pickle.dump(data, f)
        logger.debug(f"BM25 index saved ({len(self._doc_ids)} docs) to {self._persist_path}")

    @classmethod
    def load(cls, persist_path: str) -> "BM25Index":
        """Load a disk-backed index from a pickle file."""
        idx = cls(persist_path=persist_path)
        path = Path(persist_path)
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            idx._doc_ids = data.get("doc_ids", [])
            idx._doc_texts = data.get("doc_texts", [])
            if idx._doc_ids:
                idx._rebuild_model()
            logger.debug(f"BM25 index loaded ({len(idx._doc_ids)} docs) from {persist_path}")
        return idx

    def rebuild(self, ids: list[str], texts: list[str]) -> None:
        """Rebuild from external data (e.g., Chroma collection contents)."""
        self._doc_ids = list(ids)
        self._doc_texts = list(texts)
        self._rebuild_model()
        logger.info(f"BM25 index rebuilt with {len(self._doc_ids)} documents")

    def _rebuild_model(self) -> None:
        """Refit the BM25 model on current corpus."""
        if self._doc_texts:
            self._encoder.fit(self._doc_texts)
            self._fitted = True
        else:
            self._encoder = BM25SparseEncoder()
            self._fitted = False
