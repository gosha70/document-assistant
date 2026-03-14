import logging
from abc import ABC, abstractmethod
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class Reranker(ABC):
    """Interface for reranking retrieved documents."""

    @abstractmethod
    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        """Rerank documents by relevance to query and return top_k."""


class NoOpReranker(Reranker):
    """Pass-through reranker that simply truncates to top_k."""

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        return documents[:top_k]


class CrossEncoderReranker(Reranker):
    """Reranker using a sentence-transformers CrossEncoder model.

    Scores each (query, document) pair and returns the top_k by descending score.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self._model_name = model_name
        self._model = CrossEncoder(model_name)
        logger.info(f"Loaded CrossEncoder reranker: {model_name}")

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs)

        scored_docs = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True,
        )

        reranked = [doc for _, doc in scored_docs[:top_k]]
        logger.debug(
            f"Reranked {len(documents)} docs to {len(reranked)}, "
            f"score range: {scored_docs[0][0]:.3f} - {scored_docs[-1][0]:.3f}"
        )
        return reranked
