import logging
from langchain_core.documents import Document

from src.rag.vectorstore import VectorStoreBackend
from src.rag.reranking import Reranker

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves documents from a vector store, optionally reranking results."""

    def __init__(
        self,
        backend: VectorStoreBackend,
        collection_name: str,
        reranker: Reranker | None = None,
        initial_k: int = 20,
        final_k: int = 5,
        use_hybrid: bool = True,
    ):
        self._backend = backend
        self._collection_name = collection_name
        self._reranker = reranker
        self._initial_k = initial_k
        self._final_k = final_k
        self._use_hybrid = use_hybrid

    def retrieve(self, query: str) -> list[Document]:
        if self._use_hybrid:
            docs = self._backend.hybrid_search(
                query=query,
                collection_name=self._collection_name,
                k=self._initial_k,
            )
        else:
            docs = self._backend.search(
                query=query,
                collection_name=self._collection_name,
                k=self._initial_k,
            )

        logger.info(f"Retrieved {len(docs)} candidates for query")

        if self._reranker and len(docs) > self._final_k:
            docs = self._reranker.rerank(query=query, documents=docs, top_k=self._final_k)
            logger.info(f"Reranked to {len(docs)} documents")
        else:
            docs = docs[: self._final_k]

        return docs
