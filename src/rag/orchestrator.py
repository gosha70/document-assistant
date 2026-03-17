"""Query orchestrator — sits between chat routes and retriever+generator.

When all query_pipeline features are disabled, delegates directly to the
retriever and generator with no overhead.  Each feature (decomposition,
HyDE, corrective retrieval, verification) is independently enabled via
config and adds its results to the response metadata dict.
"""

import logging
import time
from typing import Any, Callable, Iterator

from langchain_core.documents import Document

from src.config.settings import Settings
from src.rag.embeddings import EmbeddingAdapter
from src.rag.generation import Generator
from src.rag.retrieval import Retriever, deduplicate_docs
from src.utils.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class QueryOrchestrator:
    """Pipeline coordinator for query-time intelligence features."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        llm: Any,
        embedding: EmbeddingAdapter,
        settings: Settings,
    ):
        self._retriever = retriever
        self._generator = generator
        self._llm = llm
        self._embedding = embedding
        self._settings = settings
        self._qp = settings.query_pipeline

    # ------------------------------------------------------------------
    # Internal retrieval helpers
    # ------------------------------------------------------------------

    def _retrieve(self, query: str, metadata: dict) -> list[Document]:
        """Run retrieval with optional decomposition and HyDE."""
        # Step 1: Query decomposition
        if self._qp.decomposition_enabled:
            docs = self._retrieve_with_decomposition(query, metadata)
        elif self._qp.hyde_enabled:
            docs = self._retrieve_with_hyde(query, metadata)
        else:
            docs = self._retriever.retrieve(query)

        # Step 2: Corrective re-retrieval if confidence is low
        if self._qp.corrective_retrieval_enabled and docs:
            docs = self._maybe_corrective(query, docs, metadata)

        return docs

    def _retrieve_with_decomposition(self, query: str, metadata: dict) -> list[Document]:
        """Decompose query into sub-queries, retrieve per sub-query, merge."""
        from src.rag.query_transform import decompose_query, merge_sub_results

        sub_queries = decompose_query(query, self._llm, self._qp.max_sub_queries)
        metadata["decomposed_queries"] = sub_queries

        if len(sub_queries) <= 1:
            if self._qp.hyde_enabled:
                return self._retrieve_with_hyde(query, metadata)
            return self._retriever.retrieve(query)

        sub_results = []
        for sq in sub_queries:
            if self._qp.hyde_enabled:
                docs = self._retrieve_with_hyde(sq, metadata)
            else:
                docs = self._retriever.retrieve(sq)
            sub_results.append(docs)

        merged = merge_sub_results(sub_results)

        # Rerank merged set against original query, or trim to final_k
        if self._retriever._reranker and len(merged) > self._retriever._final_k:
            merged = self._retriever._reranker.rerank(
                query=query,
                documents=merged,
                top_k=self._retriever._final_k,
            )
        else:
            merged = merged[: self._retriever._final_k]

        return merged

    def _retrieve_with_hyde(self, query: str, metadata: dict) -> list[Document]:
        """Generate hypothetical document, embed it, search with that vector."""
        from src.rag.query_transform import generate_hypothetical_document

        hyde_text = generate_hypothetical_document(query, self._llm)
        hyde_embedding = self._embedding.embed_documents([hyde_text])[0]
        metadata["hyde_used"] = True

        return self._retriever.retrieve_with_vector(hyde_embedding, query)

    def _maybe_corrective(
        self,
        query: str,
        docs: list[Document],
        metadata: dict,
    ) -> list[Document]:
        """If retrieval confidence is low, trigger HyDE re-retrieval and merge."""
        from src.rag.query_transform import compute_retrieval_confidence

        confidence = compute_retrieval_confidence(docs)
        metadata["retrieval_confidence"] = confidence

        if confidence >= self._qp.corrective_retrieval_threshold:
            return docs

        logger.info(
            f"Low retrieval confidence ({confidence:.3f} < "
            f"{self._qp.corrective_retrieval_threshold}), triggering corrective re-retrieval"
        )
        metadata["corrective_triggered"] = True

        from src.rag.query_transform import generate_hypothetical_document

        hyde_text = generate_hypothetical_document(query, self._llm)
        hyde_embedding = self._embedding.embed_documents([hyde_text])[0]
        hyde_docs = self._retriever.retrieve_with_vector(hyde_embedding, query)

        # Merge original + HyDE results, dedup, re-rerank or trim
        merged = deduplicate_docs(docs + hyde_docs)
        if self._retriever._reranker and len(merged) > self._retriever._final_k:
            merged = self._retriever._reranker.rerank(
                query=query,
                documents=merged,
                top_k=self._retriever._final_k,
            )
        else:
            merged = merged[: self._retriever._final_k]

        return merged

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _maybe_verify(
        self,
        answer: str,
        documents: list[Document],
        query: str,
        metadata: dict,
    ) -> str:
        """Run verification if enabled. May revise the answer."""
        if not self._qp.verification_enabled:
            return answer

        from src.rag.verification import verify_answer

        result = verify_answer(answer, documents, query, self._llm)
        metadata["verification"] = {
            "verified": result["verified"],
            "unsupported_claims": result["unsupported_claims"],
        }

        if not result["verified"] and result["revised_answer"]:
            return result["revised_answer"]

        return answer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str, history: str = "") -> dict:
        """Execute the full query pipeline (retrieve → generate → verify).

        Returns dict with keys: answer, sources, metadata.
        """
        metadata: dict[str, Any] = {}
        metrics = get_metrics_collector() if self._settings.telemetry.enabled else None

        start = time.monotonic()
        documents = self._retrieve(query, metadata)
        if metrics:
            metrics.record_retrieval(time.monotonic() - start)

        if not documents:
            if metrics:
                metrics.record_retrieval_no_source(
                    self._settings.alerting.window_seconds,
                )
            return {
                "answer": "No relevant documents found for your question.",
                "sources": [],
                "metadata": metadata,
            }

        result = self._generator.generate(
            query=query,
            documents=documents,
            history=history,
        )

        answer = self._maybe_verify(result["answer"], documents, query, metadata)

        return {
            "answer": answer,
            "sources": result["sources"],
            "metadata": metadata,
        }

    def run_stream(
        self, query: str, history: str = ""
    ) -> tuple[list[Document] | None, Iterator[str], Callable[[], list[dict]], dict]:
        """Execute retrieval, then stream generation tokens.

        Returns (documents, token_iterator, get_sources_fn, metadata).
        documents is None when no results found.

        Verification runs post-stream via the metadata dict — the caller
        can check metadata["verification"] after consuming all tokens.
        """
        metadata: dict[str, Any] = {}
        metrics = get_metrics_collector() if self._settings.telemetry.enabled else None

        start = time.monotonic()
        documents = self._retrieve(query, metadata)
        if metrics:
            metrics.record_retrieval(time.monotonic() - start)

        if not documents:
            if metrics:
                metrics.record_retrieval_no_source(
                    self._settings.alerting.window_seconds,
                )
            return None, iter([]), lambda: [], metadata

        sources = Generator.extract_sources(documents)

        accumulated_tokens: list[str] = []

        def tracking_stream():
            for token in self._generator.generate_stream(
                query=query,
                documents=documents,
                history=history,
            ):
                accumulated_tokens.append(token)
                yield token

            # Post-stream verification
            if self._qp.verification_enabled:
                full_answer = "".join(accumulated_tokens)
                from src.rag.verification import verify_answer

                result = verify_answer(full_answer, documents, query, self._llm)
                metadata["verification"] = {
                    "verified": result["verified"],
                    "unsupported_claims": result["unsupported_claims"],
                }

        return documents, tracking_stream(), lambda: sources, metadata
