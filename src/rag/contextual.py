"""Contextual chunk augmentation — enriches chunks with LLM-generated context at index time."""

import logging
import time

from langchain_core.documents import Document

from src.config.settings import get_settings
from src.rag.generation import load_prompt
from src.utils.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class ChunkContextAugmenter:
    """Generates contextual descriptions for chunks using an LLM.

    Two-pass pipeline:
    1. Extract a document-level summary (first N tokens of the full text)
    2. For each chunk, call the LLM to generate a brief context description
    3. Truncate context to max_context_tokens
    4. Store the context in metadata and produce an augmented embedding text

    The augmented text (context + original) is used for embedding and sparse
    indexing. The original page_content is preserved for citations/display.
    """

    def __init__(self, llm, settings=None):
        self._llm = llm
        self._settings = settings or get_settings()
        self._max_context_tokens = self._settings.chunking.contextual.max_context_tokens
        self._summary_tokens = self._settings.chunking.contextual.document_summary_tokens
        self._prompt = load_prompt("chunk_context")

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to max_tokens using the configured tokenizer."""
        from src.rag.chunking import get_tokenizer

        tokenizer = get_tokenizer(self._settings.chunking.tokenizer)
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return tokenizer.decode(tokens[:max_tokens])

    def _get_document_summary(self, full_text: str) -> str:
        """Extract the first N tokens of the document as a summary."""
        return self._truncate_to_tokens(full_text, self._summary_tokens)

    def augment(self, full_text: str, chunks: list[Document]) -> list[Document]:
        """Augment chunks with contextual descriptions.

        For each chunk:
        - Generates chunk_context via LLM (traced with latency + token metrics)
        - Truncates context to max_context_tokens
        - Stores in metadata["chunk_context"]
        - Stores augmented embedding text in metadata["embedding_text"]
          (context + "\\n\\n" + original page_content)
        - Preserves original page_content unchanged (for citations)

        Returns the same list with metadata enriched.
        """
        document_summary = self._get_document_summary(full_text)
        settings = self._settings

        for i, chunk in enumerate(chunks):
            try:
                formatted = self._prompt.format(
                    document_summary=document_summary,
                    chunk_text=chunk.page_content,
                )

                start = time.monotonic()
                raw = self._llm.invoke(formatted)
                latency = time.monotonic() - start

                context = raw.content if hasattr(raw, "content") else str(raw)
                context = self._truncate_to_tokens(context, self._max_context_tokens)

                if settings.telemetry.enabled and settings.telemetry.log_llm_calls:
                    get_metrics_collector().record_llm_call(latency, len(formatted), len(context))
                    logger.debug(f"Chunk {i}: context generated in {latency:.2f}s ({len(context)} chars)")

                chunk.metadata["chunk_context"] = context
                chunk.metadata["embedding_text"] = context + "\n\n" + chunk.page_content

            except Exception as e:
                logger.warning(f"Contextual augmentation failed for chunk {i}: {e}")

        return chunks
