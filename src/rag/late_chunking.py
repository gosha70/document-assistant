"""Late chunking — embed full document context, then pool per-chunk token embeddings.

Standard chunking embeds each chunk independently, losing surrounding context.
Late chunking embeds the full document (or page) through a long-context model first,
then pools token-level embeddings for each chunk's token range. This gives each
chunk vector awareness of the broader document context.

Reference: Günther et al., "Late Chunking: Contextual Chunk Embeddings Using
Long-Context Embedding Models" (2024).
"""

import logging
from typing import Optional

import numpy as np
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class LateChunkingEmbedder:
    """Embeds documents using late chunking: full-context encoding then per-chunk pooling."""

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v2-base-en",
        max_context_tokens: int = 8192,
        pooling_strategy: str = "mean",
        device: str = "cpu",
    ):
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._max_context_tokens = max_context_tokens
        self._pooling_strategy = pooling_strategy
        self._model = SentenceTransformer(model_name, device=device)
        self._tokenizer = self._model.tokenizer

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def _tokenize_full_text(self, text: str) -> dict:
        """Tokenize full text, truncating to max_context_tokens."""
        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_context_tokens,
            return_offsets_mapping=True,
        )
        return encoded

    def _get_token_embeddings(self, encoded: dict) -> np.ndarray:
        """Forward pass through the model to get per-token embeddings.

        Returns array of shape (seq_len, hidden_dim).
        """
        import torch

        with torch.no_grad():
            output = self._model[0].auto_model(
                **{k: v for k, v in encoded.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
            )
            # last_hidden_state: (1, seq_len, hidden_dim)
            token_embeddings = output.last_hidden_state[0].cpu().numpy()

        return token_embeddings

    def compute_chunk_token_ranges(
        self,
        full_text: str,
        chunks: list[Document],
        offset_mapping: list[tuple[int, int]],
    ) -> list[Optional[tuple[int, int]]]:
        """Map each chunk's text to token index ranges in the full tokenized text.

        Uses character offset mapping from the tokenizer to find which tokens
        correspond to each chunk's character span in the full text.

        Returns a list of (start_token_idx, end_token_idx) tuples, one per chunk.
        Returns None for chunks whose text cannot be located.
        """
        ranges: list[Optional[tuple[int, int]]] = []
        search_start = 0

        for chunk in chunks:
            chunk_text = chunk.page_content
            # Find the chunk's character position in the full text
            char_start = full_text.find(chunk_text, search_start)
            if char_start == -1:
                # Fallback: search from the beginning
                char_start = full_text.find(chunk_text)

            if char_start == -1:
                logger.warning(
                    f"Could not locate chunk in full text (first 50 chars: "
                    f"'{chunk_text[:50]}...'). Skipping late chunking for this chunk."
                )
                ranges.append(None)
                continue

            char_end = char_start + len(chunk_text)
            # Advance search_start for overlapping chunks
            search_start = char_start + 1

            # Map character range to token range using offset_mapping
            token_start = None
            token_end = None
            for tok_idx, (off_start, off_end) in enumerate(offset_mapping):
                if off_start == 0 and off_end == 0:
                    # Special token (CLS, SEP, PAD) — skip
                    continue
                if token_start is None and off_end > char_start:
                    token_start = tok_idx
                if off_start < char_end:
                    token_end = tok_idx + 1

            if token_start is not None and token_end is not None:
                ranges.append((token_start, token_end))
            else:
                ranges.append(None)

        return ranges

    def _pool_tokens(self, token_embeddings: np.ndarray, start: int, end: int) -> np.ndarray:
        """Pool token embeddings for a chunk range."""
        chunk_tokens = token_embeddings[start:end]
        if len(chunk_tokens) == 0:
            return np.zeros(token_embeddings.shape[1])

        if self._pooling_strategy == "mean":
            return chunk_tokens.mean(axis=0)
        elif self._pooling_strategy == "cls":
            return chunk_tokens[0]
        else:
            return chunk_tokens.mean(axis=0)

    def embed_chunks(
        self,
        full_text: str,
        chunks: list[Document],
    ) -> list[Optional[list[float]]]:
        """Embed chunks using late chunking.

        1. Tokenize full_text through the model's tokenizer
        2. Forward pass to get per-token embeddings
        3. For each chunk, find its token range and mean-pool

        Returns a list of embedding vectors (or None for chunks that couldn't be mapped).
        """
        if not chunks:
            return []

        encoded = self._tokenize_full_text(full_text)
        offset_mapping = encoded["offset_mapping"][0].tolist()
        token_embeddings = self._get_token_embeddings(encoded)

        ranges = self.compute_chunk_token_ranges(full_text, chunks, offset_mapping)

        vectors: list[Optional[list[float]]] = []
        for i, token_range in enumerate(ranges):
            if token_range is None:
                vectors.append(None)
                continue

            start, end = token_range
            pooled = self._pool_tokens(token_embeddings, start, end)

            # Normalize to unit length
            norm = np.linalg.norm(pooled)
            if norm > 0:
                pooled = pooled / norm

            vectors.append(pooled.tolist())

        return vectors

    def embed_document_chunks(
        self,
        full_text: str,
        chunks: list[Document],
        batch_by: str = "page",
    ) -> list[Document]:
        """High-level: embed chunks via late chunking and attach vectors to metadata.

        For documents exceeding max_context_tokens, splits into segments based
        on batch_by strategy and processes each segment independently.

        Sets metadata["_precomputed_vector"] on each chunk (removed by backends
        before persisting). Sets metadata["embedding_strategy"] = "late_chunking".
        """
        from src.rag.chunking import count_tokens

        tokenizer_spec = f"huggingface:{self._model_name}"

        # Check if full text fits in context window
        try:
            full_token_count = count_tokens(full_text, tokenizer_spec)
        except Exception:
            # Fallback: approximate by character count
            full_token_count = len(full_text) // 4

        if full_token_count <= self._max_context_tokens:
            # Single-pass: embed all chunks against full text
            vectors: list[Optional[list[float]]] = self.embed_chunks(full_text, chunks)
            for chunk, vec in zip(chunks, vectors):
                chunk.metadata["embedding_strategy"] = "late_chunking"
                chunk.metadata["late_chunking_model"] = self._model_name
                if vec is not None:
                    chunk.metadata["_precomputed_vector"] = vec
            return chunks

        # Multi-pass: split full text into segments and embed each batch
        segments = self._split_into_segments(full_text, batch_by)
        logger.info(
            f"Full text exceeds context window ({full_token_count} tokens > "
            f"{self._max_context_tokens}). Split into {len(segments)} segments."
        )

        for chunk in chunks:
            chunk.metadata["embedding_strategy"] = "late_chunking"
            chunk.metadata["late_chunking_model"] = self._model_name

        # Group chunk indices by the segment that contains them — one forward
        # pass per segment, not one per chunk.
        segment_chunk_map: list[list[int]] = [[] for _ in segments]
        for i, chunk in enumerate(chunks):
            for seg_idx, segment_text in enumerate(segments):
                if chunk.page_content in segment_text:
                    segment_chunk_map[seg_idx].append(i)
                    break
            else:
                logger.warning(
                    f"Chunk could not be assigned to any segment " f"(first 50 chars: '{chunk.page_content[:50]}...')"
                )

        for seg_idx, chunk_indices in enumerate(segment_chunk_map):
            if not chunk_indices:
                continue
            seg_chunks = [chunks[i] for i in chunk_indices]
            seg_vectors = self.embed_chunks(segments[seg_idx], seg_chunks)
            for ci, vec in zip(chunk_indices, seg_vectors):
                if vec is not None:
                    chunks[ci].metadata["_precomputed_vector"] = vec

        return chunks

    def _split_into_segments(self, text: str, batch_by: str) -> list[str]:
        """Split text into segments for multi-pass late chunking."""
        if batch_by == "page":
            # Split on form-feed characters (common in PDF text)
            pages = text.split("\f")
            if len(pages) <= 1:
                # Fallback: split on double newlines
                pages = text.split("\n\n\n")
            # Merge very short pages with the next one
            merged = []
            current = ""
            for page in pages:
                if len(current) + len(page) < self._max_context_tokens * 3:
                    current = current + ("\f" if current else "") + page
                else:
                    if current:
                        merged.append(current)
                    current = page
            if current:
                merged.append(current)
            return merged if merged else [text]

        # Default: split into fixed-size character windows
        window = self._max_context_tokens * 4  # approximate chars per token
        overlap = window // 10
        segments = []
        start = 0
        while start < len(text):
            end = min(start + window, len(text))
            segments.append(text[start:end])
            start += window - overlap
        return segments
