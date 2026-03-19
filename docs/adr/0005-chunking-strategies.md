# ADR-005: Chunking Strategies — Standard, Late Chunking, Contextual Augmentation

**Status:** Accepted
**Date:** 2024
**Deciders:** @gosha70

---

## Context

Chunking is the most consequential decision in a RAG pipeline. The embedding of a chunk must both:

1. Represent the chunk's content well enough for similarity search
2. Contain enough surrounding context so that the LLM can produce a coherent answer from it

Standard fixed-size chunking is a compromise that frequently fails in both directions:

- **Too small** — the chunk lacks context; the generated answer is incomplete or misleading
- **Too large** — the chunk embedding averages over too much content; retrieval precision drops

Two advanced techniques address this at different levels:

- **Late Chunking** — uses a long-context embedding model to embed the full document, then pools the per-token embeddings for each chunk. The chunk embeddings are aware of the full document context.
- **Contextual Augmentation** — uses an LLM to generate a short contextual description for each chunk (summarising where in the document the chunk appears and what the surrounding content is), then prepends this description to the chunk text before embedding.

Both techniques increase ingestion cost. Late chunking requires a long-context embedding model pass over the full document; contextual augmentation requires an LLM call per chunk. Neither should be the default.

## Decision

Implement **three mutually exclusive chunking strategies**:

| Strategy | Activation | Description |
|----------|-----------|-------------|
| **Standard** | Default | Token-aware splitting using tiktoken or HuggingFace tokenizers |
| **Late Chunking** | `chunking.late_chunking.enabled: true` | Long-context model pass → per-token embeddings → chunk-level pooling |
| **Contextual Augmentation** | `chunking.contextual.enabled: true` | LLM generates context description per chunk → prepended to embedding text |

All three strategies produce the same output format: LangChain `Document` objects with `page_content` (for display/citation) and `metadata` (including `embedding_strategy` for provenance tracking).

### Standard Chunking (Default)

Splits text recursively by paragraph, sentence, and word boundaries up to `chunk_size` tokens, with `chunk_overlap` token overlap between consecutive chunks. The tokenizer is configurable: `tiktoken:cl100k_base` (default) or any HuggingFace tokenizer.

Metadata added: `token_count`, `tokenizer`, `chunking_version`.

### Late Chunking

Process:
1. Load the document as plain text (up to `max_context_tokens`, default 8192)
2. Split into standard chunks (same splitter as standard strategy)
3. Run the full text through a long-context model (`jinaai/jina-embeddings-v2-base-en`) to get per-token embeddings
4. For each chunk, identify its token span in the full document and mean-pool the token embeddings within that span
5. Store the pooled embedding as the chunk's vector representation

The key property: each chunk's embedding is computed from the full document context, not from the chunk in isolation.

File types eligible for late chunking: `[pdf, md, txt]` (configurable). Code files are excluded — their structure is already well-captured by language-aware splitting.

Metadata added: `embedding_strategy: late_chunking`.

### Contextual Augmentation

Process:
1. Extract the full document text (up to `document_summary_tokens` characters, default 256)
2. Split into standard chunks
3. For each chunk, call the LLM with a prompt (`prompts/chunk_context.yaml`) that includes the full document summary and the chunk text, asking for a one-sentence contextual description
4. Prepend the context description to the chunk text before embedding
5. Store the original chunk text as `page_content` (for citation display)

The context description is truncated to `max_context_tokens` (default 128) and stored in `metadata["chunk_context"]`.

Metadata added: `embedding_strategy: contextual`.

## Rationale

| Criterion | Standard | Late Chunking | Contextual |
|-----------|----------|--------------|-----------|
| Ingestion cost | Low | Medium (one long-context model pass per doc) | High (one LLM call per chunk) |
| Context awareness | None | Full document context | Neighbouring context via LLM |
| Retrieval quality uplift | Baseline | High for narrative docs | High for all doc types |
| LLM dependency at ingest | No | No | Yes |
| Suited for | All use cases | Long narrative documents (PDF, reports, books) | Any document type |

Late chunking and contextual augmentation were kept as opt-in because:

- Standard chunking already performs well for most technical documents (code, API references, short articles)
- Late chunking requires a specific long-context model to be downloaded (~1.5 GB)
- Contextual augmentation multiplies ingestion time proportionally to the number of chunks
- The local-first nature of D.O.T. means some users run on resource-constrained hardware

They are mutually exclusive (enabling both raises a configuration error) because combining them would produce unpredictable provenance semantics.

## Consequences

**Positive:**
- Standard chunking is always available with no additional model dependencies
- Late chunking significantly improves retrieval for long-form documents where context matters (legal, academic, narrative)
- Contextual augmentation improves retrieval across all document types by reducing the semantic gap between the chunk and its surrounding context
- All three strategies are tracked via `embedding_strategy` in provenance, so different strategies can coexist across collections

**Negative:**
- Late chunking is limited to documents fitting within `max_context_tokens` (8192 default) — very long documents are truncated
- Contextual augmentation quality depends on the LLM — poor models produce poor context descriptions
- Mixing strategies within the same collection is **actively rejected** by both backends — ingesting documents with a different `embedding_strategy` into an existing collection raises a `ValueError` at runtime; use a separate collection for each strategy
- Neither advanced strategy has a quantitative benchmark in this codebase — `eval/` tests are smoke tests only

## Configuration

```yaml
chunking:
  chunk_size: 512
  chunk_overlap: 50
  tokenizer: "tiktoken:cl100k_base"

  late_chunking:
    enabled: false
    model_name: "jinaai/jina-embeddings-v2-base-en"
    max_context_tokens: 8192
    pooling_strategy: "mean"
    batch_by: "page"
    file_types: [pdf, md, txt]

  contextual:
    enabled: false
    max_context_tokens: 128       # characters prepended to chunk
    document_summary_tokens: 256  # characters of doc summary sent to LLM
```
