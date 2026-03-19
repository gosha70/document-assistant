# D.O.T. — System Architecture

## Overview

D.O.T. is a **local-first RAG application** built around three concerns:

1. **Ingestion** — transform raw documents into searchable vector representations
2. **Retrieval** — find the most relevant chunks for a user query using hybrid search
3. **Generation** — produce a grounded answer using an LLM and the retrieved context

All three concerns are exposed through a **FastAPI** service with a thin browser-based frontend.

---

## System Layers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            Browser Clients                               │
│          Chat UI (index.html)        Admin Console (admin.html)          │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │ HTTP / SSE
┌───────────────────────────────────▼─────────────────────────────────────┐
│                          FastAPI Application                              │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Middleware Stack                               │   │
│  │  AuthMiddleware → RateLimitMiddleware → TelemetryMiddleware       │   │
│  └──────────────────────────┬───────────────────────────────────────┘   │
│                             │                                            │
│  ┌──────────┐ ┌──────────┐ ┌▼─────────┐ ┌──────────┐ ┌─────────────┐  │
│  │ /ingest  │ │ /chat    │ │ /study   │ │ /admin   │ │ /status     │  │
│  │          │ │ /chat/   │ │ summarize│ │ collects │ │ /health     │  │
│  │          │ │ stream   │ │ glossary │ │ jobs     │ │             │  │
│  │          │ │          │ │ flashcards│ │ metrics  │ │             │  │
│  └──────┬───┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └─────────────┘  │
└─────────┼──────────┼────────────┼─────────────┼───────────────────────┘
          │          │            │             │
          ▼          ▼            ▼             ▼
┌─────────────┐  ┌────────────────────────┐  ┌──────────────────────────┐
│  Ingestion  │  │    Query Orchestrator  │  │  Admin Services          │
│  Pipeline   │  │                        │  │  JobStore · MetricsStore │
│             │  │  1. Decomposition      │  │  AlertChecker            │
│  Converters │  │  2. HyDE               │  │  ReportStore             │
│  Chunker    │  │  3. Retriever          │  │                          │
│  Embedder   │  │     hybrid_search()    │  └──────────────────────────┘
│             │  │     rerank()           │
└──────┬──────┘  │  4. Corrective Retr.  │
       │         │  5. Context Firewall   │
       │         │  6. Generator          │
       │         │  7. Verification       │
       │         └────────────┬───────────┘
       │                      │
       ▼                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Vector Store Backend                       │
│                                                             │
│   ChromaBackend (dev/local)    QdrantBackend (production)   │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │  Dense Search  +  BM25 Sparse Search  →  RRF Fusion │  │
│   └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
              │                           │
              ▼                           ▼
      ┌───────────────┐         ┌───────────────────┐
      │  Embedding    │         │  LLM Backend      │
      │  Adapter      │         │                   │
      │               │         │  Ollama           │
      │  Instructor   │         │  OpenAI-compat.   │
      │  HuggingFace  │         │  Legacy GGUF/GPTQ │
      └───────────────┘         └───────────────────┘
```

---

## Data Flow: Document Ingestion

```
HTTP POST /ingest (multipart files)
            │
            ▼
┌─────────────────────────┐
│   Upload Validation      │  ← extension whitelist, size check,
│                          │    path traversal guard, ".." rejection
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Duplicate Detection    │  ← find_by_source() on backend
│                          │    strategy: check | replace | add | skip
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   File Converter         │  ← per-extension converter (17 types)
│                          │    language-aware splitting for code files
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────┐
│   Chunking Strategy                                  │
│                                                     │
│   Standard        Late Chunking    Contextual Aug.  │
│   ──────────      ─────────────    ───────────────  │
│   token-aware     long-context     LLM-generated    │
│   splitting       pooled embeds    per-chunk context │
│   (default)       (jinaai model)   prepended to text │
└──────────┬──────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│   Embedding Adapter      │  ← Instructor | HuggingFace
│                          │    provenance stamped (model, strategy)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Vector Store Backend   │  ← store() with rollback on failure
│   + BM25 Index Update    │
└─────────────────────────┘
```

---

## Data Flow: Query Pipeline

```
User Question
      │
      ▼
┌──────────────────────────────────────────────────────────────────┐
│  Query Transformation  (optional — each stage independently set) │
│                                                                  │
│  Decomposition:  LLM splits into ≤5 sub-queries                  │
│                  → retrieve separately, merge, rerank together   │
│                                                                  │
│  HyDE:           LLM generates hypothetical answer passage       │
│                  → embed passage, retrieve by embedding          │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  Hybrid Retrieval                                                │
│                                                                  │
│  Dense search: embedding similarity (top-20)                     │
│  Sparse search: BM25 token overlap (top-20)                      │
│  Fusion: Reciprocal Rank Fusion  score = Σ 1/(k + rank)          │
│                                                                  │
│  CrossEncoder Rerank: ms-marco-MiniLM-L-6-v2 (top-5 of 20)      │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  Corrective Retrieval  (optional)                                │
│                                                                  │
│  confidence = sigmoid(mean reranker score)                       │
│  if confidence < threshold (default 0.4) → re-retrieve           │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  Context Firewall                                                │
│                                                                  │
│  Regex-based prompt injection sanitisation on retrieved text     │
│  Matched patterns replaced with [REDACTED]                       │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  LLM Generation                                                  │
│                                                                  │
│  YAML prompt template loaded (prompts/qa.yaml)                   │
│  Template rendered: {system_prompt} + {context} + {question}     │
│  Optional history: {history} injected                            │
│  LLM call traced: latency + prompt/response chars                │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  Answer Verification  (optional)                                 │
│                                                                  │
│  LLM checks each claim against retrieved sources                 │
│  Returns: verified, unsupported_claims, revised_answer           │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
              Answer + Sources + Metadata
```

---

## Middleware Stack

Requests pass through middleware in this order before reaching route handlers:

```
Request
  │
  ▼
AuthMiddleware
  Checks Authorization: Bearer <token> header
  Public paths bypass: /health, /status, /docs, /openapi.json, /redoc
  /admin/* requires admin_token
  /chat, /ingest, /study/* require user_token or admin_token
  │
  ▼
RateLimitMiddleware
  Per-IP sliding window (in-memory)
  Returns 429 + Retry-After on breach
  │
  ▼
TelemetryMiddleware
  Increments request_count on every request
  Increments error_count on 5xx responses
  │
  ▼
Route Handler
```

---

## Configuration Architecture

All runtime behaviour is driven by YAML:

```
config/defaults.yaml          ← committed, full defaults
config/local.yaml             ← git-ignored, per-deployment overrides
       │
       ▼ deep-merged at startup
src/config/settings.py        ← Pydantic Settings model
       │
       ▼ injected via
src/api/deps.py               ← singleton accessors (backend, llm, embedder, reranker)
src/api/main.py               ← lifespan context manager bootstraps all singletons
```

Override path: `DOT_CONFIG_PATH=/path/to/myconfig.yaml`

---

## Observability Architecture

```
Every Request
      │
      ▼
TelemetryMiddleware  ──────► MetricsCollector
                               ├── request_count
                               ├── error_count
                               ├── llm_call_count
                               ├── llm_total_latency
                               ├── ingest_doc_count
                               ├── ingest_error_timestamps (deque)
                               └── retrieval_no_source_timestamps (deque)
                                         │
                        ┌────────────────┼─────────────────┐
                        ▼                ▼                  ▼
                 AlertChecker      ReportStore        /admin/metrics
                 (stateless)       (daemon thread     /runtime
                 threshold eval    hourly snapshots)
                        │
                        ▼
                 /admin/alerts
```

---

## Embedding Provenance Tracking

Every collection records embedding metadata to detect model mismatches:

```json
{
  "model_name": "hkunlp/instructor-large",
  "type": "instructor",
  "embedding_strategy": "standard"   // or "late_chunking" | "contextual"
}
```

This provenance is checked when loading a collection — a mismatch between the stored provenance and the currently configured embedding model is flagged. The `/admin/collections/{name}/migrate` endpoint stamps provenance on legacy collections that predate this feature.

---

## Background Job Architecture

```
POST /admin/collections/{name}/reindex
POST /admin/collections/{name}/export
POST /admin/collections/import
              │
              ▼
         JobStore
         ├── job_id (UUID)
         ├── status: queued | running | completed | failed
         ├── progress (0-100)
         └── error message
              │
              ▼
       JobExecutor (thread pool)
              │
              ▼ updates status/progress
         JobStore ──► GET /admin/jobs/{id}
```

Jobs are in-memory only — they do not survive a server restart.

---

## Security Layers

| Layer | Mechanism | Default |
|-------|-----------|---------|
| Transport | Bind to `127.0.0.1` by default | Localhost only |
| Authentication | Bearer token middleware | Disabled |
| Authorisation | Role-based path matching (user / admin) | Disabled |
| Rate limiting | Per-IP sliding window | Disabled |
| Upload validation | Extension whitelist + size + path traversal guard | Always on |
| Prompt injection | Context firewall (regex redaction) | Always on |
| Stack traces | Never exposed in HTTP responses | Always on |
| BM25 path | SHA-256 hash of collection name + symlink traversal guard | Always on |
