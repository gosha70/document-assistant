# Document Assistant — D.O.T. (Document Of Things)

A **Retrieval-Augmented Generation (RAG)** framework with a FastAPI backend, a browser-based chat UI, a full-featured admin console, and a configurable query pipeline. Designed for local-first, private deployments: with a local LLM backend (e.g., Ollama) and a local vector store, all data stays on your machine. When an external LLM backend is configured (`llm.backend: openai`), queries and retrieved context are sent to that endpoint.

[![CI](https://github.com/gosha70/document-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/gosha70/document-assistant/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: CC-BY-SA-4.0](https://img.shields.io/badge/License-CC--BY--SA--4.0-lightgrey)](LICENSE)

---

## Features at a Glance

| Area | Highlights |
|------|------------|
| **Ingestion** | 17 file types · PDF, Markdown, HTML, CSV, Excel, JSON, YAML, XML, SQL, Java, JS, Python, RTF, DDL, XSL · Duplicate handling (check / replace / add / skip) |
| **Chunking** | Configurable token-aware splitting · Late Chunking (long-context pooling) · Contextual Augmentation (LLM-generated per-chunk context) |
| **Retrieval** | Hybrid search (dense + BM25 sparse, fused via RRF) · CrossEncoder reranking · Four optional query pipeline stages |
| **Query Pipeline** | Query Decomposition · HyDE (Hypothetical Document Embeddings) · Corrective Retrieval · Answer Verification |
| **LLM Backends** | Ollama (local) · OpenAI-compatible APIs · Legacy GGUF/GPTQ/AWQ quantised models |
| **Vector Stores** | Chroma (dev/local) · Qdrant (production) — swappable via config |
| **API** | FastAPI REST + Server-Sent Events streaming · Interactive Swagger UI (`/docs`) · ReDoc (`/redoc`) |
| **Study Mode** | `/study/summarize` · `/study/glossary` · `/study/flashcards` — LLM-generated study aids |
| **Admin Console** | Collection management · Background jobs · Live metrics · Alerts · Periodic reports |
| **Security** | Bearer-token auth (user + admin roles) · Per-IP rate limiting · Prompt injection firewall · Upload validation |
| **Observability** | Request/LLM telemetry · Windowed alerting · Hourly snapshots |

---

## Architecture Overview

```mermaid
graph TB
    subgraph Clients
        CUI[Chat UI<br/>index.html]
        AUI[Admin Console<br/>admin.html]
        API[REST Client /<br/>Swagger UI]
    end

    subgraph FastAPI["FastAPI  (src/api/)"]
        MW[Middleware Stack<br/>Auth · RateLimit · Telemetry]
        IR[/ingest]
        CR[/chat<br/>/chat/stream]
        SR[/study]
        AR[/admin]
        ST[/status · /health]
    end

    subgraph Pipeline["Query Pipeline  (src/rag/)"]
        OR[QueryOrchestrator]
        QD[Query Decomposition]
        HY[HyDE]
        RT[Retriever<br/>Hybrid Search + Rerank]
        CR2[Corrective Retrieval]
        GN[Generator<br/>Prompt Templates]
        VF[Answer Verification]
    end

    subgraph Ingest["Ingestion Pipeline  (embeddings/)"]
        FC[File Converters<br/>17 types]
        CK[Chunker<br/>Standard / Late / Contextual]
        EM[Embedding Adapter<br/>Instructor / HuggingFace]
    end

    subgraph Backends["Vector Store Backends  (src/rag/)"]
        CH[ChromaBackend<br/>dev · local]
        QD2[QdrantBackend<br/>production]
    end

    subgraph LLM["LLM Backends"]
        OL[Ollama]
        OA[OpenAI-compatible]
        LG[Legacy GGUF/GPTQ]
    end

    CUI & AUI & API --> MW
    MW --> IR & CR & SR & AR & ST
    IR --> FC --> CK --> EM --> CH & QD2
    CR --> OR --> QD & HY --> RT --> CR2 --> GN --> VF
    RT --> CH & QD2
    GN --> OL & OA & LG
    AR --> CH & QD2
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- An LLM backend — [Ollama](https://ollama.com/) is the easiest for local use

### 1. Install dependencies

```bash
poetry install
```

### 2. Configure

Copy the default config and set your LLM backend:

```bash
cp config/defaults.yaml config/local.yaml
```

Edit `config/local.yaml` — minimum for Ollama:

```yaml
llm:
  backend: "ollama"          # or "openai"
  model: "llama3.2"          # any model pulled via `ollama pull`

vectorstore:
  persist_directory: "./my_vectorstore"
  collection_name: "my_docs"
```

All other settings have sensible defaults. See [Configuration Reference](#configuration-reference) below.

### 3. Start the server

```bash
poetry run uvicorn src.api.main:app --reload
```

Server starts at `http://127.0.0.1:8000`.

- **Chat UI** → `http://127.0.0.1:8000`
- **Admin Console** → `http://127.0.0.1:8000/static/admin.html`
- **Swagger UI** → `http://127.0.0.1:8000/docs`

### 4. Ingest documents

Upload files through the Chat UI (paperclip icon) or via the API:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -F "files=@/path/to/your.pdf" \
  -F "files=@/path/to/notes.md"
```

### 5. Ask questions

Use the Chat UI or send a request:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the document say about X?"}'
```

---

## Configuration Reference

Configuration is YAML-based. The application loads `config/defaults.yaml` and deep-merges `config/local.yaml` (if present) on top. Override the config path with `DOT_CONFIG_PATH=/path/to/config.yaml`.

### LLM Backend

```yaml
llm:
  backend: "ollama"            # "ollama" | "openai" | "legacy"
  model: "llama3.2"            # model name
  base_url: "http://localhost:11434/v1"  # omit for defaults
  api_key: null                # omit for local servers
  temperature: 0.2
  max_tokens: 512
```

### Vector Store

```yaml
vectorstore:
  backend: "chroma"            # "chroma" | "qdrant"
  collection_name: "my_docs"
  persist_directory: "./vectorstore"

  # Qdrant (when backend = "qdrant")
  qdrant_url: "http://localhost:6333"
  qdrant_api_key: null

  # Hybrid search (dense + BM25 + RRF)
  hybrid:
    enabled: true
    sparse_encoder: "bm25"
    rrf_k: 60
```

### Embedding Model

```yaml
embedding:
  model_name: "hkunlp/instructor-large"
  type: "instructor"           # "instructor" | "huggingface"
  device: "cpu"                # "cpu" | "cuda" | "mps"
```

### Chunking

```yaml
chunking:
  chunk_size: 512              # tokens
  chunk_overlap: 50            # tokens
  tokenizer: "tiktoken:cl100k_base"

  # Late Chunking (long-context document-aware embeddings)
  late_chunking:
    enabled: false
    model_name: "jinaai/jina-embeddings-v2-base-en"
    max_context_tokens: 8192
    file_types: [pdf, md, txt]

  # Contextual Augmentation (LLM-generated per-chunk context)
  contextual:
    enabled: false
    max_context_tokens: 128
```

### Query Pipeline

All stages are independently opt-in:

```yaml
query_pipeline:
  decomposition_enabled: false    # Break complex queries into sub-queries
  hyde_enabled: false             # Hypothetical Document Embeddings
  corrective_retrieval_enabled: false  # Re-retrieve if confidence < threshold
  corrective_retrieval_threshold: 0.4
  verification_enabled: false     # LLM verifies answer is grounded in sources
  max_sub_queries: 5
```

### Security

```yaml
auth:
  enabled: false               # Set true to require Bearer tokens
  admin_token: "secret-admin"  # /admin/* routes
  user_token: "secret-user"    # /chat, /ingest routes

rate_limit:
  enabled: false
  max_requests: 60
  window_seconds: 60
```

### Observability & Alerting

```yaml
telemetry:
  enabled: true
  log_llm_calls: true

alerting:
  enabled: true
  error_rate_threshold: 0.10        # 10 % error rate
  memory_rss_mb_threshold: 1024     # 1 GB RSS
  ingest_error_threshold: 5         # errors in window
  retrieval_no_source_threshold: 10
  window_seconds: 3600

reporting:
  enabled: true
  interval_seconds: 3600
  max_snapshots: 24
```

---

## API Reference

Full interactive documentation at `/docs` (Swagger UI) or `/redoc`.

### Core Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check — `{"status": "ok"}` |
| `GET` | `/status` | App info, collection name, document count, embedding model |
| `POST` | `/ingest` | Upload files (multipart) → ingest into vector store |
| `POST` | `/chat` | Synchronous Q&A with sources |
| `POST` | `/chat/stream` | Server-Sent Events streaming Q&A |

### Study Mode

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/study/summarize` | Generate a summary of retrieved context |
| `POST` | `/study/glossary` | Extract terms and definitions |
| `POST` | `/study/flashcards` | Generate flashcard Q&A pairs |

### Admin Endpoints (`/admin/*`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/admin/collections` | List all collections |
| `GET` | `/admin/collections/{name}` | Collection details (count, model, backend) |
| `GET` | `/admin/collections/{name}/chunks` | Sample chunks (max 50) |
| `GET` | `/admin/collections/{name}/sources` | List ingested source files |
| `POST` | `/admin/collections/{name}/reindex` | Trigger background reindex |
| `POST` | `/admin/collections/{name}/migrate` | Stamp embedding provenance on legacy collections |
| `GET` | `/admin/jobs` | List background jobs |
| `GET` | `/admin/jobs/{id}` | Job status and progress |
| `POST` | `/admin/jobs/{id}/retry` | Retry a failed job |
| `DELETE` | `/admin/jobs/{id}` | Cancel a running job |
| `GET` | `/admin/metrics/runtime` | Live process + application metrics |
| `GET` | `/admin/alerts` | Active alert conditions |
| `GET` | `/admin/reports/summary` | Snapshot report history |
| `POST` | `/admin/reports/generate` | Generate an on-demand snapshot |

### Chat Request / Response

```jsonc
// POST /chat
{
  "question": "What is the document about?",
  "collection_name": "my_docs",   // optional
  "use_history": false,
  "template_type": "generic"      // "generic" | "llama" | "mistral"
}

// Response
{
  "answer": "...",
  "sources": [
    {
      "file": "report.pdf",
      "page": 3,
      "excerpt": "...",
      "chunk_id": "abc123",
      "search_type": "hybrid"
    }
  ],
  "metadata": {
    "decomposed_queries": [...],
    "hyde_used": false,
    "retrieval_confidence": 0.87,
    "corrective_triggered": false,
    "verification": { "verified": true, "unsupported_claims": [] }
  }
}
```

### Ingest Request

```bash
# Single file
curl -X POST http://localhost:8000/ingest \
  -F "files=@document.pdf"

# Multiple files with duplicate strategy
curl -X POST http://localhost:8000/ingest \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.md" \
  -F "collection_name=my_collection" \
  -F "duplicate_strategy=replace"
```

`duplicate_strategy` options:
- `check` (default) — return `409` if any file was previously ingested
- `replace` — delete old chunks for matching filenames, then ingest
- `add` — ingest alongside existing chunks (no deduplication)
- `skip` — silently skip previously ingested filenames

---

## Supported File Types

| Extension | Type |
|-----------|------|
| `.pdf` | PDF |
| `.md` | Markdown |
| `.html` | HTML |
| `.txt` | Plain text |
| `.csv` | CSV |
| `.xlsx` | Excel |
| `.json` | JSON |
| `.yaml` | YAML |
| `.xml` | XML |
| `.xsl` | XSL |
| `.sql` | SQL |
| `.ddl` | DDL |
| `.java` | Java |
| `.js` | JavaScript |
| `.py` | Python |
| `.rtf` | Rich text |

Maximum upload size: 50 MB per file (configurable).

---

## Retrieval Pipeline

```
User Query
    │
    ▼
[Query Decomposition]  ←  optional: break into ≤5 sub-queries
    │
    ▼
[HyDE]                 ←  optional: generate hypothetical answer, embed it
    │
    ▼
[Hybrid Search]           dense (embedding) + sparse (BM25) → RRF fusion
    │
    ▼
[CrossEncoder Rerank]     ms-marco-MiniLM-L-6-v2, top-5 of 20 candidates
    │
    ▼
[Corrective Retrieval] ←  optional: re-retrieve if confidence < 0.4
    │
    ▼
[Prompt Injection Firewall]  sanitize retrieved context
    │
    ▼
[LLM Generation]          streamed or synchronous
    │
    ▼
[Answer Verification]  ←  optional: LLM checks claims against sources
    │
    ▼
Response + Sources + Metadata
```

---

## Security

### Authentication

When `auth.enabled: true`, all non-public endpoints require a `Bearer` token:

```bash
curl -H "Authorization: Bearer <user_token>" http://localhost:8000/chat ...
curl -H "Authorization: Bearer <admin_token>" http://localhost:8000/admin/collections
```

- `/health`, `/status`, `/docs`, `/openapi.json`, `/redoc` — always public
- `/chat`, `/ingest`, `/study/*` — require `user_token` or `admin_token`
- `/admin/*` — require `admin_token` only

### Prompt Injection Defense

Retrieved document content is sanitised before LLM assembly. Common injection patterns (`ignore previous instructions`, `you are now`, `act as`, `system:`, etc.) are redacted with `[REDACTED]` — providing a defense-in-depth layer alongside the system prompt guardrail.

### Rate Limiting

Per-IP sliding-window rate limiter. Returns `429 Too Many Requests` with a `Retry-After` header when the limit is exceeded.

---

## Admin Console

Access at `http://127.0.0.1:8000/static/admin.html`.

| Tab | Features |
|-----|----------|
| **Collections** | List collections · View chunk samples · List sources · Migrate legacy collections · Trigger reindex/export/import jobs |
| **Jobs** | Monitor background jobs · Retry failed · Cancel running |
| **Metrics** | Live request/error/LLM call counts · Memory and CPU · Active alerts |
| **Reports** | Hourly snapshots · On-demand report generation |

---

## Development

```bash
# Install all deps (including dev)
poetry install

# Lint
poetry run ruff check src/ tests/

# Security scan
poetry run bandit -r src/ -c pyproject.toml -q

# Unit tests (excluding e2e)
poetry run pytest -q --tb=short --ignore=tests/e2e

# Eval smoke tests
poetry run pytest tests/test_eval_smoke.py -v --tb=short

# E2e tests (requires a live server at localhost:8000)
poetry run pytest tests/e2e/ -v --tb=short
```

### Project Layout

```
src/
├── api/
│   ├── main.py               # App factory, middleware, lifespan
│   ├── routes/               # chat, ingest, admin, study, status
│   ├── middleware/           # auth, ratelimit, telemetry, upload validation
│   ├── schemas.py            # Pydantic request/response models
│   └── deps.py               # Singleton dependency wiring
├── rag/
│   ├── vectorstore.py        # VectorStoreBackend ABC
│   ├── chroma_backend.py     # Chroma implementation
│   ├── qdrant_backend.py     # Qdrant implementation
│   ├── retrieval.py          # Retriever (hybrid search + rerank)
│   ├── generation.py         # Generator (prompt templates + LLM)
│   ├── orchestrator.py       # QueryOrchestrator (full pipeline)
│   ├── embeddings.py         # Instructor / HuggingFace adapters
│   ├── reranking.py          # CrossEncoder reranker
│   ├── chunking.py           # Token-aware text splitting
│   ├── late_chunking.py      # Long-context pooled embeddings
│   ├── contextual.py         # LLM-augmented chunk context
│   ├── query_transform.py    # Decomposition, HyDE, corrective
│   ├── verification.py       # Answer grounding verification
│   ├── context_firewall.py   # Prompt injection sanitisation
│   ├── fusion.py             # RRF score fusion
│   ├── sparse.py             # BM25 sparse encoder
│   └── study_outputs.py      # Summary / glossary / flashcards
├── config/
│   └── settings.py           # Pydantic settings, YAML loader
└── utils/
    ├── metrics.py            # MetricsCollector
    ├── alerting.py           # AlertChecker
    ├── report_store.py       # Periodic snapshot store
    └── jobs.py               # Background job executor
embeddings/
└── unstructured/             # Per-type file converters (17 types)
prompts/                      # Versioned YAML prompt templates
config/
├── defaults.yaml             # Default configuration
└── local.yaml                # Local overrides (git-ignored)
static/
├── index.html                # Chat UI
└── admin.html                # Admin Console
tests/
├── test_*.py                 # Unit + integration tests (422)
└── e2e/                      # Playwright browser tests (13)
docs/
├── architecture.md           # System design and data-flow diagrams
└── adr/                      # Architecture Decision Records
```

---

## Architecture Decision Records

Key design decisions are documented in [`docs/adr/`](docs/adr/):

- [ADR-001](docs/adr/0001-fastapi-and-rest-api.md) — FastAPI REST API over legacy Dash/CLI
- [ADR-002](docs/adr/0002-hybrid-search.md) — Hybrid search: dense + BM25 + RRF
- [ADR-003](docs/adr/0003-pluggable-vector-store.md) — Pluggable vector store backend
- [ADR-004](docs/adr/0004-query-pipeline.md) — Composable query pipeline (HyDE, decomposition, corrective, verification)
- [ADR-005](docs/adr/0005-chunking-strategies.md) — Chunking strategies: standard, late, contextual

---

## License

[CC BY-SA 4.0](LICENSE) — Creative Commons Attribution-ShareAlike 4.0
