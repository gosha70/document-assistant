# ADR-001: FastAPI REST API over Legacy CLI / Dash Application

**Status:** Accepted
**Date:** 2024
**Deciders:** @gosha70

---

## Context

The original D.O.T. application was a command-line tool paired with a Dash frontend. Users would:

1. Run `python3 -m embeddings.embedding_database ...` to build a vector store
2. Run `python3 -m app.chat_app ...` to launch a Dash-based chat interface

This model had several limitations:

- **No REST API** — integrations with other systems required forking the process or importing internal modules
- **Dash coupling** — the UI and backend were tightly coupled; changing one required touching the other
- **Single-user** — the CLI/Dash model assumed a single user session per process
- **No streaming** — Dash's synchronous request model made token streaming difficult
- **No admin operations** — managing collections, monitoring ingestion, viewing metrics required direct filesystem access or custom scripts

## Decision

Replace the CLI + Dash stack with a **FastAPI** application serving:

- A REST API (`/chat`, `/ingest`, `/study/*`, `/admin/*`, `/status`, `/health`)
- Server-Sent Events streaming (`/chat/stream`)
- Static HTML frontends for the Chat UI and Admin Console (decoupled from the backend)
- Interactive API documentation via Swagger UI (`/docs`) and ReDoc (`/redoc`)

The embedding CLI (`embeddings/`) is retained for offline/batch workflows but is no longer the primary ingestion path.

## Rationale

| Criterion | CLI + Dash | FastAPI |
|-----------|-----------|---------|
| API-first integration | No | Yes |
| Streaming responses | No | Yes (SSE) |
| Admin operations | Manual | REST endpoints |
| Multi-user (concurrent) | No | Yes (async) |
| Interactive API docs | No | Built-in |
| Frontend flexibility | Dash-locked | Any HTTP client |
| Observability | Print-based | Middleware metrics |

FastAPI was chosen over Flask/Django for:

- Native async support (important for SSE and concurrent ingestion)
- Automatic Pydantic schema validation and OpenAPI generation
- Minimal boilerplate for route definition

## Consequences

**Positive:**
- Any HTTP client can integrate with D.O.T. (curl, Python httpx, JavaScript fetch)
- Streaming token output works natively over SSE
- Admin operations (reindex, export, job monitoring, metrics) are first-class API citizens
- Frontend can be any static HTML/JS without backend coupling
- OpenAPI schema enables client generation

**Negative:**
- The old `python3 -m app.chat_app` invocation no longer works — users must start the FastAPI server
- Dash-specific UI customisation (color schemes, `app_config.json`) is no longer supported
- The server is stateful (singletons for embedding model, vector store) — horizontal scaling requires shared state

## Migration Notes

The legacy `embeddings/` CLI still works for batch pre-ingestion:

```bash
python3 -m embeddings.embedding_database --dir_path ./docs --file_types pdf --persist_directory ./mydb
```

After building the store, point the FastAPI server at the same `persist_directory` in `config/local.yaml`.
