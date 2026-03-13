# ML/AI — RAG + Knowledge Graph

## Stack
- Python 3.11+, dependency management via Poetry
- FastAPI for the API layer
- Vector store: Chroma (dev) / Pinecone or Qdrant (prod) ← UPDATE per project
- Knowledge graph: Neo4j (Cypher queries, neo4j Python driver)
- Embeddings: OpenAI text-embedding-3-small ← UPDATE per project
- LLM: GPT-4o / Claude via LiteLLM ← UPDATE per project
- Frontend: Streamlit for POC; React+Vite for production UI

## Project Structure
```
src/
├── api/                  # FastAPI routes + middleware
├── rag/
│   ├── chunking.py       # Document chunking (configurable strategy)
│   ├── embeddings.py     # Embedding generation + caching
│   ├── retrieval.py      # Hybrid search: vector + keyword + graph
│   └── generation.py     # LLM response with source citations
├── graph/
│   ├── schema.py         # Node/relationship type definitions
│   ├── ingestion.py      # Entity + relation extraction pipeline
│   └── queries.py        # Parameterized Cypher query templates
├── config/               # Pydantic settings, .env loading
└── utils/
prompts/                  # Versioned YAML prompt templates
eval/                     # Evaluation datasets + harnesses
tests/
specs/                    # SDD artifacts and lessons learned
poc/                      # Streamlit prototypes
notebooks/                # Exploration
```

## Architecture Rules
> **Non-negotiable.** Violations must be flagged during review, not silently accepted.

- Default to hybrid search (vector + BM25) before graph traversal
- All prompts in versioned YAML files — never inline strings
- Every LLM call must be traced with latency + token logging
- Graph schema changes require migration scripts
- All retrieval results must include source references

## Key Conventions
- All prompts: versioned YAML in `prompts/`, never inline strings
- Every LLM call: wrapped in traced function with latency + token logging
- Embeddings model: pinned in config, never hardcoded
- Chunking: strategy + chunk_size + overlap configurable via settings
- Graph schema: defined in code, changes via migration scripts
- All retrieval results must include source references for citation

## RAG Rules
- Default to hybrid search (vector + BM25 keyword) before graph traversal
- Graph traversal depth: configurable, default 2 hops
- Chunk overlap: 10-15% of chunk size
- Never embed raw HTML/markdown; always clean to plain text first
- Reranking step required before final context assembly

## Testing
- `pytest` with fixtures for mock LLM responses (no live API in unit tests)
- Retrieval eval: recall@k and precision@k on labeled dataset
- Answer quality: LLM-as-judge with rubrics defined in `eval/rubrics/`
- Graph tests: schema constraint validation, Cypher query correctness

## Commands
```bash
poetry install                                          # deps
poetry run pytest                                       # tests
poetry run uvicorn src.api.main:app --reload            # API
poetry run streamlit run poc/app.py                     # POC UI
docker compose up -d                                    # Neo4j + vector DB
```

## Agent Team

### Roles

| Role | Trigger | Owns |
|------|---------|------|
| **Team Lead** (default) | Planning, architecture decisions, code review, user comms | Overall coordination |
| **RAG Engineer** | Chunking, retrieval, generation, prompt engineering | `src/rag/`, `prompts/` |
| **Knowledge Graph Engineer** | Neo4j schema, entity extraction, Cypher queries | `src/graph/` |
| **Data Analyst** | Evaluation harnesses, metrics, benchmarking, data exploration | `eval/`, `notebooks/` |
| **QA Engineer** | Testing, CI, code review for bugs/security | `tests/`, all test files |

### Team Lead — Default Behavior
You ARE the Team Lead. For every user request:
1. Assess complexity. Single-file, single-domain changes → handle directly.
2. Multi-domain or >50 lines of specialized code → delegate to specialist sub-agent.
3. Always review sub-agent output against project conventions before presenting.
4. Coordinate when a task spans domains (e.g., new entity type needs Graph + RAG + tests).

### Delegation Prompts
When spawning a sub-agent via Task tool, use this pattern:
```
You are the [ROLE] on a RAG + Knowledge Graph project.

Project conventions:
- [paste relevant section from this CLAUDE.md]

Your task: [specific task description]

Constraints:
- Follow all conventions above
- Write type-annotated Python with docstrings
- Do NOT modify files outside your ownership area without noting it
- Return: code changes + brief summary of decisions made
```

### RAG Engineer
Expertise: document chunking strategies, embedding models, vector search, hybrid retrieval, reranking, prompt engineering, LLM integration.
Constraints: never hardcode model names; always use config. Every retrieval function must return source metadata. All prompts go in `prompts/` as YAML.

### Knowledge Graph Engineer
Expertise: Neo4j, Cypher query optimization, entity/relation extraction, graph schema design, graph traversal algorithms.
Constraints: all schema changes via migration scripts. Parameterize all Cypher queries (no string interpolation). Test all queries against sample data.

### Data Analyst
Expertise: evaluation framework design, retrieval metrics (recall@k, precision@k, MRR), answer quality rubrics, statistical analysis, data visualization.
Constraints: never modify production data. All eval configs in `eval/`. Document methodology in notebook markdown cells.

### QA Engineer
Expertise: pytest, test architecture, mocking LLM calls, integration testing, security review, CI/CD pipeline.
Constraints: run existing tests before writing new ones. Mock all external APIs. Coverage targets: ≥80% on `src/rag/` and `src/graph/`. Flag any untested code paths.
