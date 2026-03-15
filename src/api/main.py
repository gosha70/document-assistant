import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.config.settings import get_settings
from src.api.middleware.auth import AuthMiddleware
from src.api.middleware.ratelimit import RateLimitMiddleware
from src.api.middleware.telemetry import TelemetryMiddleware
from src.api.routes import chat, ingest, status, admin

logger = logging.getLogger(__name__)


def _create_embedding(settings):
    """Create the embedding adapter based on config."""
    embedding_type = settings.embedding.type

    if embedding_type == "instructor":
        from src.rag.embeddings import InstructorEmbeddingAdapter

        return InstructorEmbeddingAdapter(
            model_name=settings.embedding.model_name,
            device=settings.embedding.device,
            normalize_embeddings=settings.embedding.normalize_embeddings,
        )
    elif embedding_type == "huggingface":
        from src.rag.embeddings import HuggingFaceEmbeddingAdapter

        return HuggingFaceEmbeddingAdapter(
            model_name=settings.embedding.model_name,
            device=settings.embedding.device,
            normalize_embeddings=settings.embedding.normalize_embeddings,
        )
    else:
        raise ValueError(f"Unknown embedding type: '{embedding_type}'. Supported: instructor, huggingface")


def _create_backend(settings):
    """Create the vector store backend based on config."""
    backend_type = settings.vectorstore.backend
    embedding = _create_embedding(settings)

    if backend_type == "chroma":
        from src.rag.chroma_backend import ChromaBackend

        return ChromaBackend(
            embedding=embedding,
            persist_directory=settings.vectorstore.persist_directory,
            embedding_type=settings.embedding.type,
            allow_legacy_collections=settings.vectorstore.allow_legacy_collections,
        )
    elif backend_type == "qdrant":
        from src.rag.qdrant_backend import QdrantBackend

        return QdrantBackend(
            embedding=embedding,
            url=settings.vectorstore.qdrant_url,
            api_key=settings.vectorstore.qdrant_api_key,
            prefer_grpc=settings.vectorstore.qdrant_prefer_grpc,
            embedding_type=settings.embedding.type,
            allow_legacy_collections=settings.vectorstore.allow_legacy_collections,
        )
    else:
        raise ValueError(f"Unknown vectorstore backend: '{backend_type}'. Supported: chroma, qdrant")


def _create_reranker(settings):
    """Create the reranker based on config. Returns None if disabled."""
    if not settings.reranker.enabled:
        return None
    from src.rag.reranking import CrossEncoderReranker

    return CrossEncoderReranker(model_name=settings.reranker.model_name)


def _create_llm(settings):
    """Create the LLM based on config.

    Supported backends:
      - "legacy": existing GGUF/GPTQ/AWQ loaders via retrieval_qa.create_model
      - "openai": OpenAI-compatible API (works with any OpenAI-compatible server)
      - "ollama": Ollama local server (uses OpenAI-compatible endpoint)
    """
    backend = settings.llm.backend

    if backend == "legacy":
        from models.model_info import ModelInfo
        from retrieval_qa import create_model

        model_info = ModelInfo()
        model_info.model_name = settings.model.default_model_name
        model_info.model_id = settings.model.default_model_id
        model_info.model_basename = settings.model.default_model_basename

        return create_model(model_info)

    elif backend == "openai":
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": settings.llm.model,
            "temperature": settings.llm.temperature,
            "max_tokens": settings.llm.max_tokens,
        }
        if settings.llm.base_url:
            kwargs["base_url"] = settings.llm.base_url
        if settings.llm.api_key:
            kwargs["api_key"] = settings.llm.api_key

        return ChatOpenAI(**kwargs)

    elif backend == "ollama":
        from langchain_openai import ChatOpenAI

        base_url = settings.llm.base_url or "http://localhost:11434/v1"
        model = settings.llm.model or "llama3.2"

        return ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=settings.llm.api_key or "ollama",
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
        )

    else:
        raise ValueError(f"Unknown LLM backend: '{backend}'. Supported: legacy, openai, ollama")


_report_stop_event = threading.Event()


def _schedule_report_loop(interval: int, window_seconds: int):
    """Run in a daemon thread. Snapshots metrics every `interval` seconds."""
    from src.utils.metrics import get_metrics_collector
    from src.utils.report_store import get_report_store

    while not _report_stop_event.is_set():
        _report_stop_event.wait(interval)
        if _report_stop_event.is_set():
            break
        try:
            collector = get_metrics_collector()
            stats = collector.get_stats(window_seconds=2 * window_seconds)
            snapshot = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": stats["uptime_seconds"],
                "request_count": stats["request_count"],
                "error_count": stats["error_count"],
                "llm_call_count": stats["llm_call_count"],
                "ingest_total_docs": stats["ingest_total_docs"],
                "ingest_error_count": stats["ingest_error_count"],
                "retrieval_no_source_count": stats["retrieval_no_source_count"],
            }
            get_report_store().add(snapshot)
            logger.debug("Periodic report snapshot generated")
        except Exception as e:
            logger.warning(f"Periodic report snapshot failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle for the FastAPI application."""
    settings = get_settings()
    logger.info(f"Starting {settings.app.name}")

    from src.api.deps import set_vectorstore_backend, set_llm, set_reranker

    # Initialise vector store backend (config-driven)
    backend = _create_backend(settings)
    set_vectorstore_backend(backend)
    logger.info(f"Vector store backend ready: {settings.vectorstore.backend}")

    # Auto-migrate legacy collections: stamp provenance on any collection missing it
    try:
        for col in backend.list_collections():
            name = col["name"]
            if backend.get_embedding_provenance(name) is None:
                backend._set_provenance(name)
                logger.info(
                    f"Migrated legacy collection '{name}' — stamped provenance: {backend._embedding.model_name}"
                )
    except Exception as e:
        logger.warning(f"Legacy collection migration check failed: {e}")

    # Initialise reranker (config-driven)
    reranker = _create_reranker(settings)
    set_reranker(reranker)
    if reranker:
        logger.info(f"Reranker ready: {settings.reranker.model_name}")
    else:
        logger.info("Reranker disabled")

    # Initialise LLM
    try:
        llm = _create_llm(settings)
        set_llm(llm)
        logger.info("LLM ready")
    except Exception as e:
        logger.warning(f"LLM init failed ({e}); /chat will return 503 until an LLM is available")

    # Start periodic report thread
    if settings.reporting.enabled and settings.telemetry.enabled:
        from src.utils.report_store import get_report_store, ReportStore

        # Ensure store uses configured max_snapshots
        store = get_report_store()
        if store._snapshots.maxlen != settings.reporting.max_snapshots:
            store._snapshots = __import__("collections").deque(
                store._snapshots, maxlen=settings.reporting.max_snapshots
            )
        _report_stop_event.clear()
        report_thread = threading.Thread(
            target=_schedule_report_loop,
            args=(settings.reporting.interval_seconds, settings.alerting.window_seconds),
            daemon=True,
        )
        report_thread.start()
        logger.info(f"Periodic report thread started (interval={settings.reporting.interval_seconds}s)")

    yield

    # Stop periodic report thread
    _report_stop_event.set()

    # Shutdown background job executor and reset singletons
    from src.utils.jobs import reset_jobs

    reset_jobs()
    logger.info("Shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app.name,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(AuthMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(TelemetryMiddleware)

    app.include_router(chat.router)
    app.include_router(ingest.router)
    app.include_router(status.router)
    app.include_router(admin.router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/")
    def root():
        return RedirectResponse(url="/static/index.html")

    # Serve the chat UI from /static
    _static_dir = os.path.join(os.path.dirname(__file__), "..", "..", "static")
    if os.path.isdir(_static_dir):
        app.mount("/static", StaticFiles(directory=_static_dir), name="static")

    # Serve existing icons and CSS from /assets
    _assets_dir = os.path.join(os.path.dirname(__file__), "..", "..", "app", "assets")
    if os.path.isdir(_assets_dir):
        app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")

    return app


app = create_app()
