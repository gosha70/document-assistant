import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import get_settings
from src.api.middleware.auth import AuthMiddleware
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
    """Create the LLM using the existing model loading infrastructure."""
    from models.model_info import ModelInfo
    from retrieval_qa import create_model

    model_info = ModelInfo()
    model_info.model_name = settings.model.default_model_name
    model_info.model_id = settings.model.default_model_id
    model_info.model_basename = settings.model.default_model_basename

    return create_model(model_info)


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

    yield

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
    app.add_middleware(TelemetryMiddleware)

    app.include_router(chat.router)
    app.include_router(ingest.router)
    app.include_router(status.router)
    app.include_router(admin.router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
