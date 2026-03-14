import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import get_settings
from src.api.middleware.auth import AuthMiddleware
from src.api.middleware.telemetry import TelemetryMiddleware
from src.api.routes import chat, ingest, status, admin

logger = logging.getLogger(__name__)


def _create_backend(settings):
    """Create the vector store backend based on config."""
    backend_type = settings.vectorstore.backend

    if backend_type == "chroma":
        from src.rag.chroma_backend import ChromaBackend
        from src.rag.embeddings import InstructorEmbeddingAdapter

        embedding = InstructorEmbeddingAdapter(
            model_name=settings.embedding.model_name,
            device=settings.embedding.device,
            normalize_embeddings=settings.embedding.normalize_embeddings,
        )
        return ChromaBackend(
            embedding=embedding,
            persist_directory=settings.vectorstore.persist_directory,
        )
    else:
        raise ValueError(f"Unknown vectorstore backend: '{backend_type}'. Supported: chroma")


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

    from src.api.deps import set_vectorstore_backend, set_llm

    # Initialise vector store backend (config-driven)
    backend = _create_backend(settings)
    set_vectorstore_backend(backend)
    logger.info(f"Vector store backend ready: {settings.vectorstore.backend}")

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
