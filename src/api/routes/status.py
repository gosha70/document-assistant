import logging
from fastapi import APIRouter

from src.api.schemas import StatusResponse
from src.api.deps import get_vectorstore_backend
from src.config.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["status"])


@router.get("/status", response_model=StatusResponse)
def get_status():
    """Return current system status: document count, model info, backend."""
    settings = get_settings()
    backend = get_vectorstore_backend()

    try:
        doc_count = backend.count(settings.vectorstore.collection_name)
    except Exception:
        doc_count = 0

    provenance = backend.get_embedding_provenance(settings.vectorstore.collection_name)
    embedding_model = provenance["model_name"] if provenance else "unknown"

    return StatusResponse(
        app_name=settings.app.name,
        document_count=doc_count,
        collection_name=settings.vectorstore.collection_name,
        embedding_model=embedding_model,
        vectorstore_backend=settings.vectorstore.backend,
    )
