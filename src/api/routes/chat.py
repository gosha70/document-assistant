import logging
import time
from fastapi import APIRouter, HTTPException

from src.api.schemas import ChatRequest, ChatResponse, SourceCitation
from src.api.deps import get_retriever, get_generator
from src.config.settings import get_settings
from src.utils.metrics import get_metrics_collector

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Ask a question against the indexed corpus. Returns answer with source citations."""
    try:
        retriever = get_retriever(request.collection_name)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        generator = get_generator(request.template_type, request.use_history)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    settings = get_settings()
    metrics = get_metrics_collector() if settings.telemetry.enabled else None

    try:
        start = time.monotonic()
        documents = retriever.retrieve(request.question)
        if metrics:
            metrics.record_retrieval(time.monotonic() - start)

        if not documents:
            return ChatResponse(answer="No relevant documents found for your question.", sources=[])

        result = generator.generate(
            query=request.question,
            documents=documents,
        )

        sources = [
            SourceCitation(
                file=s["file"],
                page=s.get("page"),
                excerpt=s["excerpt"],
            )
            for s in result["sources"]
        ]

        return ChatResponse(answer=result["answer"], sources=sources)

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
