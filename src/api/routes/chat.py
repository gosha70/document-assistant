import json
import logging
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas import ChatRequest, ChatResponse, SourceCitation
from src.api.deps import get_retriever, get_generator
from src.config.settings import get_settings
from src.rag.generation import Generator
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
            if metrics:
                metrics.record_retrieval_no_source(settings.alerting.window_seconds)
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


@router.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """Stream answer tokens via Server-Sent Events. Sources are sent as a final event."""
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

    start = time.monotonic()
    documents = retriever.retrieve(request.question)
    if metrics:
        metrics.record_retrieval(time.monotonic() - start)

    if not documents:
        if metrics:
            metrics.record_retrieval_no_source(settings.alerting.window_seconds)

        def empty_stream():
            yield f"data: {json.dumps({'type': 'token', 'content': 'No relevant documents found for your question.'})}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    sources = Generator.extract_sources(documents)

    def event_stream():
        try:
            for token in generator.generate_stream(query=request.question, documents=documents):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
