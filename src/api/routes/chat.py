import json
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas import ChatRequest, ChatResponse, SourceCitation
from src.api.deps import get_orchestrator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Ask a question against the indexed corpus. Returns answer with source citations."""
    try:
        orchestrator = get_orchestrator(
            request.collection_name,
            request.template_type,
            request.use_history,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        result = orchestrator.run(request.question)

        sources = [
            SourceCitation(
                file=s["file"],
                page=s.get("page"),
                excerpt=s["excerpt"],
            )
            for s in result["sources"]
        ]

        return ChatResponse(
            answer=result["answer"],
            sources=sources,
            metadata=result.get("metadata"),
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """Stream answer tokens via Server-Sent Events. Sources are sent as a final event."""
    try:
        orchestrator = get_orchestrator(
            request.collection_name,
            request.template_type,
            request.use_history,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    documents, token_iter, get_sources, metadata = orchestrator.run_stream(
        request.question,
    )

    if documents is None:

        def empty_stream():
            yield f"data: {json.dumps({'type': 'token', 'content': 'No relevant documents found for your question.'})}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    def event_stream():
        try:
            for token in token_iter:
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': get_sources()})}\n\n"
            if "verification" in metadata:
                yield f"data: {json.dumps({'type': 'verification', 'result': metadata['verification']})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
