import logging
from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    FlashcardsRequest,
    FlashcardsResponse,
    Flashcard,
    GlossaryResponse,
    GlossaryTerm,
    SourceCitation,
    StudyRequest,
    SummaryResponse,
)
from src.api.deps import get_retriever
from src.rag.generation import Generator
from src.rag.study_outputs import StudyOutputGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/study", tags=["study"])


def _get_generator() -> StudyOutputGenerator:
    from src.api.deps import _llm as llm

    if llm is None:
        raise RuntimeError("LLM not initialised. Call set_llm() at startup.")
    return StudyOutputGenerator(llm)


def _build_sources(documents) -> list[SourceCitation]:
    return [
        SourceCitation(
            file=s["file"],
            page=s.get("page"),
            excerpt=s["excerpt"],
            chunk_id=s.get("chunk_id"),
            search_type=s.get("search_type"),
            full_excerpt=s.get("full_excerpt"),
        )
        for s in Generator.extract_sources(documents)
    ]


@router.post("/summarize", response_model=SummaryResponse)
def summarize(request: StudyRequest):
    """Summarise the indexed collection around a query topic, with source citations."""
    try:
        gen = _get_generator()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        docs = get_retriever(request.collection_name, k=request.k).retrieve(request.query)
        if not docs:
            raise HTTPException(status_code=404, detail="No relevant documents found.")
        summary = gen.summarize(docs)
        return SummaryResponse(summary=summary, sources=_build_sources(docs))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Study/summarize error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/glossary", response_model=GlossaryResponse)
def glossary(request: StudyRequest):
    """Extract a glossary of key terms and definitions from the indexed collection."""
    try:
        gen = _get_generator()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        docs = get_retriever(request.collection_name, k=request.k).retrieve(request.query)
        if not docs:
            raise HTTPException(status_code=404, detail="No relevant documents found.")
        items = gen.extract_glossary(docs)
        terms = [
            GlossaryTerm(term=i["term"], definition=i["definition"], source=i.get("source"))
            for i in items
            if "term" in i and "definition" in i
        ]
        return GlossaryResponse(terms=terms, sources=_build_sources(docs))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Study/glossary error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/flashcards", response_model=FlashcardsResponse)
def flashcards(request: FlashcardsRequest):
    """Generate study flashcards from the indexed collection."""
    try:
        gen = _get_generator()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        docs = get_retriever(request.collection_name, k=request.k).retrieve(request.query)
        if not docs:
            raise HTTPException(status_code=404, detail="No relevant documents found.")
        items = gen.generate_flashcards(docs, request.max_cards)
        cards = [
            Flashcard(front=i["front"], back=i["back"], source=i.get("source"))
            for i in items
            if "front" in i and "back" in i
        ]
        return FlashcardsResponse(cards=cards, sources=_build_sources(docs))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Study/flashcards error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
