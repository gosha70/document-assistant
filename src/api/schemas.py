from typing import Optional
from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    file: str
    page: Optional[int] = None
    excerpt: str
    chunk_id: Optional[str] = None
    search_type: Optional[str] = None
    full_excerpt: Optional[str] = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    collection_name: Optional[str] = None
    use_history: bool = False
    template_type: str = "generic"


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    metadata: Optional[dict] = None


class IngestRequest(BaseModel):
    collection_name: Optional[str] = None


class IngestResponse(BaseModel):
    message: str
    document_count: int
    collection_name: str
    replaced_count: int = 0


class StatusResponse(BaseModel):
    app_name: str
    document_count: int
    collection_name: str
    embedding_model: str
    vectorstore_backend: str


class CollectionInfo(BaseModel):
    name: str
    backend: str
    document_count: int
    persist_directory: Optional[str] = None
    embedding_model: Optional[str] = None


class JobInfo(BaseModel):
    id: str
    type: str
    status: str
    collection_name: Optional[str] = None
    progress: Optional[float] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class DuplicateFile(BaseModel):
    filename: str
    existing_chunk_count: int


class DuplicateDetectedResponse(BaseModel):
    message: str
    duplicates: list[DuplicateFile]
    collection_name: str


class ChunkSample(BaseModel):
    id: str
    text: str
    metadata: dict


class ChunkSampleResponse(BaseModel):
    collection_name: str
    total_count: int
    chunks: list[ChunkSample]


class SourceInfo(BaseModel):
    filename: str
    chunk_count: int


class SourceListResponse(BaseModel):
    collection_name: str
    sources: list[SourceInfo]
    truncated: bool = False
    scanned_chunks: int = 0


class AlertInfoSchema(BaseModel):
    level: str
    metric: str
    current_value: float
    threshold: float
    message: str


class AlertsResponse(BaseModel):
    alerts: list[AlertInfoSchema]
    checked_at: str


class ReportSnapshot(BaseModel):
    timestamp: str
    uptime_seconds: float
    request_count: int
    error_count: int
    llm_call_count: int
    ingest_total_docs: int
    ingest_error_count: int
    retrieval_no_source_count: int


class ErrorResponse(BaseModel):
    detail: str


class StudyRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection_name: Optional[str] = None
    k: int = Field(default=10, ge=1, le=50)


class FlashcardsRequest(StudyRequest):
    max_cards: int = Field(default=10, ge=1, le=50)


class SummaryResponse(BaseModel):
    summary: str
    sources: list[SourceCitation]


class GlossaryTerm(BaseModel):
    term: str
    definition: str
    source: Optional[str] = None


class GlossaryResponse(BaseModel):
    terms: list[GlossaryTerm]
    sources: list[SourceCitation]


class Flashcard(BaseModel):
    front: str
    back: str
    source: Optional[str] = None


class FlashcardsResponse(BaseModel):
    cards: list[Flashcard]
    sources: list[SourceCitation]
