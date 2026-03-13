from typing import Optional
from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    file: str
    page: Optional[int] = None
    excerpt: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    collection_name: Optional[str] = None
    use_history: bool = False
    template_type: str = "generic"


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]


class IngestRequest(BaseModel):
    collection_name: Optional[str] = None


class IngestResponse(BaseModel):
    message: str
    document_count: int
    collection_name: str


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
    progress: Optional[float] = None
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str
