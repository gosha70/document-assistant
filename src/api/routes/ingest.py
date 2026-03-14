import os
import logging
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

from langchain_core.documents import Document

from src.api.schemas import IngestResponse
from src.api.middleware.upload import validate_upload
from src.api.deps import get_vectorstore_backend
from src.config.settings import get_settings
from src.rag.chunking import get_text_splitter, enrich_chunk_metadata
from src.utils.metrics import get_metrics_collector

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingest"])


def _load_and_split(file_path: str) -> list[Document]:
    """Load a file and split it using the config-driven chunking layer."""
    ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
    text_splitter = get_text_splitter(ext)

    from embeddings.unstructured.file_type import FileType
    from embeddings.unstructured.document_splitter import DocumentSplitter

    file_type = FileType.get_file_type_by_extension(file_path)
    if file_type is None:
        return []

    splitter = DocumentSplitter(logging)
    converter = splitter.get_converter(file_type)
    if converter is None:
        return []

    documents = converter.load_and_split_file(text_splitter=text_splitter, file_path=file_path)
    settings = get_settings()
    return enrich_chunk_metadata(documents, settings.chunking.tokenizer)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_files(
    files: list[UploadFile] = File(...),
    collection_name: Optional[str] = Form(None),
):
    """Upload and ingest files into the vector store."""
    settings = get_settings()
    collection = collection_name or settings.vectorstore.collection_name
    backend = get_vectorstore_backend()

    total_docs = 0

    for file in files:
        validate_upload(file)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f"_{os.path.basename(file.filename)}",
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            documents = _load_and_split(tmp_path)

            if documents:
                backend.store(documents=documents, collection_name=collection)
                total_docs += len(documents)
                logger.info(f"Ingested {len(documents)} chunks from '{file.filename}'")
        finally:
            os.unlink(tmp_path)

    if total_docs > 0 and settings.telemetry.enabled:
        get_metrics_collector().record_ingest(total_docs)

    return IngestResponse(
        message=f"Ingested {total_docs} document chunks from {len(files)} file(s)",
        document_count=total_docs,
        collection_name=collection,
    )
