import os
import logging
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

from langchain_core.documents import Document

from src.api.schemas import IngestResponse, DuplicateDetectedResponse, DuplicateFile
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

    settings = get_settings()
    late_cfg = settings.chunking.late_chunking

    # Late chunking: embed full document context, pool per-chunk (mutually exclusive with contextual)
    if late_cfg.enabled and ext in late_cfg.file_types:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from src.rag.late_chunking import LateChunkingEmbedder

        # Pass 1: Extract full text
        # Use \f (form-feed) as page separator so _split_into_segments("page")
        # can recover page boundaries when the document exceeds context window.
        no_split = RecursiveCharacterTextSplitter(chunk_size=1_000_000, chunk_overlap=0)
        full_docs = converter.load_and_split_file(text_splitter=no_split, file_path=file_path)
        full_text = "\f".join(doc.page_content for doc in full_docs)

        # Pass 2: Split into chunks
        chunks = converter.load_and_split_file(text_splitter=text_splitter, file_path=file_path)
        chunks = enrich_chunk_metadata(chunks, settings.chunking.tokenizer)

        # Pass 3: Late chunking embedding
        try:
            embedder = LateChunkingEmbedder(
                model_name=late_cfg.model_name,
                max_context_tokens=late_cfg.max_context_tokens,
                pooling_strategy=late_cfg.pooling_strategy,
                device=settings.embedding.device,
            )
            chunks = embedder.embed_document_chunks(
                full_text,
                chunks,
                batch_by=late_cfg.batch_by,
            )
        except Exception as e:
            logger.warning(f"Late chunking failed, chunks will use standard embedding: {e}")

        return chunks

    if settings.chunking.contextual.enabled:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from src.rag.contextual import ChunkContextAugmenter
        from src.api.deps import _llm

        # Pass 1: Extract full text using converter with a no-split splitter
        no_split = RecursiveCharacterTextSplitter(chunk_size=1_000_000, chunk_overlap=0)
        full_docs = converter.load_and_split_file(text_splitter=no_split, file_path=file_path)
        full_text = "\n\n".join(doc.page_content for doc in full_docs)

        # Pass 2: Split into real chunks using the same converter
        chunks = converter.load_and_split_file(text_splitter=text_splitter, file_path=file_path)
        chunks = enrich_chunk_metadata(chunks, settings.chunking.tokenizer)

        if _llm is not None:
            augmenter = ChunkContextAugmenter(llm=_llm, settings=settings)
            chunks = augmenter.augment(full_text, chunks)
        else:
            logger.warning("Contextual augmentation enabled but no LLM available; skipping")

        return chunks

    documents = converter.load_and_split_file(text_splitter=text_splitter, file_path=file_path)
    return enrich_chunk_metadata(documents, settings.chunking.tokenizer)


@router.post("/ingest")
async def ingest_files(
    files: list[UploadFile] = File(...),
    collection_name: Optional[str] = Form(None),
    duplicate_strategy: Optional[str] = Form("check"),
):
    """Upload and ingest files into the vector store.

    duplicate_strategy controls behaviour when a file with the same name already exists:
      - "check"   (default) — detect duplicates and return 409 with details
      - "replace" — delete old chunks for matching filenames, then ingest
      - "add"     — ingest alongside existing chunks (no dedup)
      - "skip"    — silently skip files that already exist
    """
    settings = get_settings()
    collection = collection_name or settings.vectorstore.collection_name
    backend = get_vectorstore_backend()
    strategy = (duplicate_strategy or "check").lower()

    _VALID_STRATEGIES = {"check", "replace", "add", "skip"}
    if strategy not in _VALID_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid duplicate_strategy '{duplicate_strategy}'. Must be one of: {', '.join(sorted(_VALID_STRATEGIES))}",
        )

    # Validate all uploads before any backend interaction
    for file in files:
        validate_upload(file)

    # --- Duplicate check phase ---
    if strategy == "check":
        duplicates = []
        for file in files:
            original_name = os.path.basename(file.filename) if file.filename else "unknown"
            existing_ids = backend.find_by_source(original_name, collection)
            if existing_ids:
                duplicates.append(DuplicateFile(filename=original_name, existing_chunk_count=len(existing_ids)))
        if duplicates:
            return DuplicateDetectedResponse(
                message=f"{len(duplicates)} file(s) already exist in the collection",
                duplicates=duplicates,
                collection_name=collection,
            )
        # No duplicates — fall through to normal ingest
        strategy = "add"

    total_docs = 0
    replaced_chunks = 0
    skipped = 0

    for file in files:
        original_name = os.path.basename(file.filename) if file.filename else "unknown"

        # Skip duplicates early
        if strategy == "skip":
            existing_ids = backend.find_by_source(original_name, collection)
            if existing_ids:
                logger.info(f"Skipped duplicate '{original_name}' ({len(existing_ids)} existing chunks)")
                skipped += 1
                continue

        # For replace, find old IDs before ingesting (delete after new chunks are stored)
        old_ids_to_delete: list[str] = []
        if strategy == "replace":
            old_ids_to_delete = backend.find_by_source(original_name, collection)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f"_{original_name}",
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            documents = _load_and_split(tmp_path)

            for doc in documents:
                doc.metadata["source"] = original_name

            if documents:
                new_ids = backend.store(documents=documents, collection_name=collection)
                total_docs += len(documents)
                logger.info(f"Ingested {len(documents)} chunks from '{file.filename}'")

                # Delete old chunks only after new ones are safely stored
                if old_ids_to_delete:
                    try:
                        backend.delete(ids=old_ids_to_delete, collection_name=collection)
                        replaced_chunks += len(old_ids_to_delete)
                        logger.info(f"Replaced: deleted {len(old_ids_to_delete)} old chunks for '{original_name}'")
                    except Exception as e:
                        # Roll back: remove newly inserted chunks so we don't leave duplicates
                        logger.error(
                            f"Delete of old chunks failed for '{original_name}': {e}. Rolling back new chunks."
                        )
                        try:
                            backend.delete(ids=new_ids, collection_name=collection)
                        except Exception as rb_err:
                            logger.error(f"Rollback also failed for '{original_name}': {rb_err}")
                        raise
        except Exception as e:
            logger.error(f"Failed to ingest '{original_name}': {e}")
            if settings.telemetry.enabled:
                get_metrics_collector().record_ingest_error(settings.alerting.window_seconds)
            raise
        finally:
            os.unlink(tmp_path)

    if total_docs > 0 and settings.telemetry.enabled:
        get_metrics_collector().record_ingest(total_docs)

    msg = f"Ingested {total_docs} document chunks from {len(files)} file(s)"
    if skipped:
        msg += f" ({skipped} skipped as duplicates)"

    return IngestResponse(
        message=msg,
        document_count=total_docs,
        collection_name=collection,
        replaced_count=replaced_chunks,
    )
