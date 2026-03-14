import logging
import threading
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter, TextSplitter
from langchain_core.documents import Document

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

CHUNKING_VERSION = "2"

# Maps file extensions to LangChain Language enum for language-aware separators
LANGUAGE_MAP: dict[str, Language] = {
    "java": Language.JAVA,
    "js": Language.JS,
    "html": Language.HTML,
    "md": Language.MARKDOWN,
    "py": Language.PYTHON,
    "sql": Language.SOL,
    "ddl": Language.SOL,
}

# Cached tokenizer instances keyed by tokenizer spec (e.g. "tiktoken:cl100k_base")
_tokenizer_cache: dict[str, object] = {}
_tokenizer_cache_lock = threading.Lock()


def _parse_tokenizer(tokenizer_spec: str) -> tuple[str, str]:
    """Parse a tokenizer spec like 'tiktoken:cl100k_base' into (backend, model)."""
    if ":" not in tokenizer_spec:
        raise ValueError(
            f"Invalid tokenizer spec '{tokenizer_spec}': expected 'backend:model' "
            f"(e.g. 'tiktoken:cl100k_base' or 'huggingface:bert-base-uncased')"
        )
    backend, model = tokenizer_spec.split(":", 1)
    return backend, model


def _get_tokenizer(tokenizer_spec: str):
    """Return a cached tokenizer object for the given spec."""
    with _tokenizer_cache_lock:
        if tokenizer_spec in _tokenizer_cache:
            return _tokenizer_cache[tokenizer_spec]

    backend, model = _parse_tokenizer(tokenizer_spec)
    if backend == "tiktoken":
        import tiktoken
        tokenizer = tiktoken.get_encoding(model)
    elif backend == "huggingface":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model)
    else:
        raise ValueError(f"Unknown tokenizer backend '{backend}'. Supported: tiktoken, huggingface")

    with _tokenizer_cache_lock:
        _tokenizer_cache[tokenizer_spec] = tokenizer
    return tokenizer


def count_tokens(text: str, tokenizer_spec: str) -> int:
    """Count tokens in text using the configured tokenizer (cached)."""
    tokenizer = _get_tokenizer(tokenizer_spec)
    backend, _ = _parse_tokenizer(tokenizer_spec)
    if backend == "tiktoken":
        return len(tokenizer.encode(text))
    elif backend == "huggingface":
        return len(tokenizer.encode(text))
    return len(text.split())


def enrich_chunk_metadata(documents: list[Document], tokenizer_spec: str) -> list[Document]:
    """Attach token_count, tokenizer, and chunking_version metadata to each chunk."""
    for doc in documents:
        doc.metadata["token_count"] = count_tokens(doc.page_content, tokenizer_spec)
        doc.metadata["tokenizer"] = tokenizer_spec
        doc.metadata["chunking_version"] = CHUNKING_VERSION
    return documents


def get_text_splitter(file_extension: str | None = None) -> TextSplitter:
    """Create a tokenizer-aware text splitter.

    Chunk size and overlap are measured in tokens (not characters).
    The tokenizer backend is read from config/defaults.yaml (chunking.tokenizer).
    Language-aware separators are used when the file extension is recognized.
    """
    settings = get_settings()
    backend, model = _parse_tokenizer(settings.chunking.tokenizer)

    separators = None
    if file_extension:
        ext = file_extension.lstrip(".")
        language = LANGUAGE_MAP.get(ext)
        if language is not None:
            separators = RecursiveCharacterTextSplitter.get_separators_for_language(language)

    if backend == "tiktoken":
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=model,
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap,
            separators=separators,
            keep_separator=True,
        )
    elif backend == "huggingface":
        hf_tokenizer = _get_tokenizer(settings.chunking.tokenizer)
        return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            hf_tokenizer,
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap,
            separators=separators,
            keep_separator=True,
        )
    else:
        raise ValueError(
            f"Unknown tokenizer backend '{backend}'. Supported: tiktoken, huggingface"
        )
