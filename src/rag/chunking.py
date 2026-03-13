import logging
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter, TextSplitter

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

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


def get_text_splitter(file_extension: str | None = None) -> TextSplitter:
    """Create a text splitter, using language-aware separators when applicable.

    Chunk size and overlap are read from config/defaults.yaml.
    """
    settings = get_settings()

    separators = None
    if file_extension:
        ext = file_extension.lstrip(".")
        language = LANGUAGE_MAP.get(ext)
        if language is not None:
            separators = RecursiveCharacterTextSplitter.get_separators_for_language(language)

    return RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
        keep_separator=True,
    )
