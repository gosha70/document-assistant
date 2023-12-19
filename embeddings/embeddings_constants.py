from chromadb.config import Settings

# Char-level splits
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Chroma settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)