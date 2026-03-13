import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
_DEFAULT_CONFIG = _CONFIG_DIR / "defaults.yaml"


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, preferring override values."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class EmbeddingSettings(BaseModel):
    model_name: str
    device: str
    normalize_embeddings: bool


class ChunkingSettings(BaseModel):
    chunk_size: int
    chunk_overlap: int
    batch_size: int


class ModelSettings(BaseModel):
    default_model_name: str
    default_model_id: str
    default_model_basename: str
    n_ctx: int
    n_batch: int
    n_gpu_layers: int
    cache_dir: str


class VectorStoreSettings(BaseModel):
    backend: str
    collection_name: str
    persist_directory: Optional[str]
    chroma_anonymized_telemetry: bool


class UploadSettings(BaseModel):
    max_file_size_mb: int
    allowed_extensions: list[str]


class AuthSettings(BaseModel):
    enabled: bool
    admin_token: Optional[str]


class AppSettings(BaseModel):
    name: str
    host: str
    port: int
    debug: bool


class Settings(BaseModel):
    app: AppSettings
    embedding: EmbeddingSettings
    chunking: ChunkingSettings
    model: ModelSettings
    vectorstore: VectorStoreSettings
    upload: UploadSettings
    auth: AuthSettings
    system_prompt: str


def get_settings(override_path: Optional[str] = None) -> Settings:
    """Load settings from config/defaults.yaml, optionally merged with an override file.

    Override file path can also be set via the DOT_CONFIG_PATH environment variable.
    """
    data = _load_yaml(_DEFAULT_CONFIG)

    override_path = override_path or os.environ.get("DOT_CONFIG_PATH")
    if override_path:
        override_data = _load_yaml(Path(override_path))
        data = _deep_merge(data, override_data)

    return Settings(**data)
