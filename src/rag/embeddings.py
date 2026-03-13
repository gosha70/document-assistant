from abc import ABC, abstractmethod
from typing import Any


class EmbeddingAdapter(ABC):
    """Interface for embedding model adapters."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""

    @abstractmethod
    def get_langchain_embeddings(self) -> Any:
        """Return the underlying LangChain embeddings object for backends that require it."""


class InstructorEmbeddingAdapter(EmbeddingAdapter):
    """Adapter wrapping HuggingFaceInstructEmbeddings (current default)."""

    def __init__(
        self,
        model_name: str = "hkunlp/instructor-large",
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ):
        from langchain_community.embeddings import HuggingFaceInstructEmbeddings

        self._model_name = model_name
        self._embeddings = HuggingFaceInstructEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": normalize_embeddings},
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)

    def get_langchain_embeddings(self):
        return self._embeddings
