import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


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


class InstructorLangChainEmbeddings(Embeddings):
    """LangChain `Embeddings` view over an Instructor SentenceTransformer.

    Needed because backends hand this object to LangChain directly
    (e.g. `Chroma(embedding_function=...)`).
    """

    def __init__(
        self,
        model: Any,
        embed_instruction: str,
        query_instruction: str,
        normalize_embeddings: bool,
    ):
        self._model = model
        self._embed_instruction = embed_instruction
        self._query_instruction = query_instruction
        self._normalize_embeddings = normalize_embeddings

    def _encode(self, texts: list[str], instruction: str) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            prompt=instruction,
            normalize_embeddings=self._normalize_embeddings,
        )
        return [vector.tolist() for vector in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(list(texts), self._embed_instruction)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], self._query_instruction)[0]


class InstructorEmbeddingAdapter(EmbeddingAdapter):
    """Adapter for instruction-tuned Instructor models on sentence-transformers.

    Instructor models (e.g. hkunlp/instructor-large) require instruction prefixes
    on each embedding call, and the instruction tokens must be excluded from pooling.
    sentence-transformers supports this natively: the model's own `Pooling` module
    carries `include_prompt=False`, and the instruction is passed per call as `prompt`.

    Vectors are numerically equivalent to the previous
    `langchain_community.HuggingFaceInstructEmbeddings` implementation
    (see specs/instructor-embedding-fix/plan.md for the equivalence gate).
    """

    def __init__(
        self,
        model_name: str,
        embed_instruction: str,
        query_instruction: str,
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ):
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self._embeddings = InstructorLangChainEmbeddings(
            model=self._model,
            embed_instruction=embed_instruction,
            query_instruction=query_instruction,
            normalize_embeddings=normalize_embeddings,
        )
        logger.info(f"Loaded Instructor embedding model: {model_name} on {device}")

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)

    def get_langchain_embeddings(self):
        return self._embeddings


class HuggingFaceEmbeddingAdapter(EmbeddingAdapter):
    """Adapter wrapping HuggingFaceEmbeddings from langchain-huggingface.

    For non-instruction models (e.g. BAAI/bge-small-en, sentence-transformers/*).
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ):
        from langchain_huggingface import HuggingFaceEmbeddings

        self._model_name = model_name
        self._embeddings = HuggingFaceEmbeddings(
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
