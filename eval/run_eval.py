"""Eval CLI — retrieval metrics on baseline or live collections.

Builds an ephemeral in-memory Chroma collection from the dataset contexts
(default) or evaluates against a pre-ingested collection (--collection).

Usage:
    poetry run python -m eval.run_eval
    poetry run python -m eval.run_eval --collection my_collection
    poetry run python -m eval.run_eval --include-generation
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_EVAL_DIR = Path(__file__).resolve().parent
_DEFAULT_DATASET = _EVAL_DIR / "baseline_dataset.json"
_DEFAULT_RESULTS_DIR = _EVAL_DIR / "results"


def load_dataset(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    samples = data.get("samples", [])
    if not samples:
        logger.error("Dataset has no samples")
        sys.exit(1)
    logger.info(f"Loaded {len(samples)} samples from {path.name}")
    return data


def _recall_at_k(retrieved_texts: list[str], relevant_texts: list[str], k: int) -> float:
    """Fraction of relevant texts found in top-k retrieved results."""
    if not relevant_texts:
        return 0.0
    top_k = retrieved_texts[:k]
    hits = sum(1 for r in relevant_texts if any(r in t or t in r for t in top_k))
    return hits / len(relevant_texts)


def _precision_at_k(retrieved_texts: list[str], relevant_texts: list[str], k: int) -> float:
    """Fraction of top-k retrieved results that are relevant."""
    top_k = retrieved_texts[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for t in top_k if any(r in t or t in r for r in relevant_texts))
    return hits / len(top_k)


def _mrr(retrieved_texts: list[str], relevant_texts: list[str]) -> float:
    """Mean Reciprocal Rank — 1/(rank of first relevant result)."""
    for i, t in enumerate(retrieved_texts):
        if any(r in t or t in r for r in relevant_texts):
            return 1.0 / (i + 1)
    return 0.0


def _create_embedding(settings):
    """Create embedding adapter from settings (same logic as src.api.main)."""
    embedding_type = settings.embedding.type
    if embedding_type == "instructor":
        from src.rag.embeddings import InstructorEmbeddingAdapter

        return InstructorEmbeddingAdapter(
            model_name=settings.embedding.model_name,
            device=settings.embedding.device,
            normalize_embeddings=settings.embedding.normalize_embeddings,
        )
    elif embedding_type == "huggingface":
        from src.rag.embeddings import HuggingFaceEmbeddingAdapter

        return HuggingFaceEmbeddingAdapter(
            model_name=settings.embedding.model_name,
            device=settings.embedding.device,
            normalize_embeddings=settings.embedding.normalize_embeddings,
        )
    else:
        raise ValueError(f"Unknown embedding type: '{embedding_type}'")


def _create_backend(settings, embedding):
    """Create vectorstore backend from settings (same logic as src.api.main)."""
    backend_type = settings.vectorstore.backend
    hybrid_cfg = {
        "enabled": settings.vectorstore.hybrid.enabled,
        "sparse_encoder": settings.vectorstore.hybrid.sparse_encoder,
        "rrf_k": settings.vectorstore.hybrid.rrf_k,
    }

    if backend_type == "chroma":
        from src.rag.chroma_backend import ChromaBackend

        return ChromaBackend(
            embedding=embedding,
            persist_directory=settings.vectorstore.persist_directory,
            embedding_type=settings.embedding.type,
            allow_legacy_collections=settings.vectorstore.allow_legacy_collections,
            hybrid_settings=hybrid_cfg,
        )
    elif backend_type == "qdrant":
        from src.rag.qdrant_backend import QdrantBackend

        return QdrantBackend(
            embedding=embedding,
            url=settings.vectorstore.qdrant_url,
            api_key=settings.vectorstore.qdrant_api_key,
            prefer_grpc=settings.vectorstore.qdrant_prefer_grpc,
            embedding_type=settings.embedding.type,
            allow_legacy_collections=settings.vectorstore.allow_legacy_collections,
            hybrid_settings=hybrid_cfg,
        )
    else:
        raise ValueError(f"Unknown vectorstore backend: '{backend_type}'")


def _create_reranker(settings):
    """Create the reranker based on config (same logic as src.api.main)."""
    if not settings.reranker.enabled:
        return None
    from src.rag.reranking import CrossEncoderReranker

    return CrossEncoderReranker(model_name=settings.reranker.model_name)


def _migrate_legacy_collections(backend, settings):
    """Stamp provenance on legacy collections (same logic as lifespan in main.py)."""
    try:
        for col in backend.list_collections():
            name = col["name"]
            if backend.get_embedding_provenance(name) is None:
                backend._set_provenance(name)
                logger.info(f"Migrated legacy collection '{name}' — stamped provenance")
    except Exception as e:
        logger.warning(f"Legacy collection migration check failed: {e}")


def build_ephemeral_collection(samples: list[dict]) -> tuple:
    """Build an in-memory Chroma collection from dataset contexts.

    Returns (backend, collection_name, retriever).
    """
    from src.config.settings import get_settings
    from src.rag.chroma_backend import ChromaBackend
    from src.rag.reranking import NoOpReranker
    from src.rag.retrieval import Retriever

    settings = get_settings()
    embedding = _create_embedding(settings)
    reranker = _create_reranker(settings) or NoOpReranker()
    collection_name = f"eval_ephemeral_{int(time.time())}"

    backend = ChromaBackend(
        embedding=embedding,
        persist_directory=None,
    )

    # Collect unique context chunks across all samples
    chunks: list[Document] = []
    seen: set[str] = set()
    for sample in samples:
        for ctx in sample.get("contexts", []):
            if ctx not in seen:
                seen.add(ctx)
                chunks.append(Document(page_content=ctx, metadata={"source": "eval_dataset"}))

    logger.info(f"Ingesting {len(chunks)} unique context chunks into ephemeral collection '{collection_name}'")
    backend.store(chunks, collection_name)

    retriever = Retriever(
        backend=backend,
        collection_name=collection_name,
        reranker=reranker,
        initial_k=20,
        final_k=settings.reranker.top_k,
        use_hybrid=False,  # ephemeral Chroma — dense only
    )
    return backend, collection_name, retriever


def build_live_retriever(collection_name: str) -> tuple:
    """Build a retriever against an existing collection.

    Creates the backend directly from settings, then runs the same legacy
    collection migration that the app lifespan performs at startup.
    """
    from src.config.settings import get_settings
    from src.rag.reranking import NoOpReranker
    from src.rag.retrieval import Retriever

    settings = get_settings()
    embedding = _create_embedding(settings)
    backend = _create_backend(settings, embedding)
    reranker = _create_reranker(settings) or NoOpReranker()

    # Mirror the app lifespan: auto-migrate legacy collections
    _migrate_legacy_collections(backend, settings)

    retriever = Retriever(
        backend=backend,
        collection_name=collection_name,
        reranker=reranker,
        initial_k=20,
        final_k=settings.reranker.top_k,
        use_hybrid=settings.vectorstore.hybrid.enabled,
    )
    return backend, collection_name, retriever


def run_retrieval_eval(
    samples: list[dict],
    retriever,
    k: int = 5,
) -> dict:
    """Run retrieval metrics across all samples.

    Negative samples (type="negative") are excluded from recall/precision/MRR
    because their contexts are intentional distractors with no correct answer.
    They are scored separately: a negative sample is "rejected" correctly when
    none of its distractor contexts appear in the top-k results.
    """
    recalls = []
    precisions = []
    mrrs = []
    negative_total = 0
    negative_rejected = 0

    for i, sample in enumerate(samples):
        question = sample["question"]
        is_negative = sample.get("type") == "negative"

        docs = retriever.retrieve(question)
        retrieved_texts = [doc.page_content for doc in docs]

        if is_negative:
            # For negatives, success = NOT retrieving the distractor contexts
            negative_total += 1
            distractors = sample.get("contexts", [])
            hit = any(any(d in t or t in d for d in distractors) for t in retrieved_texts[:k])
            if not hit:
                negative_rejected += 1
            logger.debug(f"  [{i+1}/{len(samples)}] negative — rejected={not hit}")
        else:
            relevant = sample.get("contexts", [])
            r = _recall_at_k(retrieved_texts, relevant, k)
            p = _precision_at_k(retrieved_texts, relevant, k)
            m = _mrr(retrieved_texts, relevant)

            recalls.append(r)
            precisions.append(p)
            mrrs.append(m)

            logger.debug(f"  [{i+1}/{len(samples)}] recall@{k}={r:.2f} precision@{k}={p:.2f} mrr={m:.2f}")

    result = {
        f"recall@{k}": sum(recalls) / len(recalls) if recalls else 0,
        f"precision@{k}": sum(precisions) / len(precisions) if precisions else 0,
        "mrr": sum(mrrs) / len(mrrs) if mrrs else 0,
        "num_positive_samples": len(recalls),
    }
    if negative_total > 0:
        result["negative_rejection_rate"] = negative_rejected / negative_total
        result["num_negative_samples"] = negative_total
    return result


def run_generation_eval(
    samples: list[dict],
    retriever,
    generator,
) -> dict:
    """Run generation (answer quality) eval. Placeholder — returns structure only."""
    logger.warning("Generation eval not yet implemented; returning placeholders")
    return {
        "faithfulness": None,
        "answer_relevancy": None,
        "num_samples": len(samples),
    }


def save_results(results: dict, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = results_dir / f"{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Document Assistant eval harness")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_DEFAULT_DATASET,
        help="Path to evaluation dataset JSON",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Evaluate against a pre-ingested collection (skip ephemeral build)",
    )
    parser.add_argument(
        "--include-generation",
        action="store_true",
        help="Include generation metrics (not yet implemented; omitted by default)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_DEFAULT_RESULTS_DIR,
        help="Directory for result JSON files",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="k value for recall@k and precision@k",
    )
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    samples = data["samples"]

    if args.collection:
        logger.info(f"Evaluating against existing collection: {args.collection}")
        backend, col_name, retriever = build_live_retriever(args.collection)
    else:
        logger.info("Building ephemeral collection from dataset contexts")
        backend, col_name, retriever = build_ephemeral_collection(samples)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "collection": col_name,
        "ephemeral": args.collection is None,
        "dataset_version": data.get("version"),
    }

    logger.info(f"Running retrieval eval (k={args.k})...")
    results["retrieval"] = run_retrieval_eval(samples, retriever, k=args.k)
    logger.info(
        f"Retrieval: recall@{args.k}={results['retrieval'][f'recall@{args.k}']:.3f} "
        f"precision@{args.k}={results['retrieval'][f'precision@{args.k}']:.3f} "
        f"mrr={results['retrieval']['mrr']:.3f}"
    )

    if args.include_generation:
        results["generation"] = run_generation_eval(samples, retriever, generator=None)

    path = save_results(results, args.results_dir)
    print(f"\nResults: {path}")


if __name__ == "__main__":
    main()
