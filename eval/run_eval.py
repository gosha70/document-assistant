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
import yaml

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_EVAL_DIR = Path(__file__).resolve().parent
_DEFAULT_DATASET = _EVAL_DIR / "baseline_dataset.json"
_DEFAULT_RESULTS_DIR = _EVAL_DIR / "results"
_DEFAULT_THRESHOLDS = _EVAL_DIR / "thresholds.yaml"


def load_thresholds(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def check_thresholds(metrics: dict, thresholds: dict, k: int) -> bool:
    """Compare retrieval metrics against thresholds. Returns True if all pass."""
    retrieval_thresholds = thresholds.get("retrieval", {})
    all_passed = True
    for key, minimum in retrieval_thresholds.items():
        # normalise "recall@5" to match the actual k used
        if key.startswith("recall@"):
            actual_key = f"recall@{k}"
        elif key.startswith("precision@"):
            actual_key = f"precision@{k}"
        else:
            actual_key = key
        value = metrics.get(actual_key)
        if value is None:
            continue
        passed = value >= minimum
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {actual_key}: {value:.4f} (threshold: {minimum})")
        if not passed:
            all_passed = False
    return all_passed


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
    orchestrator=None,
) -> dict:
    """Run retrieval metrics across all samples.

    When an orchestrator is provided, retrieval goes through the orchestrator's
    pipeline (decomposition, HyDE, corrective) instead of the raw retriever.

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
    orchestrator_metadata = []

    for i, sample in enumerate(samples):
        question = sample["question"]
        is_negative = sample.get("type") == "negative"

        if orchestrator is not None:
            meta: dict = {}
            docs = orchestrator._retrieve(question, meta)
            orchestrator_metadata.append(meta)
        else:
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

    result: dict[str, Any] = {
        f"recall@{k}": sum(recalls) / len(recalls) if recalls else 0,
        f"precision@{k}": sum(precisions) / len(precisions) if precisions else 0,
        "mrr": sum(mrrs) / len(mrrs) if mrrs else 0,
        "num_positive_samples": len(recalls),
    }
    if negative_total > 0:
        result["negative_rejection_rate"] = negative_rejected / negative_total
        result["num_negative_samples"] = negative_total

    # Summarise orchestrator pipeline usage across samples
    if orchestrator_metadata:
        result["orchestrator_stats"] = {
            "decomposition_count": sum(1 for m in orchestrator_metadata if len(m.get("decomposed_queries", [])) > 1),
            "hyde_used_count": sum(1 for m in orchestrator_metadata if m.get("hyde_used")),
            "corrective_triggered_count": sum(1 for m in orchestrator_metadata if m.get("corrective_triggered")),
        }

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


def _build_orchestrator(retriever, settings, embedding):
    """Build a QueryOrchestrator wrapping the given retriever."""
    from src.rag.orchestrator import QueryOrchestrator
    from src.rag.generation import Generator

    llm = _create_llm(settings)
    generator = Generator(
        llm=llm,
        system_prompt=settings.system_prompt,
        prompt_name="qa",
    )
    return QueryOrchestrator(
        retriever=retriever,
        generator=generator,
        llm=llm,
        embedding=embedding,
        settings=settings,
    )


def _create_llm(settings):
    """Create LLM from settings (mirrors src.api.main._create_llm)."""
    backend = settings.llm.backend

    if backend == "openai":
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": settings.llm.model,
            "temperature": settings.llm.temperature,
            "max_tokens": settings.llm.max_tokens,
        }
        if settings.llm.base_url:
            kwargs["base_url"] = settings.llm.base_url
        if settings.llm.api_key:
            kwargs["api_key"] = settings.llm.api_key
        return ChatOpenAI(**kwargs)

    elif backend == "ollama":
        from langchain_openai import ChatOpenAI

        base_url = settings.llm.base_url or "http://localhost:11434/v1"
        model = settings.llm.model or "llama3.2"
        return ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=settings.llm.api_key or "ollama",
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
        )

    else:
        raise ValueError(f"Orchestrator eval requires 'openai' or 'ollama' LLM backend, got '{backend}'")


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

    # Orchestrator feature flags
    parser.add_argument(
        "--decomposition",
        action="store_true",
        help="Enable query decomposition",
    )
    parser.add_argument(
        "--hyde",
        action="store_true",
        help="Enable HyDE (Hypothetical Document Embeddings)",
    )
    parser.add_argument(
        "--corrective",
        action="store_true",
        help="Enable corrective re-retrieval on low confidence",
    )
    parser.add_argument(
        "--verification",
        action="store_true",
        help="Enable answer verification against sources",
    )
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Enable all orchestrator features",
    )
    parser.add_argument(
        "--check-thresholds",
        action="store_true",
        help="Compare metrics against thresholds in eval/thresholds.yaml; exit 1 if any fail",
    )
    parser.add_argument(
        "--thresholds-file",
        type=Path,
        default=_DEFAULT_THRESHOLDS,
        help="Path to thresholds YAML file",
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

    # Apply orchestrator feature flags to settings
    orchestrator = None
    any_orchestrator_flag = (
        args.decomposition or args.hyde or args.corrective or args.verification or args.full_pipeline
    )

    if any_orchestrator_flag:
        from src.config.settings import get_settings

        settings = get_settings()
        qp = settings.query_pipeline
        if args.full_pipeline or args.decomposition:
            qp.decomposition_enabled = True
        if args.full_pipeline or args.hyde:
            qp.hyde_enabled = True
        if args.full_pipeline or args.corrective:
            qp.corrective_retrieval_enabled = True
        if args.full_pipeline or args.verification:
            qp.verification_enabled = True

        embedding = _create_embedding(settings)
        orchestrator = _build_orchestrator(retriever, settings, embedding)

        enabled = [
            f
            for f, v in [
                ("decomposition", qp.decomposition_enabled),
                ("hyde", qp.hyde_enabled),
                ("corrective", qp.corrective_retrieval_enabled),
                ("verification", qp.verification_enabled),
            ]
            if v
        ]
        logger.info(f"Orchestrator features enabled: {', '.join(enabled)}")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "collection": col_name,
        "ephemeral": args.collection is None,
        "dataset_version": data.get("version"),
    }

    if any_orchestrator_flag:
        results["orchestrator_features"] = {
            "decomposition": qp.decomposition_enabled,
            "hyde": qp.hyde_enabled,
            "corrective": qp.corrective_retrieval_enabled,
            "verification": qp.verification_enabled,
        }

    logger.info(f"Running retrieval eval (k={args.k})...")
    results["retrieval"] = run_retrieval_eval(
        samples,
        retriever,
        k=args.k,
        orchestrator=orchestrator if any_orchestrator_flag else None,
    )
    logger.info(
        f"Retrieval: recall@{args.k}={results['retrieval'][f'recall@{args.k}']:.3f} "
        f"precision@{args.k}={results['retrieval'][f'precision@{args.k}']:.3f} "
        f"mrr={results['retrieval']['mrr']:.3f}"
    )

    if args.include_generation:
        results["generation"] = run_generation_eval(samples, retriever, generator=None)

    path = save_results(results, args.results_dir)
    print(f"\nResults: {path}")

    if args.check_thresholds:
        thresholds = load_thresholds(args.thresholds_file)
        print("\nThreshold check:")
        passed = check_thresholds(results["retrieval"], thresholds, args.k)
        if not passed:
            logger.error("One or more metrics failed threshold checks")
            sys.exit(1)
        print("All thresholds passed.")


if __name__ == "__main__":
    main()
