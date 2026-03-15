"""Reciprocal Rank Fusion utility for combining ranked result lists."""

from langchain_core.documents import Document


def rrf_fuse(
    dense_results: list[Document],
    sparse_results: list[Document],
    k: int = 60,
    top_n: int | None = None,
) -> list[Document]:
    """Fuse two ranked lists using Reciprocal Rank Fusion.

    Score = sum(1 / (k + rank)) for each list the document appears in.
    Documents are matched by a stable backend document ID (metadata["id"]),
    falling back to page_content only when no ID is present.
    Returns results sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked_list in [dense_results, sparse_results]:
        for rank, doc in enumerate(ranked_list, start=1):
            key = doc.metadata.get("id") or doc.page_content.strip()
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in doc_map:
                doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda k_: scores[k_], reverse=True)
    fused = [doc_map[key] for key in sorted_keys]

    if top_n is not None:
        fused = fused[:top_n]
    return fused
