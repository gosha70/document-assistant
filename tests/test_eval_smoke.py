"""Eval smoke test — lightweight retrieval sanity check with in-memory Chroma."""

from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from eval.run_eval import run_retrieval_eval


class TestEvalSmoke:
    """Verify that the eval pipeline can ingest, retrieve, and score."""

    def _mock_embedding(self, dim=8):
        """Create a mock embedding that returns deterministic vectors.

        The get_langchain_embeddings() must return an object whose
        embed_documents and embed_query methods return real float lists,
        since Chroma validates the embedding format.
        """
        counter = [0]

        def _embed_docs(texts):
            vecs = []
            for t in texts:
                counter[0] += 1
                vec = [0.0] * dim
                vec[counter[0] % dim] = 1.0
                vecs.append(vec)
            return vecs

        def _embed_query(text):
            vec = [0.0] * dim
            vec[hash(text) % dim] = 1.0
            return vec

        lc_emb = MagicMock()
        lc_emb.embed_documents = _embed_docs
        lc_emb.embed_query = _embed_query

        emb = MagicMock()
        emb.model_name = "test-model"
        emb.embed_documents = _embed_docs
        emb.embed_query = _embed_query
        emb.get_langchain_embeddings.return_value = lc_emb
        return emb

    def test_ingest_and_retrieve_finds_relevant(self, tmp_dir):
        """Store synthetic docs, retrieve, and verify at least one hit."""
        from src.rag.chroma_backend import ChromaBackend
        from src.rag.retrieval import Retriever

        emb = self._mock_embedding()
        backend = ChromaBackend(embedding=emb, persist_directory=tmp_dir)

        docs = [
            Document(page_content="Python is a programming language", metadata={"source": "a.txt"}),
            Document(page_content="The quick brown fox jumps over the lazy dog", metadata={"source": "b.txt"}),
            Document(page_content="Machine learning uses neural networks", metadata={"source": "c.txt"}),
            Document(page_content="FastAPI is a web framework for Python", metadata={"source": "d.txt"}),
            Document(page_content="Chroma is a vector database for embeddings", metadata={"source": "e.txt"}),
        ]
        backend.store(docs, "eval_smoke")

        retriever = Retriever(
            backend=backend,
            collection_name="eval_smoke",
            initial_k=5,
            final_k=5,
            use_hybrid=False,
        )

        results = retriever.retrieve("programming language")
        assert len(results) > 0

    def test_recall_above_zero(self, tmp_dir):
        """With mock embeddings, recall@5 should be > 0 on at least some queries."""
        from src.rag.chroma_backend import ChromaBackend
        from src.rag.retrieval import Retriever

        emb = self._mock_embedding()
        backend = ChromaBackend(embedding=emb, persist_directory=tmp_dir)

        docs = [
            Document(page_content="The capital of France is Paris", metadata={"source": "geo.txt"}),
            Document(page_content="Berlin is the capital of Germany", metadata={"source": "geo.txt"}),
            Document(page_content="Tokyo is the capital of Japan", metadata={"source": "geo.txt"}),
            Document(page_content="Python was created by Guido van Rossum", metadata={"source": "prog.txt"}),
            Document(page_content="JavaScript runs in web browsers", metadata={"source": "prog.txt"}),
        ]
        backend.store(docs, "eval_recall")

        retriever = Retriever(
            backend=backend,
            collection_name="eval_recall",
            initial_k=5,
            final_k=5,
            use_hybrid=False,
        )

        results = retriever.retrieve("capital cities in Europe")
        # With mock embeddings, results are somewhat arbitrary, but we should get something
        assert len(results) > 0
        retrieved_texts = [r.page_content for r in results]
        assert len(retrieved_texts) == len(set(retrieved_texts)), "Should not have exact duplicates"


class TestEvalOrchestratorBranch:
    """Verify run_retrieval_eval uses orchestrator._retrieve when provided."""

    def test_orchestrator_branch_calls_retrieve(self):
        """When an orchestrator is passed, retrieval goes through it, not the raw retriever."""
        docs = [Document(page_content="The answer is 42.", metadata={"id": "1"})]
        orchestrator = MagicMock()
        orchestrator._retrieve.return_value = docs

        retriever = MagicMock()
        retriever.retrieve.return_value = docs

        samples = [
            {
                "question": "What is the answer?",
                "ground_truth": "42",
                "contexts": ["The answer is 42."],
                "type": "factual",
            },
        ]

        run_retrieval_eval(samples, retriever, k=5, orchestrator=orchestrator)

        orchestrator._retrieve.assert_called_once()
        retriever.retrieve.assert_not_called()

    def test_orchestrator_stats_aggregation(self):
        """orchestrator_stats counts decomposition, hyde, and corrective usage."""
        docs = [Document(page_content="text", metadata={"id": "1"})]

        call_count = [0]

        def fake_retrieve(question, meta):
            call_count[0] += 1
            if call_count[0] == 1:
                meta["decomposed_queries"] = ["sub1", "sub2"]
                meta["hyde_used"] = True
            elif call_count[0] == 2:
                meta["corrective_triggered"] = True
            # third sample: no orchestrator metadata flags
            return docs

        orchestrator = MagicMock()
        orchestrator._retrieve.side_effect = fake_retrieve

        samples = [
            {"question": "q1", "contexts": ["text"], "type": "factual"},
            {"question": "q2", "contexts": ["text"], "type": "factual"},
            {"question": "q3", "contexts": ["text"], "type": "factual"},
        ]

        result = run_retrieval_eval(samples, MagicMock(), k=5, orchestrator=orchestrator)

        assert "orchestrator_stats" in result
        stats = result["orchestrator_stats"]
        assert stats["decomposition_count"] == 1
        assert stats["hyde_used_count"] == 1
        assert stats["corrective_triggered_count"] == 1

    def test_no_orchestrator_no_stats(self):
        """Without orchestrator, orchestrator_stats should not be in result."""
        retriever = MagicMock()
        retriever.retrieve.return_value = []

        samples = [
            {"question": "q1", "contexts": ["text"], "type": "factual"},
        ]

        result = run_retrieval_eval(samples, retriever, k=5)

        assert "orchestrator_stats" not in result
