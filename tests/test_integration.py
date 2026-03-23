"""Integration tests for the retriever and RAG chain.

These tests require a populated ChromaDB and a valid OPENAI_API_KEY.
They are skipped automatically if the key is missing.
"""

import os

import pytest

requires_api = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping integration test",
)


@requires_api
class TestRetriever:
    """Test retrieval from the real ChromaDB vector store."""

    def test_retrieve_returns_documents(self):
        from src.retriever import retrieve

        docs = retrieve("What is the vacation policy?")
        assert isinstance(docs, list)
        assert len(docs) > 0

    def test_retrieve_with_scores_filters_correctly(self):
        from src.retriever import retrieve_with_scores

        results = retrieve_with_scores("employee dental benefits", score_threshold=0.3)
        for _doc, score in results:
            assert score >= 0.3

    def test_retrieve_irrelevant_query_may_return_fewer(self):
        from src.retriever import retrieve_with_scores

        results = retrieve_with_scores(
            "quantum physics black hole entropy", score_threshold=0.5
        )
        # Irrelevant query with a higher threshold should return few or no results
        assert len(results) <= 3

    def test_category_filter(self):
        from src.retriever import retrieve_with_scores

        results = retrieve_with_scores(
            "parental leave", category_filter="Leave Policies"
        )
        for doc, _score in results:
            assert doc.metadata.get("category") == "Leave Policies"


@requires_api
class TestRAGChain:
    """Test the full RAG chain end-to-end."""

    def test_ask_returns_expected_keys(self):
        from src.chain import ask

        result = ask("What are the employee dental benefits?")
        assert "answer" in result
        assert "sources" in result
        assert "has_context" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 10

    def test_ask_provides_sources(self):
        from src.chain import ask

        result = ask("How many vacation days does an employee get?")
        if result["has_context"]:
            assert len(result["sources"]) > 0
            assert "source" in result["sources"][0]
