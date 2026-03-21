"""Retrieval logic — wrap ChromaDB as a LangChain retriever with score filtering."""

import logging
from typing import Optional

from langchain_core.documents import Document

from src.config import RETRIEVAL_SCORE_THRESHOLD, RETRIEVAL_TOP_K
from src.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def retrieve_with_scores(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    score_threshold: float = RETRIEVAL_SCORE_THRESHOLD,
    category_filter: Optional[str] = None,
) -> list[tuple[Document, float]]:
    """Retrieve relevant documents with similarity scores.

    Uses ChromaDB's similarity_search_with_relevance_scores which returns
    scores in [0, 1] range where higher = more similar.

    Args:
        query: The user's question.
        top_k: Maximum number of results to return.
        score_threshold: Minimum similarity score to include a result.
        category_filter: Optional policy category to filter by.

    Returns:
        List of (Document, score) tuples, sorted by descending relevance.
    """
    vector_store = get_vector_store()

    filter_dict = None
    if category_filter:
        filter_dict = {"category": category_filter}

    results = vector_store.similarity_search_with_relevance_scores(
        query,
        k=top_k,
        filter=filter_dict,
    )

    # Filter by score threshold
    filtered = [(doc, score) for doc, score in results if score >= score_threshold]

    logger.info(
        "Retrieved %d/%d chunks above threshold %.2f for query: '%s'",
        len(filtered),
        len(results),
        score_threshold,
        query[:80],
    )

    return filtered


def retrieve(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    category_filter: Optional[str] = None,
) -> list[Document]:
    """Retrieve relevant documents (without scores).

    Convenience wrapper that returns just the documents.

    Args:
        query: The user's question.
        top_k: Maximum number of results.
        category_filter: Optional policy category filter.

    Returns:
        List of relevant Documents, or empty list if nothing passes threshold.
    """
    results = retrieve_with_scores(query, top_k=top_k, category_filter=category_filter)
    return [doc for doc, _score in results]


def get_retriever(top_k: int = RETRIEVAL_TOP_K):
    """Get a LangChain-compatible retriever for use in chains.

    Returns:
        A LangChain retriever backed by ChromaDB with MMR search.
    """
    vector_store = get_vector_store()
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k,
            "fetch_k": top_k * 3,
        },
    )
