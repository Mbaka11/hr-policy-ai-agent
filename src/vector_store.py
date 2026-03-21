"""ChromaDB vector store management."""

import logging
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import CHROMA_COLLECTION_NAME, CHROMA_DB_DIR
from src.embeddings import get_embeddings

logger = logging.getLogger(__name__)


def get_vector_store() -> Chroma:
    """Get or create the ChromaDB vector store.

    Returns:
        A Chroma instance backed by persistent local storage.
    """
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DB_DIR),
    )


def add_documents(chunks: list[Document]) -> Chroma:
    """Embed and store document chunks in ChromaDB.

    Args:
        chunks: Pre-split document chunks with metadata.

    Returns:
        The Chroma vector store with documents added.
    """
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    logger.info("Added %d chunks to ChromaDB collection '%s'", len(chunks), CHROMA_COLLECTION_NAME)
    return vector_store


def reset_vector_store() -> None:
    """Delete the existing ChromaDB collection and storage.

    Use this before re-ingesting to start fresh.
    """
    if CHROMA_DB_DIR.exists():
        shutil.rmtree(CHROMA_DB_DIR)
        logger.info("Deleted ChromaDB storage at %s", CHROMA_DB_DIR)
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)


def get_collection_stats() -> dict:
    """Get basic stats about the current ChromaDB collection.

    Returns:
        Dictionary with collection name and document count.
    """
    vector_store = get_vector_store()
    collection = vector_store._collection
    return {
        "collection_name": CHROMA_COLLECTION_NAME,
        "document_count": collection.count(),
    }
