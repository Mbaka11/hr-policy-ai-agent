"""Split documents into chunks for embedding and retrieval."""

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Create a configured text splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks, preserving metadata.

    Uses Markdown-aware separators so chunks break at heading boundaries
    whenever possible, falling back to paragraphs and sentences.

    Args:
        documents: List of LangChain Documents to split.

    Returns:
        List of chunked Documents with original metadata preserved.
    """
    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)
    logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
    return chunks
