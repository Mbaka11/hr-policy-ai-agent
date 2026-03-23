"""Unit tests for the text splitter module."""

from langchain_core.documents import Document

from src.text_splitter import get_text_splitter, split_documents
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


class TestGetTextSplitter:
    """Test the text splitter configuration."""

    def test_returns_splitter_instance(self):
        splitter = get_text_splitter()
        assert splitter is not None
        assert splitter._chunk_size == CHUNK_SIZE
        assert splitter._chunk_overlap == CHUNK_OVERLAP

    def test_markdown_separators_included(self):
        splitter = get_text_splitter()
        assert "\n## " in splitter._separators
        assert "\n### " in splitter._separators


class TestSplitDocuments:
    """Test document splitting behavior."""

    def test_short_document_not_split(self):
        """A document shorter than CHUNK_SIZE should remain as one chunk."""
        doc = Document(page_content="Short content.", metadata={"source": "test.md"})
        chunks = split_documents([doc])
        assert len(chunks) == 1
        assert chunks[0].page_content == "Short content."

    def test_long_document_is_split(self):
        """A document much larger than CHUNK_SIZE should be split into multiple chunks."""
        long_text = "This is a sentence. " * 500  # ~10,000 chars
        doc = Document(page_content=long_text, metadata={"source": "long.md"})
        chunks = split_documents([doc])
        assert len(chunks) > 1

    def test_metadata_preserved_after_split(self):
        long_text = "Word " * 1000
        doc = Document(
            page_content=long_text,
            metadata={"source": "policy.md", "category": "Benefits"},
        )
        chunks = split_documents([doc])
        for chunk in chunks:
            assert chunk.metadata["source"] == "policy.md"
            assert chunk.metadata["category"] == "Benefits"

    def test_empty_list_returns_empty(self):
        chunks = split_documents([])
        assert chunks == []

    def test_chunk_sizes_within_limits(self):
        """Each chunk should be roughly within CHUNK_SIZE (small overruns are OK)."""
        long_text = "A medium sentence for testing. " * 300
        doc = Document(page_content=long_text, metadata={"source": "x.md"})
        chunks = split_documents([doc])
        for chunk in chunks:
            # Allow some tolerance for separator keeping
            assert len(chunk.page_content) <= CHUNK_SIZE + 100
