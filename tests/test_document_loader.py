"""Unit tests for the document loader module."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.config import RAW_DATA_DIR
from src.document_loader import (
    CATEGORY_MAP,
    _load_markdown,
    load_documents,
)


class TestLoadMarkdown:
    """Test loading individual Markdown files.

    _load_markdown uses file_path.relative_to(RAW_DATA_DIR), so we create
    temp files *inside* a mock RAW_DATA_DIR structure to avoid ValueError.
    """

    def test_load_markdown_returns_single_document(self, tmp_path):
        sub = tmp_path / "test_cat"
        sub.mkdir()
        md_file = sub / "test_policy.md"
        md_file.write_text("# Policy\nSome content.", encoding="utf-8")
        # Patch RAW_DATA_DIR so relative_to works
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.document_loader.RAW_DATA_DIR", tmp_path)
            docs = _load_markdown(md_file, "Test Category")
        assert len(docs) == 1
        assert isinstance(docs[0], Document)

    def test_load_markdown_contains_text(self, tmp_path):
        sub = tmp_path / "cat"
        sub.mkdir()
        md_file = sub / "test_policy.md"
        md_file.write_text("# Title\nBody text here.", encoding="utf-8")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.document_loader.RAW_DATA_DIR", tmp_path)
            docs = _load_markdown(md_file, "Cat")
        assert "Body text here" in docs[0].page_content

    def test_load_markdown_metadata(self, tmp_path):
        sub = tmp_path / "benefits"
        sub.mkdir()
        md_file = sub / "benefits.md"
        md_file.write_text("content", encoding="utf-8")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.document_loader.RAW_DATA_DIR", tmp_path)
            docs = _load_markdown(md_file, "Employee Benefits")
        assert docs[0].metadata["category"] == "Employee Benefits"
        assert docs[0].metadata["file_type"] == "markdown"
        assert docs[0].metadata["source"] == "benefits.md"


class TestLoadDocuments:
    """Test the full document loading pipeline."""

    def test_load_documents_returns_list(self):
        """Loading from the real data directory should return documents."""
        docs = load_documents()
        assert isinstance(docs, list)
        assert len(docs) > 0

    def test_all_documents_have_required_metadata(self):
        docs = load_documents()
        for doc in docs:
            assert "source" in doc.metadata
            assert "category" in doc.metadata
            assert "file_type" in doc.metadata
            assert doc.metadata["file_type"] in ("markdown", "pdf")

    def test_known_categories_present(self):
        """At least some known categories should appear."""
        docs = load_documents()
        categories = {doc.metadata["category"] for doc in docs}
        # We know the data dir has these subfolders
        assert "Leave Policies" in categories
        assert "Employee Benefits" in categories

    def test_empty_directory_returns_empty(self, tmp_path):
        """An empty directory should return an empty list without errors."""
        docs = load_documents(data_dir=tmp_path)
        assert docs == []

    def test_nonexistent_directory_returns_empty(self, tmp_path):
        fake_dir = tmp_path / "does_not_exist"
        docs = load_documents(data_dir=fake_dir)
        assert docs == []

    def test_category_map_completeness(self):
        """Every subfolder in data/raw should have a CATEGORY_MAP entry."""
        if not RAW_DATA_DIR.exists():
            pytest.skip("data/raw not present")
        for subfolder in RAW_DATA_DIR.iterdir():
            if subfolder.is_dir():
                assert subfolder.name in CATEGORY_MAP, (
                    f"Subfolder '{subfolder.name}' missing from CATEGORY_MAP"
                )
