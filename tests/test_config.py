"""Unit tests for configuration module."""

from pathlib import Path

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_DIR,
    DATA_DIR,
    MEMORY_WINDOW_SIZE,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_MODEL,
    PROMPTS_DIR,
    RAW_DATA_DIR,
    RETRIEVAL_SCORE_THRESHOLD,
    RETRIEVAL_TOP_K,
    ROOT_DIR,
)


class TestConfigPaths:
    """Verify that all configured paths are well-formed."""

    def test_root_dir_exists(self):
        assert ROOT_DIR.exists()
        assert ROOT_DIR.is_dir()

    def test_data_dir_under_root(self):
        assert DATA_DIR == ROOT_DIR / "data"

    def test_raw_data_dir_under_data(self):
        assert RAW_DATA_DIR == DATA_DIR / "raw"

    def test_chroma_db_dir_under_data(self):
        assert CHROMA_DB_DIR == DATA_DIR / "chroma_db"

    def test_prompts_dir_under_root(self):
        assert PROMPTS_DIR == ROOT_DIR / "prompts"


class TestConfigConstants:
    """Verify default constant values are sensible."""

    def test_openai_model_default(self):
        assert OPENAI_MODEL in ("gpt-4o-mini", "gpt-4o")

    def test_embedding_model(self):
        assert OPENAI_EMBEDDING_MODEL == "text-embedding-3-small"

    def test_chroma_collection_name(self):
        assert CHROMA_COLLECTION_NAME == "hr_policies"

    def test_retrieval_top_k_positive(self):
        assert RETRIEVAL_TOP_K > 0

    def test_retrieval_score_threshold_range(self):
        assert 0.0 <= RETRIEVAL_SCORE_THRESHOLD <= 1.0

    def test_chunk_size_positive(self):
        assert CHUNK_SIZE > 0

    def test_chunk_overlap_less_than_size(self):
        assert 0 <= CHUNK_OVERLAP < CHUNK_SIZE

    def test_memory_window_size_positive(self):
        assert MEMORY_WINDOW_SIZE > 0
