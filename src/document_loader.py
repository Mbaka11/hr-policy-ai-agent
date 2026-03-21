"""Load HR policy documents (Markdown and PDF) from the data/raw directory."""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

from src.config import RAW_DATA_DIR

logger = logging.getLogger(__name__)

# Map subfolder names to human-readable policy categories
CATEGORY_MAP = {
    "code_of_conduct": "Code of Conduct",
    "employee_benefits": "Employee Benefits",
    "leave_policies": "Leave Policies",
    "performance_evaluation": "Performance Evaluation",
    "training_development": "Training & Development",
    "vacation_pto": "Vacation & PTO",
    "hr_communications": "HR Communications",
}


def _load_markdown(file_path: Path, category: str) -> list[Document]:
    """Load a single Markdown file as a LangChain Document."""
    text = file_path.read_text(encoding="utf-8")
    metadata = {
        "source": file_path.name,
        "source_path": str(file_path.relative_to(RAW_DATA_DIR)),
        "category": category,
        "file_type": "markdown",
    }
    return [Document(page_content=text, metadata=metadata)]


def _load_pdf(file_path: Path, category: str) -> list[Document]:
    """Load a PDF file using PyMuPDF. Returns one Document per page."""
    loader = PyMuPDFLoader(str(file_path))
    docs = loader.load()
    for doc in docs:
        doc.metadata.update({
            "source": file_path.name,
            "source_path": str(file_path.relative_to(RAW_DATA_DIR)),
            "category": category,
            "file_type": "pdf",
        })
    return docs


def load_documents(data_dir: Path | None = None) -> list[Document]:
    """Load all HR policy documents from the raw data directory.

    Walks through subdirectories in data/raw/, categorizing documents
    by their parent folder name. Supports .md and .pdf files.

    Args:
        data_dir: Override the default RAW_DATA_DIR path.

    Returns:
        A list of LangChain Documents with metadata (source, category, file_type).
    """
    data_dir = data_dir or RAW_DATA_DIR
    documents: list[Document] = []

    if not data_dir.exists():
        logger.warning("Data directory does not exist: %s", data_dir)
        return documents

    for subfolder in sorted(data_dir.iterdir()):
        if not subfolder.is_dir():
            continue

        category = CATEGORY_MAP.get(subfolder.name, subfolder.name)

        for file_path in sorted(subfolder.iterdir()):
            if not file_path.is_file():
                continue

            try:
                if file_path.suffix.lower() == ".md":
                    docs = _load_markdown(file_path, category)
                elif file_path.suffix.lower() == ".pdf":
                    docs = _load_pdf(file_path, category)
                else:
                    logger.debug("Skipping unsupported file: %s", file_path)
                    continue

                documents.extend(docs)
                logger.info("Loaded %d chunk(s) from %s", len(docs), file_path.name)

            except Exception:
                logger.exception("Failed to load %s", file_path)

    logger.info("Total documents loaded: %d", len(documents))
    return documents
