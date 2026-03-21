"""One-time script to ingest HR policy documents into ChromaDB.

Usage:
    python scripts/ingest.py [--reset]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so `src` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.vector_store import add_documents, get_collection_stats, reset_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest HR policy documents into ChromaDB")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing ChromaDB collection before ingesting",
    )
    args = parser.parse_args()

    # Step 0: Optionally reset
    if args.reset:
        logger.info("Resetting vector store...")
        reset_vector_store()

    # Step 1: Load documents
    logger.info("Loading documents from data/raw/...")
    documents = load_documents()
    if not documents:
        logger.error("No documents found. Place HR policy files in data/raw/ subdirectories.")
        sys.exit(1)
    logger.info("Loaded %d document(s)", len(documents))

    # Step 2: Split into chunks
    logger.info("Splitting documents into chunks...")
    chunks = split_documents(documents)
    logger.info("Created %d chunk(s)", len(chunks))

    # Step 3: Embed and store
    logger.info("Embedding and storing in ChromaDB...")
    add_documents(chunks)

    # Step 4: Report stats
    stats = get_collection_stats()
    logger.info(
        "Ingestion complete! Collection '%s' now has %d documents.",
        stats["collection_name"],
        stats["document_count"],
    )


if __name__ == "__main__":
    main()
