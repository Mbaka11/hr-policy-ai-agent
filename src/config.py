"""Configuration and settings for the HR Policy AI Agent."""

from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
PROMPTS_DIR = ROOT_DIR / "prompts"

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# --- ChromaDB ---
CHROMA_COLLECTION_NAME = "hr_policies"

# --- Retrieval ---
RETRIEVAL_TOP_K = 5
RETRIEVAL_SCORE_THRESHOLD = 0.3  # Minimum similarity score (0-1); below this → fallback

# --- Text splitting ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Conversation memory ---
MEMORY_WINDOW_SIZE = 5  # Number of past turns to keep
