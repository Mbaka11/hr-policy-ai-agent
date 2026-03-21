"""OpenAI embedding model wrapper."""

from langchain_openai import OpenAIEmbeddings

from src.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL


def get_embeddings() -> OpenAIEmbeddings:
    """Create an OpenAI embeddings instance.

    Returns:
        Configured OpenAIEmbeddings using text-embedding-3-small.
    """
    return OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )
