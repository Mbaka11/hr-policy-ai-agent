"""RAG chain assembly — retriever + prompt + LLM with conversation memory."""

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    PROMPTS_DIR,
    RETRIEVAL_SCORE_THRESHOLD,
)
from src.retriever import retrieve_with_scores

logger = logging.getLogger(__name__)

# Confidence fallback message when no relevant documents are found
NO_CONTEXT_FALLBACK = (
    "I don't have specific information about that in the HR policy documents "
    "I have access to. I recommend contacting the HR department directly for assistance."
)


def _load_system_prompt() -> str:
    """Load the system prompt from prompts/system_prompt.md."""
    prompt_path = PROMPTS_DIR / "system_prompt.md"
    return prompt_path.read_text(encoding="utf-8")


def _format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a context string for the prompt."""
    if not docs:
        return "No relevant HR policy documents were found for this question."

    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        category = doc.metadata.get("category", "Unknown")
        page = doc.metadata.get("page", None)

        header = f"[Document {i}] Source: {source} | Category: {category}"
        if page is not None:
            header += f" | Page: {page + 1}"

        formatted.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)


def get_llm() -> ChatOpenAI:
    """Create the OpenAI LLM instance."""
    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0.1,
        max_tokens=1024,
    )


def build_prompt() -> ChatPromptTemplate:
    """Build the chat prompt template with system prompt, history, and user input."""
    system_prompt = _load_system_prompt()

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])


def ask(
    question: str,
    chat_history: list | None = None,
) -> dict:
    """Run the full RAG pipeline: retrieve → build prompt → call LLM.

    Args:
        question: The user's HR policy question.
        chat_history: List of previous (HumanMessage, AIMessage) tuples for context.

    Returns:
        Dictionary with keys:
        - answer: The LLM's response string
        - sources: List of source document metadata dicts
        - has_context: Whether relevant documents were found
    """
    chat_history = chat_history or []

    # Step 1: Retrieve with scores for confidence check
    results_with_scores = retrieve_with_scores(question)
    documents = [doc for doc, _score in results_with_scores]
    has_context = len(documents) > 0

    # Step 2: Format context
    context = _format_docs(documents)

    # Step 3: Build and invoke chain
    prompt = build_prompt()
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "question": question,
    })

    # Step 4: Extract source metadata for the UI
    sources = []
    for doc in documents:
        source_info = {
            "source": doc.metadata.get("source", "Unknown"),
            "category": doc.metadata.get("category", "Unknown"),
            "page": doc.metadata.get("page"),
        }
        if source_info not in sources:
            sources.append(source_info)

    logger.info(
        "Generated answer for '%s' using %d source(s), has_context=%s",
        question[:60],
        len(sources),
        has_context,
    )

    return {
        "answer": answer,
        "sources": sources,
        "has_context": has_context,
    }
