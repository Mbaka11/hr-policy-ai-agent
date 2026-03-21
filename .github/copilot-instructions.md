# HR Policy AI Agent — Copilot Instructions

## Project Context

This is a **Retrieval-Augmented Generation (RAG)** AI agent that answers employee questions about HR policies. It is a take-home assignment for the AI Internship at Talsom.

## Tech Stack

- **Language:** Python 3.11+
- **RAG Framework:** LangChain
- **LLM:** OpenAI GPT-4o-mini (default), GPT-4o (optional)
- **Embeddings:** OpenAI text-embedding-3-small
- **Vector Store:** ChromaDB (persistent, local)
- **UI:** Streamlit
- **PDF Loader:** PyMuPDF (`PyMuPDFLoader` from langchain-community)
- **Testing:** pytest
- **Containerization:** Docker (optional)

## Architecture

```
User → Streamlit → Agent → Retriever (ChromaDB) → Context + System Prompt → OpenAI → Response with Citations → User
```

Key modules in `src/`:

- `config.py` — Environment variables and settings
- `document_loader.py` — Load PDFs and text files using LangChain loaders
- `text_splitter.py` — Chunk documents with RecursiveCharacterTextSplitter
- `embeddings.py` — OpenAI embedding wrapper
- `vector_store.py` — ChromaDB collection management
- `retriever.py` — Similarity search retrieval logic
- `chain.py` — LangChain RAG chain assembly
- `agent.py` — Agent with tools for routing, escalation, and query classification
- `app.py` — Streamlit chat UI entry point

## Coding Conventions

- Use **type hints** on all function signatures
- Use **docstrings** (Google style) for public functions
- Keep functions focused and small (single responsibility)
- Use `python-dotenv` to load environment variables from `.env`
- Use `pathlib.Path` for file paths
- Constants and config in `src/config.py`
- All LangChain imports should use the latest v0.3+ import paths:
  - `from langchain_openai import ChatOpenAI, OpenAIEmbeddings`
  - `from langchain_chroma import Chroma`
  - `from langchain_community.document_loaders import PyMuPDFLoader`
  - Do NOT use `PyPDFLoader` or deprecated `from langchain.xxx` imports
- Error handling: catch specific exceptions, log errors, fail gracefully

## Agent Behavior

- The agent acts as a professional, empathetic HR policy assistant
- It MUST cite sources (document name, page/section) in responses
- It MUST NOT fabricate information — say "I don't have information on that" when unsure
- It MUST escalate sensitive topics (harassment, discrimination, legal issues) to HR
- It handles out-of-scope questions by politely redirecting

## File Organization

- `data/raw/` — Original HR policy PDFs (not committed, downloaded during setup)
- `data/chroma_db/` — ChromaDB persistent storage (gitignored)
- `prompts/system_prompt.md` — The agent's system prompt
- `scripts/ingest.py` — One-time document ingestion script
- `tests/` — pytest test files
- `docs/TECHNICAL_DOCUMENT.md` — Main deliverable document
- `docs/PROJECT_PLAN.md` — Phase-by-phase project plan with status tracking
