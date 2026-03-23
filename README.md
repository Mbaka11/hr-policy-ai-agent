# HR Policy AI Agent

An AI-powered agent that answers employee questions about HR policies using **Retrieval-Augmented Generation (RAG)**. Built with LangChain, OpenAI, ChromaDB, and Streamlit.

> Take-home assignment for the AI Internship at **Talsom**.

---

## Features

- **Accurate answers** grounded in real HR policy documents (RAG — no hallucination)
- **Source citations** — every answer references the document name and page/section
- **Conversational memory** — maintains context across follow-up questions (sliding window)
- **Smart routing** — classifies queries as HR, sensitive (→ escalate), or off-topic (→ redirect)
- **Confidence fallback** — when no relevant documents are found, the agent says so instead of guessing
- **Edge case handling** — escalates sensitive topics to HR, redirects off-topic queries
- **Interactive UI** — Streamlit chat interface with expandable source panel
- **Dockerized** — run the full stack with a single `docker run` command

## Architecture

```
User
 │
 ▼
Streamlit Chat UI (src/app.py)
 │
 ▼
HRAgent (src/agent.py)
 ├── Query Classifier (regex-based)
 │    ├── ESCALATE → warning + optional policy context
 │    ├── OFF_TOPIC → polite redirect
 │    └── HR_QUERY ──►─┐
 │                      ▼
 │              RAG Chain (src/chain.py)
 │                      │
 │          ┌───────────┴────────────┐
 │          ▼                        ▼
 │   Retriever                System Prompt
 │   (src/retriever.py)       (prompts/system_prompt.md)
 │          │
 │          ▼
 │   ChromaDB Vector Store
 │   (174 chunks, cosine similarity)
 │          │
 │          ▼
 │   Retrieved Context (top-k with score threshold)
 │          │
 │          └──────────┐
 │                     ▼
 │            ChatPromptTemplate
 │            (system + history + question + context)
 │                     │
 │                     ▼
 │            OpenAI GPT-4o-mini
 │                     │
 └─ Conversation Memory (sliding window, 5 turns)
                       │
                       ▼
              Answer + Source Citations
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| RAG Framework | LangChain v0.3+ |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | ChromaDB (persistent, local) |
| PDF Loader | PyMuPDF (`PyMuPDFLoader`) |
| UI | Streamlit |
| Testing | pytest (67 tests) |
| Containerization | Docker |

## Project Structure

```
hr-policy-ai-agent/
├── .github/
│   └── copilot-instructions.md     # Copilot project context
├── data/
│   └── raw/                         # HR policy documents (29 md + 2 pdf)
│       ├── code_of_conduct/
│       ├── employee_benefits/
│       ├── leave_policies/
│       ├── performance_evaluation/
│       ├── training_development/
│       └── vacation_pto/
├── docs/
│   ├── PROJECT_PLAN.md              # Phase-by-phase plan & status
│   └── TECHNICAL_DOCUMENT.md        # Deliverable — architecture & design
├── prompts/
│   └── system_prompt.md             # Agent system prompt
├── scripts/
│   └── ingest.py                    # One-time document ingestion
├── src/
│   ├── config.py                    # Settings & env vars
│   ├── document_loader.py           # Load Markdown & PDF documents
│   ├── text_splitter.py             # Chunk documents (markdown-aware)
│   ├── embeddings.py                # OpenAI embedding wrapper
│   ├── vector_store.py              # ChromaDB operations
│   ├── retriever.py                 # Similarity search with score filtering
│   ├── chain.py                     # RAG chain assembly
│   ├── agent.py                     # Agent with classification & routing
│   └── app.py                       # Streamlit entry point
├── tests/
│   ├── conftest.py                  # Shared fixtures
│   ├── test_config.py               # Config validation tests
│   ├── test_document_loader.py      # Document loading tests
│   ├── test_text_splitter.py        # Chunking tests
│   ├── test_agent.py                # Agent classification & routing tests
│   ├── test_edge_cases.py           # 5 edge case scenarios
│   └── test_integration.py          # Retriever & RAG chain tests
├── .env.example                     # API key template
├── Dockerfile                       # Container build
├── .dockerignore
├── pyproject.toml                   # Project config & dependencies
├── requirements.txt                 # Dependencies
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### 1. Clone & Install

```bash
git clone https://github.com/Mbaka11/hr-policy-ai-agent.git
cd hr-policy-ai-agent

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Ingest Documents

```bash
python scripts/ingest.py
```

This loads, chunks, and embeds the HR policy documents into the ChromaDB vector store.  
Use `--reset` to clear and rebuild from scratch.

### 4. Run the App

```bash
streamlit run src/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Docker

### Build

```bash
docker build -t hr-policy-agent .
```

### Run

**Recommended — load your `.env` file directly (keeps your key out of shell history):**

```bash
docker run -p 8501:8501 --env-file .env hr-policy-agent
```

**Alternative — pass the key inline:**

```bash
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... hr-policy-agent
```

> **Note:** Never use the placeholder `your-key-here` — replace it with your real `sk-...` key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys).

Open [http://localhost:8501](http://localhost:8501) once you see `You can now view your Streamlit app in your browser`.

### What's Inside

The Docker image includes the application code, prompts, ingestion script, and raw HR documents. On first startup, it automatically ingests the documents into an in-container ChromaDB store (`python scripts/ingest.py`), then launches Streamlit on port 8501.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Unit tests only (no API key required)
pytest tests/test_config.py tests/test_document_loader.py tests/test_text_splitter.py tests/test_agent.py -v

# Integration tests (requires OPENAI_API_KEY + populated ChromaDB)
pytest tests/test_integration.py tests/test_edge_cases.py -v
```

### Edge Cases Tested

| # | Scenario | Expected Behavior |
|---|----------|-------------------|
| 1 | Out-of-scope question (e.g., weather) | Polite redirect listing HR topics |
| 2 | Sensitive topic (harassment, threats) | Escalation warning + link to HR contacts |
| 3 | Multi-source / contradictory info | Retrieves from multiple docs, cites all |
| 4 | Vague / ambiguous question | Returns helpful answer without crashing |
| 5 | Inappropriate / unsafe request | Refuses gracefully, no sources |

## Documentation

- [Technical Document (English)](docs/TECHNICAL_DOCUMENT.md) — Architecture, system prompt, edge cases, evaluation
- [Document Technique (Français)](docs/TECHNICAL_DOCUMENT_FR.md) — Architecture, prompt système, cas limites, évaluation
- [Project Plan](docs/PROJECT_PLAN.md) — Phase-by-phase plan with status tracking

## License

MIT
