# HR Policy AI Agent

An AI-powered agent that answers employee questions about HR policies using **Retrieval-Augmented Generation (RAG)**. Built with LangChain, OpenAI, ChromaDB, and Streamlit.

> Take-home assignment for the AI Internship at **Talsom**.

---

## Features

- **Accurate answers** grounded in real HR policy documents (RAG — no hallucination)
- **Source citations** — every answer references the document name and page/section
- **Conversational** — maintains context across follow-up questions
- **Edge case handling** — escalates sensitive topics to HR, redirects off-topic queries
- **Interactive UI** — Streamlit chat interface with expandable source panel

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| RAG Framework | LangChain v0.3+ |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | ChromaDB (persistent, local) |
| UI | Streamlit |
| Testing | pytest |

## Project Structure

```
hr-policy-ai-agent/
├── .github/
│   └── copilot-instructions.md     # Copilot project context
├── data/
│   └── raw/                         # HR policy PDFs (add your own)
├── docs/
│   ├── PROJECT_PLAN.md              # Phase-by-phase plan & status
│   └── TECHNICAL_DOCUMENT.md        # Deliverable — architecture & design
├── examples/
│   └── sample_queries.md            # Example Q&A pairs
├── prompts/
│   └── system_prompt.md             # Agent system prompt
├── scripts/
│   └── ingest.py                    # One-time document ingestion
├── src/
│   ├── config.py                    # Settings & env vars
│   ├── document_loader.py           # Load PDFs/docs
│   ├── text_splitter.py             # Chunk documents
│   ├── embeddings.py                # OpenAI embedding wrapper
│   ├── vector_store.py              # ChromaDB operations
│   ├── retriever.py                 # Retrieval logic
│   ├── chain.py                     # RAG chain assembly
│   ├── agent.py                     # Agent with tools & routing
│   └── app.py                       # Streamlit entry point
├── tests/                           # pytest test suite
├── .env.example                     # API key template
├── pyproject.toml                   # Project config & dependencies
├── requirements.txt                 # Pinned dependencies
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

### 3. Add HR Policy Documents

Place your HR policy PDFs in the `data/raw/` directory.

### 4. Ingest Documents

```bash
python scripts/ingest.py
```

This loads, chunks, and embeds the documents into the ChromaDB vector store.

### 5. Run the App

```bash
streamlit run src/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Testing

```bash
pytest tests/ -v
```

## Documentation

- [Technical Document](docs/TECHNICAL_DOCUMENT.md) — Architecture, system prompt, edge cases, evaluation
- [Project Plan](docs/PROJECT_PLAN.md) — Phase-by-phase plan with status tracking

## License

MIT
