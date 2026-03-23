# Project Plan: HR Policy AI Agent (RAG)

## Overview

Build a full-featured RAG-based AI agent for HR policy Q&A.

- **Stack:** Python + LangChain + OpenAI (GPT-4o-mini) + ChromaDB + Streamlit
- **Deadline:** March 23, 2026 at 12:00 PM (noon)
- **Deliverables:** Working prototype, Technical document, Public GitHub repo, Video presentation

---

## Phases & Status

### Phase 1: Project Foundation — `✅ COMPLETE`

| #   | Task                                                                             | Status  |
| --- | -------------------------------------------------------------------------------- | ------- |
| 1   | Finalize folder structure                                                        | ✅ Done |
| 2   | Create config files (.gitignore, .env.example, pyproject.toml, requirements.txt) | ✅ Done |
| 3   | Create .github/copilot-instructions.md                                           | ✅ Done |
| 4   | Write initial README.md                                                          | ✅ Done |

### Phase 2: Data Sourcing & Ingestion — `✅ COMPLETE`

| #   | Task                                                 | Status                                      |
| --- | ---------------------------------------------------- | ------------------------------------------- |
| 5   | Source 5-10 publicly available HR policy documents   | ✅ Done (29 md + 2 pdf across 6 categories) |
| 6   | Implement document loader (`src/document_loader.py`) | ✅ Done                                     |
| 7   | Implement text splitter (`src/text_splitter.py`)     | ✅ Done                                     |
| 8   | Implement embeddings wrapper (`src/embeddings.py`)   | ✅ Done                                     |
| 9   | Implement vector store (`src/vector_store.py`)       | ✅ Done                                     |
| 10  | Create ingestion script (`scripts/ingest.py`)        | ✅ Done                                     |

### Phase 3: RAG Pipeline & Agent — `✅ COMPLETE`

| #   | Task                                              | Status  |
| --- | ------------------------------------------------- | ------- |
| 11  | Implement retriever (`src/retriever.py`)          | ✅ Done |
| 12  | Design system prompt (`prompts/system_prompt.md`) | ✅ Done |
| 13  | Implement RAG chain (`src/chain.py`)              | ✅ Done |
| 14  | Implement agent layer (`src/agent.py`)            | ✅ Done |
| 15  | Implement config module (`src/config.py`)         | ✅ Done |

### Phase 4: Streamlit UI — `✅ COMPLETE`

| #   | Task                                | Status  |
| --- | ----------------------------------- | ------- |
| 16  | Build chat interface (`src/app.py`) | ✅ Done |
| 17  | Add source document display         | ✅ Done |
| 18  | Add sidebar with project info       | ✅ Done |

### Phase 5: Testing & Evaluation — `✅ COMPLETE`

| #   | Task                                        | Status  |
| --- | ------------------------------------------- | ------- |
| 19  | Write unit tests (config, loader, splitter, agent) | ✅ Done (47 tests) |
| 20  | Write edge case tests (5 scenarios)         | ✅ Done (9 tests) |
| 21  | Write integration tests (retriever, chain)  | ✅ Done (6 tests) |
| 22  | All 67 tests passing                        | ✅ Done |

### Phase 6: Documentation & Deployment — `🔄 IN PROGRESS`

| #   | Task                                                                              | Status  |
| --- | --------------------------------------------------------------------------------- | ------- |
| 23  | Create Dockerfile + .dockerignore                                                 | ✅ Done |
| 24  | Polish README with architecture diagram, Docker, testing sections                 | ✅ Done |
| 25  | Write TECHNICAL_DOCUMENT.md (architecture, system prompt, edge cases, evaluation) | ⬜      |

### Phase 7: Video — `⬜ NOT STARTED`

| #   | Task                                               | Status |
| --- | -------------------------------------------------- | ------ |
| 26  | Prepare video outline & talking points             | ⬜     |
| 27  | Record 5-10 min video (1 min EN intro, rest in FR) | ⬜     |

---

## Architecture

```
User ──► Streamlit UI ──► Agent/Chain
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
             Query Classifier      RAG Retriever
             (on/off topic)            │
                                       ▼
                                  ChromaDB
                                  (vectors)
                                       │
                                       ▼
                              Retrieved Context
                                       │
                                       ▼
                            System Prompt + Context
                                       │
                                       ▼
                              OpenAI GPT-4o-mini
                                       │
                                       ▼
                            Response + Citations
                                       │
                                       ▼
                                Streamlit UI ──► User
```

## Key Decisions

| Decision     | Choice                       | Rationale                                                                     |
| ------------ | ---------------------------- | ----------------------------------------------------------------------------- |
| LLM          | GPT-4o-mini                  | Cost-efficient, fast, good quality for a prototype                            |
| Embeddings   | text-embedding-3-small       | Good quality/cost ratio                                                       |
| Vector Store | ChromaDB (persistent)        | Lightweight, local, no external service needed                                |
| Framework    | LangChain                    | Well-documented, lots of integrations, industry standard                      |
| UI           | Streamlit                    | Fast to build, looks polished for demos                                       |
| PDF Loader   | PyMuPDF (PyMuPDFLoader)      | Faster, more robust than pypdf, better text extraction                        |
| Chunk size   | ~1000 chars, ~200 overlap    | Standard for document Q&A                                                     |
| Memory       | 5-turn conversation window   | Sufficient for Q&A context                                                    |
| Confidence   | Score threshold on retrieval | Return fallback message instead of hallucinating when no relevant chunk found |
| Docker       | Optional Dockerfile          | Shows production awareness, makes demo portable                               |

## Timeline

| Day              | Focus                            | Phases     |
| ---------------- | -------------------------------- | ---------- |
| March 21 (Day 1) | Foundation + Data + RAG Pipeline | Phases 1-3 |
| March 22 (Day 2) | UI + Tests + Docs + Video        | Phases 4-7 |

## Verification Checklist

- [ ] `pip install -r requirements.txt` — no errors
- [ ] `python scripts/ingest.py` with PDFs in `data/raw/` — ChromaDB populated
- [ ] `streamlit run src/app.py` — UI loads, HR questions get cited answers
- [ ] `pytest tests/` — all tests pass
- [ ] 5 edge cases tested manually through the UI
- [ ] `docs/TECHNICAL_DOCUMENT.md` covers all 4 required sections
- [ ] README setup instructions work from a fresh clone
