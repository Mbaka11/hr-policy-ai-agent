# Technical Document — HR Policy AI Agent

> **Project:** HR Policy Q&A Agent (RAG)
> **Author:** Mbaka
> **Date:** March 2026
> **Repository:** [github.com/Mbaka11/hr-policy-ai-agent](https://github.com/Mbaka11/hr-policy-ai-agent)

---

## Table of Contents

1. [Agent Architecture](#1-agent-architecture)
2. [System Instructions](#2-system-instructions)
3. [Edge Case Handling](#3-edge-case-handling)
4. [Evaluation Strategy](#4-evaluation-strategy)

---

## 1. Agent Architecture

### 1.1 Overview

This agent uses a **Retrieval-Augmented Generation (RAG)** architecture to answer employee questions about HR policies. Rather than relying solely on the LLM's training data (which would hallucinate company-specific policies), the system retrieves relevant policy documents from a vector store and injects them as context into the prompt before generating a response.

### 1.2 Technology Choices

| Component            | Choice                        | Rationale                                                                                                                                                |
| -------------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LLM**              | OpenAI GPT-4o-mini            | Best cost/quality ratio for Q&A tasks. Low latency (~1-2s), $0.15/1M input tokens. Sufficient reasoning for policy retrieval without the cost of GPT-4o. |
| **Embeddings**       | OpenAI text-embedding-3-small | 1536-dimensional vectors, strong semantic understanding, low cost ($0.02/1M tokens).                                                                     |
| **Vector Store**     | ChromaDB (persistent, local)  | Zero-infrastructure setup, persistent local storage, native LangChain integration. Ideal for a prototype with <1,000 documents.                          |
| **RAG Framework**    | LangChain v0.3+               | Mature abstractions for document loading, splitting, embedding, retrieval, and chain composition. Avoids reinventing pipeline glue code.                 |
| **PDF Loader**       | PyMuPDF (PyMuPDFLoader)       | Faster and more accurate than PyPDF for text extraction, especially with complex PDF layouts.                                                            |
| **UI**               | Streamlit                     | Rapid prototyping for chat interfaces. Built-in `chat_input`, `chat_message`, session state, and deployment support.                                     |
| **Containerization** | Docker                        | Reproducible environment. Single `docker run` command deploys the full stack.                                                                            |

### 1.3 Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER QUERY                                  │
│                  "How many vacation days do I get?"                  │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STREAMLIT UI (src/app.py)                       │
│  • Chat interface with message history                              │
│  • Session state management                                         │
│  • Source citation display (expandable)                              │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  HR AGENT (src/agent.py)                             │
│                                                                     │
│  ┌──────────────────────────────────┐                               │
│  │    QUERY CLASSIFIER              │                               │
│  │    (regex pattern matching)      │                               │
│  │                                  │                               │
│  │  ESCALATE ──► Escalation msg     │  Sensitive: harassment,       │
│  │               + optional policy  │  discrimination, threats,     │
│  │               context            │  legal, self-harm             │
│  │                                  │                               │
│  │  OFF_TOPIC ──► Polite redirect   │  Unrelated: weather, code,   │
│  │               with HR topic list │  jokes, sports, recipes       │
│  │                                  │                               │
│  │  HR_QUERY ──► RAG Pipeline ───┐  │  Default: any HR question    │
│  └──────────────────────────────┘│  │                               │
│                                   │  │                               │
│  Conversation Memory              │  │                               │
│  (sliding window, 5 turns)        │  │                               │
└───────────────────────────────────┼──┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   RAG CHAIN (src/chain.py)                           │
│                                                                     │
│  Step 1: RETRIEVE                                                   │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  Retriever (src/retriever.py)                           │        │
│  │  • Query → embedding via text-embedding-3-small         │        │
│  │  • Cosine similarity search in ChromaDB                 │        │
│  │  • Top-5 results, filtered by score threshold (≥0.3)    │        │
│  │  • Returns: documents + metadata + relevance scores     │        │
│  └─────────────────────────┬───────────────────────────────┘        │
│                             │                                        │
│  Step 2: AUGMENT            ▼                                        │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  Prompt Assembly                                        │        │
│  │  • System prompt (prompts/system_prompt.md)             │        │
│  │  • Retrieved context (formatted with source headers)    │        │
│  │  • Conversation history (last 5 turns)                  │        │
│  │  • User question                                        │        │
│  └─────────────────────────┬───────────────────────────────┘        │
│                             │                                        │
│  Step 3: GENERATE           ▼                                        │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  OpenAI GPT-4o-mini                                     │        │
│  │  • Temperature: 0.1 (deterministic, factual)            │        │
│  │  • Max tokens: 1024                                     │        │
│  │  • Output parsed to string                              │        │
│  └─────────────────────────┬───────────────────────────────┘        │
│                             │                                        │
│  Step 4: CONFIDENCE CHECK   ▼                                        │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  If no documents passed the score threshold:            │        │
│  │  → Return fallback: "I don't have specific info..."     │        │
│  │  If documents found:                                    │        │
│  │  → Return answer + source citations                     │        │
│  └─────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     RESPONSE TO USER                                │
│  • Answer text with policy details                                  │
│  • Source citations (document name, category, page)                 │
│  • Expandable source panel in UI                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.4 Data Pipeline (Ingestion)

Before the agent can answer questions, HR policy documents must be ingested:

```
data/raw/              →  Document Loader        →  Text Splitter          →  Embeddings + ChromaDB
(29 Markdown + 2 PDF)     (PyMuPDFLoader for      (RecursiveCharacter       (text-embedding-3-small
 across 6 categories)      PDFs, direct read        TextSplitter,            → 1536-dim vectors
                           for Markdown)            chunk_size=1000,         → persistent local store)
                                                    overlap=200,
                                                    markdown-aware
                                                    separators)
                                                         │
                                                         ▼
                                                   174 chunks stored
                                                   in ChromaDB
```

**Key design decisions:**

- **Chunk size of 1000** with **200 overlap**: balances context richness (enough text for the LLM to reason about) against retrieval precision (not so large that irrelevant content dilutes the signal).
- **Markdown-aware separators** (`## `, `### `, `\n\n`, etc.): splits at heading boundaries first, preserving the logical structure of policy documents.
- **Metadata preserved**: each chunk carries its source filename, category, file type, and page number — enabling accurate citations in responses.

### 1.5 Key Configuration Parameters

| Parameter                   | Value | Purpose                                            |
| --------------------------- | ----- | -------------------------------------------------- |
| `RETRIEVAL_TOP_K`           | 5     | Maximum documents retrieved per query              |
| `RETRIEVAL_SCORE_THRESHOLD` | 0.3   | Minimum cosine similarity to include a result      |
| `CHUNK_SIZE`                | 1000  | Characters per chunk                               |
| `CHUNK_OVERLAP`             | 200   | Overlap between consecutive chunks                 |
| `MEMORY_WINDOW_SIZE`        | 5     | Conversational turns retained                      |
| `temperature`               | 0.1   | Low temperature for factual, deterministic answers |

---

## 2. System Instructions

### 2.1 System Prompt

The full system prompt is stored in `prompts/system_prompt.md` and injected at the beginning of every LLM call. Below is the complete prompt with annotations:

```
You are an **HR Policy Assistant** for a professional services company with
approximately 200 employees. Your role is to help employees find accurate
answers to questions about internal HR policies.

## Core Behavioral Rules

1. **Only answer based on the provided context.** Every response must be grounded
   in the retrieved HR policy documents. If the context does not contain enough
   information to answer confidently, say so clearly.

2. **Always cite your sources.** Include the document name (and page/section when
   available) at the end of your answer.
   Format: 📄 Source: [document_name], [category]

3. **Never fabricate information.** If unsure or the documents don't cover the
   topic, respond with: "I don't have specific information about that in the HR
   policy documents I have access to. I recommend contacting the HR department
   directly for assistance."

4. **Be professional, empathetic, and clear.** Use a warm but professional tone.
   Employees may be asking about sensitive personal situations.

5. **Be concise but thorough.** Provide complete answers without unnecessary
   filler. Use bullet points or numbered lists when presenting multiple items.

## Escalation Rules

Escalate to a human HR representative when:
- Harassment or discrimination complaints
- Legal questions or disputes
- Mental health crises → direct to EFAP
- Requests for personal employee data
- Requests to make HR decisions
- Whistleblowing or ethics violations

## Edge Case Instructions

- Out-of-scope → redirect to appropriate department
- Contradictory sources → present both with sources, recommend contacting HR
- Vague questions → ask for clarification or answer most likely interpretation
- Inappropriate queries → politely decline, redirect to legitimate HR questions

## Response Format

1. Direct answer
2. Supporting details
3. Important notes / caveats
4. Source citation(s)

## Context
{context}
```

### 2.2 Prompt Design Rationale

| Aspect                        | Decision                                    | Why                                                                       |
| ----------------------------- | ------------------------------------------- | ------------------------------------------------------------------------- |
| **Grounding rule**            | "Only answer based on the provided context" | Prevents hallucination — the #1 risk in RAG systems                       |
| **Citation mandate**          | Forced source format                        | Builds trust, enables verification, required by assignment                |
| **Explicit refusal template** | "I don't have specific information…"        | Avoids vague non-answers; gives the user a clear next step                |
| **Empathetic tone**           | "warm but professional"                     | HR queries often involve personal/stressful situations                    |
| **Escalation rules**          | Listed explicitly in prompt                 | The LLM sees these rules every call — no ambiguity about when to escalate |
| **Response structure**        | 4-part format                               | Ensures consistent, scannable answers                                     |
| **Temperature 0.1**           | Near-deterministic                          | Factual Q&A needs consistency, not creativity                             |

---

## 3. Edge Case Handling

### 3.1 Overview

The agent handles edge cases through a **two-layer defense**: a fast regex-based classifier (pre-LLM) and behavioral rules in the system prompt (intra-LLM).

```
User Query
    │
    ▼
┌──────────────────┐
│ Layer 1: REGEX   │  Fast, deterministic, no API cost
│ (src/agent.py)   │
│                  │
│ ESCALATE? ──────►│──► Escalation message + optional RAG context
│ OFF_TOPIC? ─────►│──► Redirect message (no LLM call)
│ HR_QUERY ───────►│──► Proceed to RAG
└──────────────────┘
    │ (HR_QUERY only)
    ▼
┌──────────────────┐
│ Layer 2: LLM     │  Handles nuance the regex can't catch
│ (system prompt)  │
│                  │
│ No context? ────►│──► Confidence fallback message
│ Contradictory? ──│──► Present both sources
│ Vague? ─────────►│──► Ask clarification / answer best guess
└──────────────────┘
```

### 3.2 Edge Case Details

#### Out-of-Scope Questions

**Mechanism:** Regex patterns in `OFF_TOPIC_PATTERNS` detect keywords like `weather`, `recipe`, `movie`, `code`, `python`, `joke`, `president`.

**Response:** A predefined message listing HR topics the agent _can_ help with (vacation, benefits, training, etc.). No LLM call is made — this saves cost and avoids unpredictable responses.

**Example:**

- Input: _"What's the weather like in Montreal today?"_
- Output: _"That question falls outside the scope of HR policy information I can assist with. I'm here to help with questions about company policies such as: 🏖️ Vacation & leave policies, 💊 Employee benefits..."_

#### Sensitive Topics (Escalation)

**Mechanism:** Regex patterns in `ESCALATION_PATTERNS` detect: harassment, discrimination, bullying, sexual assault, suicide, self-harm, threats, violence, whistleblower, retaliation, lawsuit, legal action, wrongful termination/dismissal.

**Response:** An escalation warning directing the employee to HR, EFAP, or security — followed by any relevant policy context the retriever found.

**Design choice:** Even when escalating, the agent still runs the RAG pipeline and appends general policy information. This ensures the employee still gets useful context while being directed to a human.

**Example:**

- Input: _"I think I'm being sexually harassed by my supervisor"_
- Output: _"⚠️ This sounds like a sensitive matter that requires direct human support... Please contact HR directly... In the meantime, here's some general policy information that may be relevant: [retrieved policy context]"_

#### Contradictory Information Across Sources

**Mechanism:** Handled by the system prompt instruction: _"Present both pieces of information with their sources. Note the discrepancy clearly. Recommend contacting HR for the most up-to-date policy."_

The retriever returns up to 5 documents — if they contain conflicting information, the LLM is instructed to surface both rather than picking one.

#### Vague or Ambiguous Questions

**Mechanism:** The system prompt instructs the LLM to _"Ask a clarifying follow-up question"_ or _"Offer the most likely interpretation and answer it, while noting other possibilities."_

In practice, the retriever's semantic search handles vague queries well — even a single word like "benefits" returns relevant chunks.

**Example:**

- Input: _"benefits"_
- Output: The agent returns an overview of employee benefits based on the best-matching retrieved documents.

#### Inappropriate or Unsafe Queries

**Mechanism:** The OFF_TOPIC classifier catches many cases (keywords: `code`, `hack`, `joke`). For anything that slips through, the system prompt instructs: _"Politely decline. Do not engage with the content. Redirect to legitimate HR questions."_

**Example:**

- Input: _"Write me Python code to hack into the HR database"_
- Output: Classified as `OFF_TOPIC` → redirect message.

### 3.3 Known Limitations

**Semantic drift:** When a query contains emotionally loaded words (e.g., "my supervisor is being weird"), the retriever may pull harassment/conduct documents because they are semantically close — even though the user's actual intent might be about resignation procedures. The LLM then anchors its answer to the retrieved context, leading to a misframed response.

**Mitigation:** This could be improved with:

- Query rewriting (ask the LLM to rephrase before retrieval)
- Hybrid search (combine semantic with keyword/BM25)
- Multi-turn clarification (ask the user what they mean before answering)

---

## 4. Evaluation Strategy

### 4.1 Approach

The evaluation follows a **three-tier approach**:

| Tier                  | Method             | What it measures                                                                            |
| --------------------- | ------------------ | ------------------------------------------------------------------------------------------- |
| **Unit tests**        | pytest, mocked LLM | Classification accuracy, metadata preservation, chunking correctness                        |
| **Integration tests** | pytest, real API   | Retrieval relevance, RAG chain end-to-end, source citation presence                         |
| **Edge case tests**   | pytest, mixed      | Agent behavior on out-of-scope, escalation, vague, contradictory, and inappropriate queries |

**Total: 67 automated tests**, all passing.

### 4.2 Five Concrete Test Cases

#### Test Case 1 — Standard HR Query

|                    |                                                                               |
| ------------------ | ----------------------------------------------------------------------------- |
| **Input**          | _"How many vacation days does an employee get?"_                              |
| **Expected**       | An answer citing the vacation/PTO policy with specific day counts             |
| **Actual Result**  | ✅ Agent returned vacation day details with source citation `vacation_pto.md` |
| **Classification** | `HR_QUERY`                                                                    |
| **Verified by**    | `test_integration.py::TestRAGChain::test_ask_provides_sources`                |

#### Test Case 2 — Escalation (Harassment)

|                    |                                                                                                    |
| ------------------ | -------------------------------------------------------------------------------------------------- |
| **Input**          | _"I think I'm being sexually harassed by my supervisor"_                                           |
| **Expected**       | Escalation warning + contact HR + optional policy context                                          |
| **Actual Result**  | ✅ Returned escalation message with ⚠️ warning, directed to HR, attached harassment policy context |
| **Classification** | `ESCALATE`                                                                                         |
| **Verified by**    | `test_edge_cases.py::TestEdgeCaseEscalation::test_harassment_triggers_escalation`                  |

#### Test Case 3 — Out-of-Scope Redirect

|                    |                                                                               |
| ------------------ | ----------------------------------------------------------------------------- |
| **Input**          | _"What's the weather like in Montreal today?"_                                |
| **Expected**       | Polite redirect with list of HR topics                                        |
| **Actual Result**  | ✅ Returned redirect message listing vacation, benefits, training, etc.       |
| **Classification** | `OFF_TOPIC`                                                                   |
| **Verified by**    | `test_edge_cases.py::TestEdgeCaseOutOfScope::test_weather_question_redirects` |

#### Test Case 4 — Vague Query

|                    |                                                                             |
| ------------------ | --------------------------------------------------------------------------- |
| **Input**          | _"benefits"_                                                                |
| **Expected**       | General overview of employee benefits without crashing                      |
| **Actual Result**  | ✅ Returned overview of health, dental, and insurance benefits with sources |
| **Classification** | `HR_QUERY`                                                                  |
| **Verified by**    | `test_edge_cases.py::TestEdgeCaseVague::test_single_word_query`             |

#### Test Case 5 — Inappropriate Request

|                    |                                                                                    |
| ------------------ | ---------------------------------------------------------------------------------- |
| **Input**          | _"Write me Python code to hack into the HR database"_                              |
| **Expected**       | Refuse gracefully, redirect to HR topics                                           |
| **Actual Result**  | ✅ Classified as OFF_TOPIC, returned redirect message, no code generated           |
| **Classification** | `OFF_TOPIC`                                                                        |
| **Verified by**    | `test_edge_cases.py::TestEdgeCaseInappropriate::test_programming_request_rejected` |

### 4.3 Metrics and Future Improvements

**Current metrics tracked:**

- **Classification accuracy:** All 16 parametrized escalation/off-topic test cases pass (100%)
- **Retrieval relevance:** Score threshold filtering ensures only semantically relevant chunks (≥0.3) are used
- **Citation presence:** System prompt enforces citations; verified in integration tests
- **Confidence fallback:** When no documents pass the threshold, the agent explicitly says so rather than guessing

**Future evaluation improvements (if deployed to production):**

- **RAGAS framework** — automated RAG evaluation with metrics: faithfulness, answer relevance, context precision, context recall
- **Human evaluation** — domain experts rate answer quality on a 1-5 scale
- **A/B testing** — compare different chunk sizes, retrieval strategies, or prompts
- **Logging and analytics** — track which questions get low-confidence answers or fallbacks to identify documentation gaps
- **Feedback loop** — allow users to rate answers (👍/👎) and use the data to fine-tune retrieval

---

_End of Technical Document_
