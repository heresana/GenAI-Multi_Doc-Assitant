# GenAI Multi-Document Assistant

A production-grade Retrieval-Augmented Generation (RAG) system that enables intelligent, citation-aware question answering over multiple PDF documents — powered by Google Gemini 2.5 Flash and ChromaDB.


## Objective

Most LLMs hallucinate when asked about specific documents. This project solves that by grounding every answer in retrieved document content — so responses are accurate, traceable, and cited by source and page number.


## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Streamlit UI                         │
│   ┌──────────────┐              ┌──────────────────────┐    │
│   │  Sidebar     │              │   Chat Interface     │    │
│   │  PDF Upload  │              │   Q&A + Citations    │    │
│   │  Doc Mgmt    │              │   Conversation Hist  │    │
│   └──────┬───────┘              └──────────┬───────────┘    │
└──────────┼───────────────────────────────── ┼───────────────┘
           │                                  │
           ▼                                  ▼
┌─────────────────────┐          ┌────────────────────────┐
│   Preprocessing     │          │    Context Manager     │
│   pypdf extraction  │          │    build_prompt()      │
│   UTF-8 normalize   │          │    build_context()     │
│   Sliding-window    │          │    Citation assembly   │
│   chunking (800w)   │          │    History injection   │
└────────┬────────────┘          └────────────┬───────────┘
         │                                    │
         ▼                                    ▼
┌─────────────────────┐          ┌────────────────────────┐
│   ChromaDB          │◄─────────│   SentenceTransformer  │
│   Vector Store      │  embed   │   all-MiniLM-L6-v2     │
│   cosine similarity │  query   │   384-dim embeddings   │
│   doc_id filtering  │          └────────────────────────┘
└─────────────────────┘
         │ top-k hits
         ▼
┌─────────────────────┐
│   Gemini 2.5 Flash  │
│   Grounded response │
│   + source citations│
└─────────────────────┘
```



## Key Features

- **Multi-document support** — index and query multiple PDFs simultaneously, with cross-document synthesis
- **Sliding-window chunking** — 150-word overlap prevents context loss at chunk boundaries
- **Citation-aware prompting** — every chunk is tagged `[Source: file.pdf | Page N]` before LLM injection
- **Conversation memory** — last 6 turns are injected into each prompt for follow-up reasoning
- **Hallucination mitigation** — model is explicitly instructed to acknowledge gaps rather than fabricate
- **Document lifecycle** — add or remove documents at runtime without full re-indexing
- **Chat export** — download the conversation history as a `.txt` file for offline reference or record-keeping
---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Google Gemini 2.5 Flash |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (in-memory, cosine similarity) |
| PDF Parsing | pypdf |
| Orchestration | Python 3.9+ |
---

## Future Roadmap

**Hybrid Search** — Combine dense vector retrieval with BM25 sparse retrieval (reciprocal rank fusion) to improve recall on keyword-heavy queries and named entities.

**Agentic Tool Use** — Extend the system with tool-calling so the model can request additional chunks, compare specific pages, or trigger document-wide computations autonomously.

**Evaluation Framework** — Integrate RAGAS metrics (faithfulness, answer relevancy, context recall) to automatically evaluate retrieval and generation quality across a benchmark question set.
