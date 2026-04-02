# Agentic Doc Intelligence — Repo Overview

This repository implements a FastAPI service that ingests local documents into **Postgres + Pinecone** and answers questions using an **agentic LangGraph workflow** (retrieve → analyze → verify).

## What’s implemented (high level)

- **Ingestion API (`POST /ingest`)**
  - Parses PDFs with PyMuPDF (deterministic text extraction; no OCR), chunks them (recursive or semantic), runs LLM-based **classification + extraction**, then stores:
    - full chunk text + metadata in **Postgres**
    - embeddings + minimal metadata filters in **Pinecone**
- **Query API (`POST /query`)**
  - Runs a LangGraph QA workflow with:
    - **hybrid retrieval** (Pinecone semantic + BM25 over Postgres chunks)
    - **analysis** (parallel sub-analyses → synthesis) and **verification** (grounding/citation enforcement)
    - optional re-retrieval rounds if verification fails grounding

---

## End-to-end architecture (ASCII)

```
                   ┌─────────────────────────────────────────┐
                   │               FastAPI API               │
                   │                (main.py)                │
                   └───────────────┬───────────────┬─────────┘
                                   │               │
                            POST /ingest       POST /query
                                   │               │
                          (BackgroundTasks)        │
                                   │               │
                                   ▼               ▼
                 ┌─────────────────────────┐   ┌─────────────────────────┐
                 │ Ingestion LangGraph     │   │ QA LangGraph             │
                 │ (document_ingestion_*)  │   │ (qa_graph.py)            │
                 └───────────┬─────────────┘   └───────────┬─────────────┘
                             │                               │
                             ▼                               ▼
                              PyMuPDF parse                      RetrievalAgent
                              (PDF text)                         - Pinecone semantic
                             │                          - BM25 from Postgres
                             ▼
                      chunk_documents                           │
                     - recursive splitter                        ▼
                     - semantic chunker                    AnalysisAgent
                             │                             - parallel subtasks
                             ▼                             - synthesis w/ citations
                      ExtractionAgent                             │
                     - classify DocType                           ▼
                     - extract entities                     VerifierAgent
                             │                             - grounding check
                             ▼                             - revised answer
            ┌───────────────────────────────────┐                │
            │ Storage                           │                ▼
            │ - Postgres (SQLStore)             │         QueryResponse
            │ - Pinecone (VectorStore)          │         (answer + grounding)
            └───────────────────────────────────┘
```

---

## Runtime services and configuration

### Required services

| Service | Used for | Where referenced | Key env vars |
|---|---|---|---|
| OpenAI | classification/extraction, analysis, verification | `app/agents/*` | `OPENAI_API_KEY`, `OPENAI_MODEL` (optional) |
| Postgres | store documents, chunks, agent run records; BM25 corpus source | `app/storage/sql_store.py`, `app/storage/schemas.py` | `DATABASE_URL` |
| Pinecone | vector search over chunk embeddings | `app/storage/vector_store.py`, `app/agents/retrieval_agent.py` | `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` |
| Ollama | embeddings (default: `qwen3-embedding:8b`) | `app/embeddings/embedding.py` | `OLLAMA_MODEL`, `OLLAMA_BASE_URL`/`OLLAMA_HOST` (optional) |

### Parsing

Ingestion supports **PDF files only** and parses using **PyMuPDF** (`pymupdf` / `fitz`) for deterministic text extraction (no OCR).

### Environment variables (single view)

| Env var | Required | Used by | Notes |
|---|---:|---|---|
| `OPENAI_API_KEY` | Yes | all LLM agents | required at agent init time |
| `OPENAI_MODEL` | No | all LLM agents | default: `gpt-4.1-mini` |
| `DATABASE_URL` | Yes | SQLAlchemy async engine | expects sync scheme `postgresql://…` and converts to `postgresql+asyncpg://…` internally |
| `PINECONE_API_KEY` | Yes | Pinecone client | used during `VectorStore` init |
| `PINECONE_INDEX_NAME` | Yes | Pinecone vector store | used by retrieval + upsert |
| `OLLAMA_MODEL` | No | embeddings factory | default: `qwen3-embedding:8b` |
| `OLLAMA_BASE_URL` / `OLLAMA_HOST` | No | embeddings factory | default: `http://127.0.0.1:11434` |

---

## Workflows (LangGraph)

### QA workflow (`app/workflows/qa_graph.py`)

**Intent:** Answer a question using retrieved evidence, then verify grounding. If not grounded, re-retrieve with broader `k` and try again (up to `max_rounds`).

**ASCII graph**

```
 START
   │
   ▼
receive_query
   │  (fan-out: Send one retrieve per document_id)
   ├───────────────┬───────────────┬───────────────┐
   ▼               ▼               ▼               ▼
retrieve         retrieve         retrieve       ...
   │               │               │
   └──────┬────────┴──────┬────────┴───────────────┘
          ▼               ▼
      merge_evidence  (dedupe by chunk_id)
          │
          ▼
        analyze      (AnalysisAgent)
          │
          ▼
        verify       (VerifierAgent)
          │
          ├── if NOT grounded AND rounds remain ───────┐
          │                                            │
          ▼                                            │
   format_response                                     │
          │                                            │
          ▼                                            │
         END   ◀───────────────────────────────────────┘
             (loop goes to receive_query; expands semantic_k/bm25_k)
```

**State shape (selected)**

- Inputs: `query`, optional `document_ids`
- Retrieval config:
  - `retrieval_round`, `max_rounds`
  - `semantic_k`, `bm25_k`
- Evidence:
  - `evidence_bundles`: parallel reducer list of `{document_id, docs[]}`
  - `merged_evidence`: deduped list of serialized Documents
- Agent outputs: `analysis` (serialized `AnalysisResult`), `verification` (serialized `VerificationResult`)
- Final: `response` (dict returned to FastAPI)

**Notable behavior**

- If `document_ids` is empty, it defaults to “all documents with status COMPLETE in Postgres”.
- On each round, `semantic_k` and `bm25_k` are expanded by `+6 * retrieval_round`.
- Evidence docs are deduped by `metadata.chunk_id` (or fallback key).

---

### Ingestion workflow (`app/workflows/document_ingestion_graph.py`)

**Intent:** Load a local file, parse elements, chunk, classify/extract, embed+upsert to Pinecone, then persist chunks + extraction metadata to Postgres.

**ASCII graph**

```
 START
   │
   ▼
 load
   │  - create or reuse document row
   │  - set status=PROCESSING
   │  - create agent_run(workflow="ingestion")
   ▼
 parse
   │  - load_document() via PyMuPDF
   ▼
 chunk
   │  - chunk_documents(strategy)
   │  - ExtractionAgent.run(chunks) → doc_type + extraction
   ▼
 embed
   │  - semantic strategy only: smoke-test embeddings
   ▼
 upsert_pinecone
   │  - create chunk UUIDs
   │  - upsert vectors to Pinecone (chunk UUID == Pinecone record id)
   ▼
 upsert_postgres
   │  - bulk insert chunks
   │  - update document: status=COMPLETE, doc_type, doc_metadata.extraction
   │  - agent_run SUCCESS
   ▼
 END
```

**Notable behavior**

- The FastAPI endpoint pre-creates the `documents` row so it can return a stable `document_id` immediately.
- `_chunk_to_row()` creates a **new UUID per chunk**, and uses that UUID both as the Postgres chunk primary key and the Pinecone record id.
- `embed_node` does a quick embedding smoke-test when semantic chunking is selected.

---

## Storage model (Postgres + Pinecone)

### Postgres tables (`app/storage/schemas.py`)

```
┌──────────────────┐        1-to-many        ┌──────────────────┐
│ documents         │────────────────────────►│ chunks            │
│ - id (UUID)       │                        │ - id (UUID)       │
│ - status          │                        │ - document_id FK  │
│ - filename/path   │                        │ - text/content    │
│ - doc_type        │                        │ - chunk_index     │
│ - metadata (JSON) │                        │ - page_number     │
└──────────────────┘                        │ - element_type    │
                                            │ - start_index     │
                                            │ - pinecone_id     │
                                            │ - metadata (JSON) │
                                            └──────────────────┘

┌──────────────────┐
│ agent_runs        │
│ - id (UUID)       │
│ - workflow        │  ("ingestion" or "qa")
│ - thread_id       │  (optional checkpoint thread)
│ - status          │
│ - document_id     │
│ - input_query     │
│ - output (JSON)   │
│ - error_message   │
└──────────────────┘
```

### Pinecone record schema (`app/storage/vector_store.py`)

- **ID:** chunk UUID (string)
- **Vector:** embedding produced by Ollama embeddings
- **Metadata (minimal filters):**
  - `chunk_id`, `document_id`, `element_type`, `chunk_index`, `page_number`, `doc_type`

---

## Per-file / per-module function

### Entry and ops

- `main.py`
  - Purpose: FastAPI app, lifecycle initialization, request models, endpoints.
  - Public API:
    - `app = FastAPI(...)`
    - `POST /ingest` schedules `run_ingestion_job(...)` in background
    - `POST /query` calls `run_qa_graph(...)` and returns `QueryResponse`
    - `main()` runs uvicorn (non-reload)
  - Key behavior:
    - DB init errors are captured into `app.state.services.init_error` (app stays up but endpoints return 500).
    - Embeddings init may fail at startup; `_require_services(..., require_embeddings=True)` attempts one-time lazy re-init.

- `commands.txt`
  - Purpose: quick run command + example curl calls for ingestion/query.

- `README.md`
  - Purpose: setup/run documentation; expected `.env` variables; high-level repo structure.

- `pyproject.toml` / `uv.lock` / `.python-version`
  - Purpose: dependency + interpreter + lockfile for reproducible installs.

### Workflows

- `app/workflows/qa_graph.py`
  - Purpose: user-facing QA orchestration using `StateGraph`.
  - Main functions:
    - `build_qa_graph(...)` → compiled graph
    - `run_qa_graph(...)` → convenience executor returning response dict
  - Key helpers:
    - `_bm25_corpus_for_doc(...)`: loads all chunks for a doc and builds BM25 corpus
    - `_dedupe_docs_by_chunk_id(...)`: avoids duplicated evidence
  - Config inputs: `max_rounds`, `semantic_k`, `bm25_k`, optional `thread_id`.

- `app/workflows/document_ingestion_graph.py`
  - Purpose: ingestion pipeline orchestration.
  - Main functions:
    - `build_document_ingestion_graph(...)` → compiled ingestion graph
    - `run_ingestion_job(...)` → convenience executor
  - Key helpers:
    - `_chunk_to_row(...)`: converts chunk Documents to DB row dicts (creates UUIDs)
    - `error_handler(...)`: best-effort document status and agent_run failure recording

### Agents

- `app/agents/extraction_agent.py`
  - Purpose: classify document type + extract structured facts/entities from “head” chunks.
  - Public API: `ExtractionAgent.classify(...)`, `ExtractionAgent.extract(...)`, `ExtractionAgent.run(...)`.
  - Key types: `DocType`, `ClassificationResult`, `ExtractionResult`, `ExtractedEntity`.
  - Key constants: `HEAD_CHUNKS = 10`.
  - Env vars: `OPENAI_API_KEY`, `OPENAI_MODEL`.

- `app/agents/analysis_agent.py`
  - Purpose: evidence-grounded reasoning; parallel sub-analyses (summary, claims, contradictions) then synthesis.
  - Public API: `AnalysisAgent.run(question, evidence_docs)` → `AnalysisResult`.
  - Key types: `AnalysisResult`, `Citation`, `KeyClaim`, `Contradiction`.
  - Env vars: `OPENAI_API_KEY`, `OPENAI_MODEL`.

- `app/agents/retrieval_agent.py`
  - Purpose: hybrid retrieval (Pinecone semantic + BM25 keyword) with deterministic merge.
  - Public API:
    - `hybrid_retrieve(...)` → candidate Documents
    - `retrieve(...)` → final evidence Documents
  - Key types: `HybridWeights`.
  - Env vars: `OPENAI_API_KEY`, `OPENAI_MODEL`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`.

- `app/agents/verifier_agent.py`
  - Purpose: grounding verification and citation enforcement; produces revised answer without unsupported claims.
  - Public API:
    - `VerifierAgent.verify(...)` → `VerificationResult`
    - `VerifierAgent.verify_analysis_result(...)`
    - `VerifierAgent.filter_supported_documents(...)`
  - Env vars: `OPENAI_API_KEY`, `OPENAI_MODEL`.

### Embeddings

- `app/embeddings/embedding.py`
  - Purpose: construct `OllamaEmbeddings` and fail fast if Ollama/model is unreachable.
  - Public API: `get_embedding_model(model=None, base_url=None)`.
  - Key types: `QwenEmbeddingError`.
  - Env vars: `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_HOST`.

- `app/embeddings/__init__.py`
  - Purpose: re-export embedding factory + error type.

### Ingestion utilities

- `app/ingestion/ingestion.py`
  - Purpose: load local PDFs using PyMuPDF into page-level Documents.
  - Public API: `load_document(file_path)` (generator), `load_directory(dir_path, ...)`.
  - Validation: `_validate_path` ensures exists, file, supported extension.

- `app/ingestion/chunking.py`
  - Purpose: chunking strategies with `start_index` metadata.
  - Public API:
    - `chunk_recursive(...)`
    - `chunk_semantic(...)`
    - `chunk_documents(..., strategy=...)`

### Storage

- `app/storage/schemas.py`
  - Purpose: ORM models + Pydantic DTOs.
  - ORM: `DocumentModel`, `ChunkModel`, `AgentRunModel`.
  - Enums: `DocumentStatus`, `AgentRunStatus`.
  - DTOs: `DocumentCreate`, `DocumentOut`, `ChunkOut`, `AgentRunOut`.

- `app/storage/sql_store.py`
  - Purpose: async SQLAlchemy engine/session and a `SQLStore` service encapsulating DB operations.
  - Key functions:
    - `ensure_schema()` idempotent shim for legacy tables/enum types
  - `SQLStore` methods:
    - documents: create/get/list/update/delete
    - chunks: bulk insert; fetch chunks for BM25 corpus
    - agent_runs: create/complete/fail
  - Env vars: `DATABASE_URL`.

- `app/storage/vector_store.py`
  - Purpose: encapsulate Pinecone operations and metadata conventions.
  - Public API:
    - `upsert_chunks(chunks, doc_type)`
    - `as_retriever(...)`
    - `similarity_search_with_score(...)`
    - `delete_document_vectors(document_id)`
  - Env vars: `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`.

---

## Tech stack / libraries used (from `pyproject.toml`)

### API & runtime

- **FastAPI**: HTTP API framework
- **Uvicorn**: ASGI server
- **Pydantic**: request/response models and typed outputs
- **python-dotenv**: `.env` configuration loading

### Agent/workflow orchestration

- **LangGraph**: graph-based workflow orchestration (`StateGraph`, `Send`, `Command`)
- **LangChain (core + full)**: LLM chains, prompts, runnables, document abstractions
- **langchain-openai** + **openai**: OpenAI model integration

### Retrieval & vector

- **Pinecone** + **langchain-pinecone**: vector store and semantic search
- **langchain-community**: BM25 retriever

### Embeddings

- **langchain-ollama**: Ollama embeddings client

### Document parsing

- **PyMuPDF** (`pymupdf` / `fitz`): deterministic PDF text extraction (no OCR)

### Storage

- **SQLAlchemy**: ORM and async DB access
- **asyncpg**: async Postgres driver
- **psycopg2-binary**: sync Postgres driver (declared; app uses asyncpg)

### Present but not exercised by default

- **langchain-elasticsearch**: potential future retrieval backend
- **langgraph-checkpoint-postgres**: Postgres checkpointing option (code currently defaults to `MemorySaver`)
- **langchain-experimental**: semantic chunker implementation
- **mcp**: Model Context Protocol library (not used in current code paths)

---

## Notable behaviors / trade-offs

- Startup is resilient: DB init errors keep the server alive but endpoints return 500 with a clear message.
- Evidence grounding is explicit: verifier produces a revised answer that removes unsupported claims.

## Repo contents (non-code)

- `data/raw/`, `data/processed/`: example data directories.
- `.gitignore`, `.git/`: git metadata and ignore rules.
- `__pycache__/`, `.venv/`: local runtime artifacts.
