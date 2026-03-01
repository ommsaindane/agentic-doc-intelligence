# Agentic Doc Intelligence

FastAPI service for ingesting documents into Postgres + Pinecone and answering questions with an agentic LangGraph workflow (retrieve → analyze → verify).

## What it does

- **Ingest** local documents (PDF/images/Office/etc.) with Unstructured (hi-res layout + OCR), chunk them (recursive or semantic), classify/extract structured facts, then store:
	- chunks + metadata in **Postgres**
	- embeddings + filters in **Pinecone**
- **Query** using **hybrid retrieval** (Pinecone semantic + BM25 over Postgres chunks), then run LLM agents to produce an answer with grounding checks.

## Prerequisites

- Python **3.11+**
- A running **Postgres** instance (used via `DATABASE_URL`)
- A **Pinecone** index (set `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`)
- **OpenAI API key** (set `OPENAI_API_KEY`)
- **Ollama** for embeddings (defaults to `qwen3-embedding:8b` at `http://localhost:11434`)
	- `ollama serve`
	- `ollama pull qwen3-embedding:8b`

Notes on parsing/OCR:
- Ingestion uses `langchain-unstructured` with `strategy="hi_res"`. For PDFs/images this commonly requires system dependencies (e.g., **Tesseract OCR**). Install what Unstructured needs for your OS and ensure those binaries are discoverable on `PATH`.

## Setup

### Option A (recommended): uv

This repo includes `uv.lock`, so `uv` is the intended workflow.

```bash
uv sync
```

### Option B: venv + pip

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate

pip install -U pip
pip install -e .
```

## Configuration (.env)

Several modules call `python-dotenv`’s `load_dotenv()`, so a local `.env` file is the simplest way to configure the service.

Create `.env` in the repo root:

```dotenv
# OpenAI
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini

# Pinecone
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=your_index_name

# Postgres (NOTE: use the *sync* scheme; the app converts to asyncpg internally)
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
```

## Run

Development (recommended):

```bash
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Production-ish:

```bash
python main.py
```

API will be available at:
- `http://localhost:8000/docs` (Swagger UI)

## API

### `POST /ingest`

Schedules an ingestion job in the background and returns a stable `document_id` immediately.

Request body:

```json
{
	"file_path": "C:/path/to/file.pdf",
	"chunk_strategy": "recursive"
}
```

`chunk_strategy`:
- `recursive` (default; fast, no embeddings required)
- `semantic` (requires embeddings; uses Ollama)

Response:

```json
{ "document_id": "...", "status": "scheduled" }
```

### `POST /query`

Runs the QA graph (retrieve → analyze → verify; with optional re-retrieval rounds).

Request body:

```json
{
	"query": "What is the contract termination notice period?",
	"document_ids": [],
	"max_rounds": 2,
	"semantic_k": 10,
	"bm25_k": 10,
	"compress": true,
	"thread_id": null
}
```

Response shape:
- `answer`: final answer (may be revised by verifier)
- `is_grounded`: whether the verifier found sufficient evidence
- `citations`: list of citation identifiers/strings
- `rejected_claims`, `missing_evidence`: grounding diagnostics

## Repo structure (high level)

- `main.py`: FastAPI app + endpoints
- `app/agents/`: LLM-backed agents (extraction / analysis / retrieval / verifier)
- `app/ingestion/`: loading + chunking
- `app/storage/`: Postgres models + SQL store + Pinecone wrapper
- `app/workflows/`: LangGraph workflows for ingestion and QA

## Security notes

- Do **not** commit `.env` (it’s gitignored). If secrets were ever committed, rotate them immediately.
