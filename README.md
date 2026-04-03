# Agentic Doc Intelligence

FastAPI service for ingesting PDFs into **Postgres + Pinecone** and answering questions using a **LangGraph** QA workflow (retrieve → analyze → verify).

This repo is built to be **strict and deterministic**:

- LLM calls run with `temperature=0`.
- Extraction / analysis / verification use **typed, structured outputs**.
- The API **fails fast** if required configuration is missing.
- No silent fallbacks (no OCR unless explicitly enabled).

---

## What you can do

- Ingest a PDF (recommended): upload bytes via `POST /ingest/upload`.
- Ingest a PDF (dev-only): point the server at a local file path via `POST /ingest`.
- Track document status via `GET /documents` and `GET /documents/{document_id}`.
- Ask questions via `POST /query` (optionally with `debug=true`).
- Resolve citations into source snippets via `POST /chunks/by_ids`.
- Use the minimal UI in `streamlit_app.py` (upload → status → query → citations/snippets).

---

## Requirements

You need the following services/config to run the full pipeline end-to-end:

- **Python**: 3.11+
- **Postgres**: used for documents/chunks/agent runs and BM25 corpus
- **Pinecone**: used for semantic vector search
- **OpenAI API key**: used by extraction/analysis/verification agents
- **Ollama** (optional at startup, required for ingest/query): used for embeddings

Notes:

- `GET /health` only checks DB reachability and core agent init; it does **not** require Ollama.
- `POST /ingest`, `POST /ingest/upload`, and `POST /query` require embeddings and will fail with a clear error if Ollama is unreachable.

---

## Configuration (.env)

Create a `.env` file in the repo root. These are required:

```env
OPENAI_API_KEY="..."
PINECONE_API_KEY="..."
PINECONE_INDEX_NAME="agentic-doc-intel"
DATABASE_URL="postgresql://user:pass@host:5432/dbname"
```

Optional:

```env
# OpenAI chat model used by extraction/analysis/verification agents.
# Default: gpt-4.1-mini
OPENAI_MODEL=gpt-5-mini

# Ollama embeddings
# Defaults: OLLAMA_MODEL=qwen3-embedding:8b, OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3-embedding:8b
OLLAMA_BASE_URL=http://127.0.0.1:11434

# Optional features
ENABLE_RERANKER=false
ENABLE_OBSERVABILITY=false
```

Important:

- `DATABASE_URL` must use the **sync** scheme (`postgresql://...`). The app converts it internally to `postgresql+asyncpg://...`.
- Pinecone index creation is not automated here; the index named by `PINECONE_INDEX_NAME` must exist.

---

## Install

Create/activate a virtualenv, then install dependencies:

```bash
pip install -e .
```

Optional extras:

- Streamlit UI:

```bash
pip install -e ".[ui]"
```

- Cross-encoder reranker (only if `ENABLE_RERANKER=true`):

```bash
pip install -e ".[reranker]"
```

---

## Run the API (local dev)

Start Postgres however you like (Docker is easiest), then:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Smoke test:

```bash
curl http://127.0.0.1:8000/health
```

---

## Run with Docker Compose

This starts **Postgres + API**.

```bash
docker compose up --build
```

By default, the compose file sets:

- `DATABASE_URL` to point at the `postgres` service
- `OLLAMA_BASE_URL` to `http://host.docker.internal:11434`

So on Windows/macOS you can run Ollama on your host machine and the container can reach it.

If you are running Ollama somewhere else, set `OLLAMA_BASE_URL` in your environment before `docker compose up`.

---

## Run the Streamlit UI

The UI runs on your host machine and calls the API over HTTP.

1) Ensure the API is running (either local Uvicorn **or** Docker Compose).
2) Install the UI extra and start Streamlit:

```bash
pip install -e ".[ui]"
streamlit run streamlit_app.py
```

In the sidebar, set **API base URL** to:

- Local dev: `http://127.0.0.1:8000`
- Docker Compose (default port mapping): `http://127.0.0.1:8000`

---

## API Endpoints

### Health

```bash
curl http://127.0.0.1:8000/health
```

### Ingest (upload bytes) — recommended

```bash
curl -X POST http://127.0.0.1:8000/ingest/upload \
	-F "file=@data/raw/your.pdf" \
	-F "chunk_strategy=recursive" \
	-F "layout_aware=false" \
	-F "enable_ocr=false" \
	-F "ocr_language=eng"
```

Response:

```json
{ "document_id": "<uuid>", "status": "scheduled" }
```

### Ingest (server-local path) — dev-only

This requires the file to exist on the API server’s filesystem.

```bash
curl -X POST http://127.0.0.1:8000/ingest \
	-H "Content-Type: application/json" \
	-d '{
		"file_path": "data/raw/your.pdf",
		"chunk_strategy": "recursive",
		"layout_aware": false,
		"enable_ocr": false,
		"ocr_language": "eng"
	}'
```

### List documents / status

```bash
curl http://127.0.0.1:8000/documents
curl http://127.0.0.1:8000/documents/<document_id>
```

### Query

```bash
curl -X POST http://127.0.0.1:8000/query \
	-H "Content-Type: application/json" \
	-d '{
		"query": "What are the key obligations?",
		"document_ids": [],
		"max_rounds": 2,
		"semantic_k": 10,
		"bm25_k": 10,
		"debug": false
	}'
```

If `debug=true`, the response includes a `debug` object with retrieved chunk metadata. If reranking is enabled, those debug rows include `_rerank_score`.

### Resolve citations to snippets

`/query` returns `citations` as `chunk_id` strings. Use these to fetch the underlying chunk text.

```bash
curl -X POST http://127.0.0.1:8000/chunks/by_ids \
	-H "Content-Type: application/json" \
	-d '{"chunk_ids": ["<chunk_id>", "<chunk_id>"]}'
```

---

## How ingestion works (high level)

- PDFs are parsed deterministically using **PyMuPDF**.
- If `layout_aware=true`, parsing uses **PyMuPDF4LLM** (and optionally PyMuPDF-Layout) to emit smaller layout elements with metadata like `layout_type` and `section_title`.
- OCR is **strict opt-in**: it is never used unless `enable_ocr=true`.
- Chunks are stored in Postgres and embedded/upserted to Pinecone; QA uses Pinecone + BM25 over Postgres chunks.

---

## Repository map

- `main.py`: FastAPI app + endpoints + lifecycle config validation
- `app/workflows/`: LangGraph graphs for ingestion + QA
- `app/agents/`: ExtractionAgent, RetrievalAgent, AnalysisAgent, VerifierAgent
- `app/storage/`: SQLAlchemy models/store and Pinecone vector store wrapper
- `app/ingestion/`: PDF loading + chunking utilities
- `streamlit_app.py`: minimal UI (uploads via API)
- `docker-compose.yml` / `Dockerfile`: containerized API + Postgres

For a deeper walkthrough of modules and data flow, see `REPO_OVERVIEW.md`.

---

## Troubleshooting

- `RuntimeError: Missing required environment variable ...`
	- Ensure `.env` contains `OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, and `DATABASE_URL`.

- `/health` returns 503
	- The DB is unreachable. Check `DATABASE_URL` and that Postgres is running.

- `/query` or `/ingest/upload` returns an error about Ollama
	- Start Ollama and pull the embedding model (default is `qwen3-embedding:8b`), or set `OLLAMA_BASE_URL`/`OLLAMA_MODEL`.

- Reranker enabled errors
	- If `ENABLE_RERANKER=true`, install optional extras: `pip install -e ".[reranker]"`.
