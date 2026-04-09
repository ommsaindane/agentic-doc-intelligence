# Agentic Doc Intelligence

FastAPI service for ingesting PDFs into **Postgres + Pinecone** and answering questions using a **LangGraph** QA workflow (retrieve → analyze → verify).

The system has two primary flows:

1. **Ingestion**: PDF → parse → chunk → classify/extract → store chunks in Postgres → embed/upsert vectors to Pinecone
2. **QA**: query → hybrid retrieval (Pinecone semantic + BM25 keyword) → reranking (cross-encoder) → analysis → verification (grounding enforcement) → answer with citations

---

## What you can do

- Ingest a PDF
- Track document status
- Ask questions regarding the ingested pdf
- Resolve citations into source snippets

---

<img width="1853" height="963" alt="Image" src="https://github.com/user-attachments/assets/4971c5aa-3fa5-483d-a5b2-3147e211f6f9" />

---

## Requirements

You need the following services/config to run the full pipeline end-to-end:

- **Python**: 3.11+
- **Postgres**: used for documents/chunks/agent runs and BM25 corpus
- **Pinecone**: used for semantic vector search
- **OpenAI API key**: gpt-5-mini used by extraction/analysis/verification agents
- **Ollama** : Qwen3-Embedding-8B used.

Notes:

- `GET /health` checks DB reachability and core agent init; it does **not** require Ollama.
- `POST /ingest`, `POST /ingest/upload`, and `POST /query` require embeddings and will fail with a clear error if Ollama is unreachable.

---

## Configuration (.env)

Create a `.env` file in the repo root. These are required:

```env
OPENAI_API_KEY="..."
PINECONE_API_KEY="..."
PINECONE_INDEX_NAME="..."
DATABASE_URL="postgresql://user:pass@host:5432/dbname"
ENABLE_RERANKER=true
```

---

## Install

Create/activate a virtualenv, then install dependencies:

```bash
pip install -e .
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
streamlit run streamlit_app.py
```

In the sidebar, set **API base URL** to:

- Local dev: `http://127.0.0.1:8000`
- Docker Compose (default port mapping): `http://127.0.0.1:8000`

---

## How ingestion works

- PDFs are parsed using **PyMuPDF**.
- If `layout_aware=true`, parsing uses **PyMuPDF4LLM** (and optionally PyMuPDF-Layout) to emit smaller layout elements with metadata like `layout_type` and `section_title`.
- OCR is **opt-in**: it is never used unless `enable_ocr=true`.
- Chunks are stored in Postgres and embedded/upserted to Pinecone; QA uses Pinecone + BM25 over Postgres chunks.

---
