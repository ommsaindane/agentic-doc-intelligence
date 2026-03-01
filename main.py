from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agents.analysis_agent import AnalysisAgent
from app.agents.extraction_agent import ExtractionAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.verifier_agent import VerifierAgent
from app.embeddings.embedding import QwenEmbeddingError, get_embedding_model
from app.storage.schemas import Base
from app.storage.sql_store import AsyncSessionLocal, SQLStore, engine, ensure_schema
from app.storage.vector_store import VectorStore
from app.workflows.document_ingestion_graph import run_ingestion_job
from app.workflows.qa_graph import run_qa_graph


logger = logging.getLogger(__name__)


class IngestRequest(BaseModel):
    file_path: str = Field(description="Path to a local file (e.g. PDF) to ingest.")
    chunk_strategy: str = Field(
        default="recursive",
        description="Chunking strategy: 'recursive' (default) or 'semantic'.",
    )


class IngestResponse(BaseModel):
    document_id: str
    status: str


class QueryRequest(BaseModel):
    query: str
    document_ids: list[str] = Field(
        default_factory=list,
        description="Optional scope: only search within these document UUIDs.",
    )
    max_rounds: int = 2
    semantic_k: int = 10
    bm25_k: int = 10
    compress: bool = True
    thread_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional LangGraph thread id. If supplied, the graph can checkpoint "
            "(MemorySaver by default) under this thread." 
        ),
    )


class QueryResponse(BaseModel):
    answer: str
    is_grounded: bool = False
    citations: list[str] = Field(default_factory=list)
    rejected_claims: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)


class _AppServices:
    """Runtime services cached in `app.state` (not checkpoint-safe)."""

    def __init__(self):
        self.embeddings = None
        self.embeddings_error: Optional[str] = None
        self.vector_store: Optional[VectorStore] = None
        self.extraction_agent: Optional[ExtractionAgent] = None
        self.retrieval_agent: Optional[RetrievalAgent] = None
        self.analysis_agent: Optional[AnalysisAgent] = None
        self.verifier_agent: Optional[VerifierAgent] = None
        self.init_error: Optional[str] = None


async def _init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await ensure_schema()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    app.state.services = _AppServices()

    # Ensure DB tables exist (minimal bootstrap; for production prefer migrations).
    try:
        await _init_db()
    except Exception as exc:
        # Keep the app alive; endpoints will error with a clear message.
        msg = f"Database init failed: {exc}"
        app.state.services.init_error = msg
        logger.exception(msg)

    # Embeddings (Ollama) — used by Pinecone retrieval and optional semantic chunking.
    try:
        embeddings = get_embedding_model()
        app.state.services.embeddings = embeddings
        app.state.services.vector_store = VectorStore(embeddings)
        app.state.services.retrieval_agent = RetrievalAgent(embeddings=embeddings)
    except QwenEmbeddingError as exc:
        app.state.services.embeddings_error = str(exc)
        logger.warning("Embeddings unavailable: %s", exc)
    except Exception as exc:
        app.state.services.init_error = f"Service init failed (embeddings/pinecone): {exc}"
        logger.exception("Service init failed")

    # LLM-backed agents (OpenAI).
    try:
        app.state.services.extraction_agent = ExtractionAgent()
        app.state.services.analysis_agent = AnalysisAgent()
        app.state.services.verifier_agent = VerifierAgent()
    except Exception as exc:
        app.state.services.init_error = f"Service init failed (LLM agents): {exc}"
        logger.exception("LLM agent init failed")

    yield


app = FastAPI(title="Agentic Doc Intelligence", lifespan=lifespan)


async def _get_sql_store() -> SQLStore:
    # Minimal, local session for this request.
    session = AsyncSessionLocal()
    return SQLStore(session)


def _require_services(app: FastAPI, *, require_embeddings: bool) -> _AppServices:
    services: _AppServices = app.state.services
    if services.init_error:
        raise HTTPException(status_code=500, detail=services.init_error)
    if require_embeddings and services.embeddings is None:
        detail = services.embeddings_error or "Embeddings are not initialized."
        raise HTTPException(status_code=500, detail=detail)
    return services


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, background_tasks: BackgroundTasks) -> IngestResponse:
    services = _require_services(app, require_embeddings=True)
    if services.vector_store is None or services.extraction_agent is None:
        raise HTTPException(status_code=500, detail="Ingestion services are not initialized.")

    # Create the document row up-front so we can return a stable id immediately.
    sql_store = await _get_sql_store()
    try:
        doc = await sql_store.create_document(
            filename=Path(req.file_path).name,
            file_path=req.file_path,
            doc_metadata={"source": req.file_path},
        )
        document_id = str(doc.id)
    finally:
        await sql_store.session.close()

    # Schedule ingestion as a background async task.
    # Starlette BackgroundTasks supports async callables.
    async def _run() -> None:
        sql_store = await _get_sql_store()
        try:
            await run_ingestion_job(
                file_path=req.file_path,
                sql_store=sql_store,
                vector_store=services.vector_store,
                extraction_agent=services.extraction_agent,
                embeddings=services.embeddings,
                document_id=doc.id,
                chunk_strategy=req.chunk_strategy,
            )
        finally:
            await sql_store.session.close()

    background_tasks.add_task(_run)

    return IngestResponse(document_id=document_id, status="scheduled")


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    services = _require_services(app, require_embeddings=True)
    if (
        services.retrieval_agent is None
        or services.analysis_agent is None
        or services.verifier_agent is None
    ):
        raise HTTPException(status_code=500, detail="QA services are not initialized.")

    sql_store = await _get_sql_store()
    try:
        response = await run_qa_graph(
            query=req.query,
            sql_store=sql_store,
            retrieval_agent=services.retrieval_agent,
            analysis_agent=services.analysis_agent,
            verifier_agent=services.verifier_agent,
            document_ids=req.document_ids or None,
            max_rounds=req.max_rounds,
            semantic_k=req.semantic_k,
            bm25_k=req.bm25_k,
            compress=req.compress,
            thread_id=req.thread_id,
        )
    finally:
        await sql_store.session.close()

    return QueryResponse(**response)


def main() -> None:
    """CLI entrypoint."""
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    # When running `python main.py`, start the API server.
    # For dev, prefer: `uvicorn main:app --reload`.
    main()
