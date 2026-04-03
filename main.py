from __future__ import annotations

import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi import File, Form, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import text

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


def _require_env(name: str) -> str:
    val = (os.environ.get(name) or "").strip()
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def _validate_required_config() -> None:
    # Fail fast on missing critical config.
    _require_env("OPENAI_API_KEY")
    _require_env("PINECONE_API_KEY")
    _require_env("PINECONE_INDEX_NAME")
    _require_env("DATABASE_URL")


class IngestRequest(BaseModel):
    model_config = {"extra": "forbid", "populate_by_name": True}

    file_path: str = Field(description="Path to a local file (e.g. PDF) to ingest.")
    chunk_strategy: str = Field(
        default="recursive",
        description="Chunking strategy: 'recursive' (default), 'semantic', or 'section_aware'.",
    )

    layout_aware: bool = Field(
        default=False,
        description=(
            "If true, use layout-aware parsing (PyMuPDF4LLM + PyMuPDF-Layout) to produce "
            "structured elements and section metadata."
        ),
    )
    enable_ocr: bool = Field(
        default=False,
        validation_alias="ocr_enabled",
        description=(
            "If true, OCR may be applied during parsing. This is strict opt-in: OCR is never used unless enabled."
        ),
    )
    ocr_language: str = Field(
        default="eng",
        description=(
            "OCR language code (Tesseract-style, e.g. 'eng' or 'eng+deu'). Only used when enable_ocr=true."
        ),
    )


class IngestResponse(BaseModel):
    document_id: str
    status: str


class QueryRequest(BaseModel):
    model_config = {"extra": "forbid"}

    query: str
    document_ids: list[str] = Field(
        default_factory=list,
        description="Optional scope: only search within these document UUIDs.",
    )
    max_rounds: int = 2
    semantic_k: int = 10
    bm25_k: int = 10
    thread_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional LangGraph thread id. If supplied, the graph can checkpoint "
            "(MemorySaver by default) under this thread." 
        ),
    )

    debug: bool = Field(
        default=False,
        description="If true, include retrieved chunk debug info in the response.",
    )


class QueryResponse(BaseModel):
    answer: str
    is_grounded: bool = False
    citations: list[str] = Field(default_factory=list)
    rejected_claims: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)

    # Included only when QueryRequest.debug=true
    debug: dict | None = None


class DocumentInfo(BaseModel):
    document_id: str
    filename: str | None = None
    status: str
    doc_type: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]


class ChunksByIdsRequest(BaseModel):
    chunk_ids: list[str] = Field(min_length=1)


class ChunkSnippet(BaseModel):
    chunk_id: str
    document_id: str
    page_number: int | None = None
    element_type: str | None = None
    chunk_index: int
    section_title: str | None = None
    section_level: int | None = None
    layout_type: str | None = None
    content: str


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

    _validate_required_config()

    # Ensure DB tables exist (minimal bootstrap; for production prefer migrations).
    await _init_db()

    # Embeddings (Ollama) — initialized lazily on demand.
    app.state.services.embeddings = None
    app.state.services.vector_store = None
    app.state.services.retrieval_agent = None

    # LLM-backed agents (OpenAI).
    app.state.services.extraction_agent = ExtractionAgent()
    app.state.services.analysis_agent = AnalysisAgent()
    app.state.services.verifier_agent = VerifierAgent()

    yield


app = FastAPI(title="Agentic Doc Intelligence", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    """Basic health check for container orchestration.

    Returns 200 only if DB is reachable and core services are initialized.
    """
    services = app.state.services
    if getattr(services, "init_error", None):
        raise HTTPException(status_code=503, detail=services.init_error)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unreachable: {exc}")

    return {"status": "ok"}


async def _get_sql_store() -> SQLStore:
    # Minimal, local session for this request.
    session = AsyncSessionLocal()
    return SQLStore(session)


def _require_services(app: FastAPI, *, require_embeddings: bool) -> _AppServices:
    services: _AppServices = app.state.services
    if services.init_error:
        raise HTTPException(status_code=500, detail=services.init_error)
    if require_embeddings and services.embeddings is None:
        # Initialize embeddings on demand. Verify connectivity/model so failures are explicit.
        try:
            embeddings = get_embedding_model(verify=True)
            services.embeddings = embeddings
            services.vector_store = VectorStore(embeddings)
            services.retrieval_agent = RetrievalAgent(embeddings=embeddings)
            services.embeddings_error = None
        except Exception as exc:
            services.embeddings_error = str(exc)
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
                layout_aware=req.layout_aware,
                enable_ocr=req.enable_ocr,
                ocr_language=req.ocr_language,
            )
        finally:
            await sql_store.session.close()

    background_tasks.add_task(_run)

    return IngestResponse(document_id=document_id, status="scheduled")


def _safe_filename(name: str) -> str:
    # Avoid path traversal and weird separators.
    return Path(name).name or "upload.bin"


def _save_upload(upload: UploadFile) -> str:
    raw_dir = Path("data") / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = _safe_filename(upload.filename or "upload.bin")
    dest = raw_dir / f"{uuid.uuid4()}_{filename}"
    with dest.open("wb") as out:
        shutil.copyfileobj(upload.file, out)
    return str(dest)


@app.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunk_strategy: str = Form("recursive"),
    layout_aware: bool = Form(False),
    enable_ocr: bool = Form(False),
    ocr_language: str = Form("eng"),
) -> IngestResponse:
    services = _require_services(app, require_embeddings=True)
    if services.vector_store is None or services.extraction_agent is None:
        raise HTTPException(status_code=500, detail="Ingestion services are not initialized.")

    saved_path = _save_upload(file)

    sql_store = await _get_sql_store()
    try:
        doc = await sql_store.create_document(
            filename=Path(saved_path).name,
            file_path=saved_path,
            doc_metadata={"source": "upload", "original_filename": _safe_filename(file.filename or "")},
        )
        document_id = str(doc.id)
    finally:
        await sql_store.session.close()

    async def _run() -> None:
        sql_store = await _get_sql_store()
        try:
            await run_ingestion_job(
                file_path=saved_path,
                sql_store=sql_store,
                vector_store=services.vector_store,
                extraction_agent=services.extraction_agent,
                embeddings=services.embeddings,
                document_id=uuid.UUID(document_id),
                chunk_strategy=chunk_strategy,
                layout_aware=layout_aware,
                enable_ocr=enable_ocr,
                ocr_language=ocr_language,
            )
        finally:
            await sql_store.session.close()

    background_tasks.add_task(_run)
    return IngestResponse(document_id=document_id, status="scheduled")


@app.get("/documents", response_model=DocumentsResponse)
async def list_documents() -> DocumentsResponse:
    sql_store = await _get_sql_store()
    try:
        docs = await sql_store.list_documents()
        items = [
            DocumentInfo(
                document_id=str(d.id),
                filename=getattr(d, "filename", None),
                status=str(d.status.value if hasattr(d.status, "value") else d.status),
                doc_type=getattr(d, "doc_type", None),
                created_at=getattr(d, "created_at", None),
                updated_at=getattr(d, "updated_at", None),
            )
            for d in docs
        ]
        return DocumentsResponse(documents=items)
    finally:
        await sql_store.session.close()


@app.get("/documents/{document_id}", response_model=DocumentInfo)
async def get_document(document_id: str) -> DocumentInfo:
    sql_store = await _get_sql_store()
    try:
        doc = await sql_store.get_document(uuid.UUID(document_id))
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return DocumentInfo(
            document_id=str(doc.id),
            filename=getattr(doc, "filename", None),
            status=str(doc.status.value if hasattr(doc.status, "value") else doc.status),
            doc_type=getattr(doc, "doc_type", None),
            created_at=getattr(doc, "created_at", None),
            updated_at=getattr(doc, "updated_at", None),
        )
    finally:
        await sql_store.session.close()


@app.post("/chunks/by_ids", response_model=list[ChunkSnippet])
async def chunks_by_ids(req: ChunksByIdsRequest) -> list[ChunkSnippet]:
    sql_store = await _get_sql_store()
    try:
        chunks = await sql_store.get_chunks_by_ids(req.chunk_ids)
        out: list[ChunkSnippet] = []
        for c in chunks:
            out.append(
                ChunkSnippet(
                    chunk_id=str(c.pinecone_id or c.id),
                    document_id=str(c.document_id),
                    page_number=c.page_number,
                    element_type=c.element_type,
                    chunk_index=c.chunk_index,
                    section_title=getattr(c, "section_title", None),
                    section_level=getattr(c, "section_level", None),
                    layout_type=getattr(c, "layout_type", None),
                    content=c.content,
                )
            )
        return out
    finally:
        await sql_store.session.close()


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
            debug=req.debug,
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
