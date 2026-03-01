"""app.workflows.document_ingestion_graph

LangGraph workflow for document ingestion.

Nodes (sequential):
	load → OCR/parse → chunk → embed → upsert to Pinecone → upsert to Postgres

Notes
-----
- OCR/parse is handled by `UnstructuredLoader(strategy="hi_res")` in
  `app.ingestion.ingestion.load_document`.
- Chunking uses `app.ingestion.chunking.chunk_documents`.
- Classification + structured extraction uses `app.agents.extraction_agent.ExtractionAgent`.
- Pinecone upsert goes through `app.storage.vector_store.VectorStore`.
- Postgres upsert goes through `app.storage.sql_store.SQLStore`.

This graph is designed to be run as a background job (e.g., triggered by a
FastAPI `/ingest` endpoint using `BackgroundTasks`).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langgraph.graph import END, START, StateGraph

from app.agents.extraction_agent import DocType, ExtractionAgent, ExtractionResult
from app.ingestion.chunking import chunk_documents
from app.ingestion.ingestion import load_document
from app.storage.schemas import ChunkModel, DocumentStatus
from app.storage.sql_store import SQLStore
from app.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestionState(TypedDict, total=False):
	"""State passed between ingestion nodes."""

	# Inputs
	file_path: str
	chunk_strategy: str  # "recursive" | "semantic"

	# Runtime services (injected via closure; kept here for debugging only)
	# NOTE: not checkpoint-safe; intended for in-process background execution.

	# Document + agent run tracking
	document_id: uuid.UUID
	agent_run_id: uuid.UUID

	# Parsed docs + chunks
	elements: list[Document]
	chunks: list[Document]

	# Extraction
	doc_type: str
	extraction: dict

	# Prepared storage payloads
	chunk_rows: list[dict]
	pinecone_ids: list[str]

	# Error
	error: str


@dataclass(frozen=True)
class IngestionServices:
	sql: SQLStore
	vector: VectorStore
	extractor: ExtractionAgent
	embeddings: Optional[Embeddings] = None


def _chunk_to_row(chunk: Document, *, document_id: uuid.UUID, chunk_index: int) -> dict:
	"""Convert a LangChain Document chunk into a DB row dict."""
	chunk_id = uuid.uuid4()
	metadata = chunk.metadata or {}

	# Unstructured commonly uses `page_number` and `category`.
	page_number = metadata.get("page_number")
	element_type = metadata.get("category") or metadata.get("element_type")

	# `add_start_index=True` adds this to metadata.
	start_index = metadata.get("start_index")

	return {
		"id": chunk_id,
		"document_id": document_id,
		"content": chunk.page_content,
		"chunk_index": chunk_index,
		"start_index": start_index,
		"page_number": page_number,
		"element_type": element_type,
		"pinecone_id": str(chunk_id),
		"chunk_metadata": metadata,
	}


def build_document_ingestion_graph(
	*,
	sql_store: SQLStore,
	vector_store: VectorStore,
	extraction_agent: ExtractionAgent,
	embeddings: Optional[Embeddings] = None,
):
	"""Build and compile the ingestion StateGraph with injected services."""

	services = IngestionServices(
		sql=sql_store,
		vector=vector_store,
		extractor=extraction_agent,
		embeddings=embeddings,
	)

	async def load_node(state: IngestionState) -> dict[str, Any]:
		file_path = state["file_path"]

		# Allow callers (e.g., FastAPI) to pre-create the document row so they
		# can immediately return an id to the client.
		precreated_id = state.get("document_id")
		if precreated_id is not None:
			doc = await services.sql.get_document(precreated_id)
			if doc is None:
				raise ValueError(f"Document not found: {precreated_id}")
			await services.sql.update_document_status(precreated_id, DocumentStatus.PROCESSING)
		else:
			doc = await services.sql.create_document(
				filename=Path(file_path).name,
				file_path=file_path,
				doc_metadata={"source": file_path},
			)
			await services.sql.update_document_status(doc.id, DocumentStatus.PROCESSING)

		run = await services.sql.create_agent_run(
			workflow="ingestion",
			document_id=doc.id,
		)

		logger.info("Ingestion started for %s (document_id=%s)", file_path, doc.id)
		return {"document_id": doc.id, "agent_run_id": run.id}

	async def parse_node(state: IngestionState) -> dict[str, Any]:
		file_path = state["file_path"]
		elements = list(load_document(file_path))
		logger.info("Parsed %d elements from %s", len(elements), file_path)
		return {"elements": elements}

	async def chunk_node(state: IngestionState) -> dict[str, Any]:
		strategy = state.get("chunk_strategy", "recursive")
		elements = state.get("elements", [])
		if not elements:
			return {"chunks": [], "doc_type": DocType.OTHER.value, "extraction": {}}

		chunks = chunk_documents(
			elements,
			strategy=strategy,
			embeddings=services.embeddings,
		)
		logger.info("Chunked %d elements → %d chunks (strategy=%s)", len(elements), len(chunks), strategy)

		# Run classification + structured extraction on chunks so we can
		# enrich Postgres metadata and Pinecone filtering fields.
		extraction: ExtractionResult = await services.extractor.run(chunks)
		return {
			"chunks": chunks,
			"doc_type": extraction.doc_type.value,
			"extraction": extraction.model_dump(),
		}

	async def embed_node(state: IngestionState) -> dict[str, Any]:
		"""Smoke-test embeddings to fail fast before upserting."""
		strategy = state.get("chunk_strategy", "recursive")
		if strategy != "semantic":
			return {}

		if services.embeddings is None:
			raise ValueError("Semantic chunking requested but no embeddings were provided.")

		chunks = state.get("chunks", [])
		if not chunks:
			return {}

		# Cheap smoke-test; do not store vectors in state.
		services.embeddings.embed_query(chunks[0].page_content[:200] or "ping")
		return {}

	async def upsert_pinecone_node(state: IngestionState) -> dict[str, Any]:
		document_id = state["document_id"]
		chunks = state.get("chunks", [])
		doc_type = state.get("doc_type") or DocType.OTHER.value

		rows = [_chunk_to_row(c, document_id=document_id, chunk_index=i) for i, c in enumerate(chunks)]

		# Build transient ChunkModel objects so VectorStore can treat them
		# as if they were loaded from Postgres.
		chunk_models = [ChunkModel(**row) for row in rows]
		pinecone_ids = await services.vector.upsert_chunks(chunk_models, doc_type=doc_type)

		return {"chunk_rows": rows, "pinecone_ids": pinecone_ids}

	async def upsert_postgres_node(state: IngestionState) -> dict[str, Any]:
		document_id = state["document_id"]
		doc_type = state.get("doc_type")
		rows = state.get("chunk_rows", [])
		extraction = state.get("extraction")

		if rows:
			await services.sql.save_chunks(rows)

		# Persist classification + extraction summary into the document row.
		# Keep it compact: the full extracted structure can also be stored
		# in agent_runs.output if you prefer.
		doc_metadata: dict = {"source": state["file_path"]}
		if extraction:
			doc_metadata["extraction"] = extraction

		await services.sql.update_document_status(
			document_id=document_id,
			status=DocumentStatus.COMPLETE,
			doc_type=doc_type,
			doc_metadata=doc_metadata,
		)

		await services.sql.complete_agent_run(
			run_id=state["agent_run_id"],
			output={
				"document_id": str(document_id),
				"doc_type": doc_type,
				"chunks": len(rows),
				"pinecone_ids": state.get("pinecone_ids", []),
			},
			total_tokens=0,
		)
		logger.info("Ingestion complete for document_id=%s (chunks=%d)", document_id, len(rows))
		return {}

	async def error_handler(state: IngestionState, exc: Exception) -> dict[str, Any]:
		"""Best-effort failure handling."""
		logger.exception("Ingestion failed: %s", exc)
		try:
			if "document_id" in state:
				await services.sql.update_document_status(
					document_id=state["document_id"],
					status=DocumentStatus.FAILED,
				)
			if "agent_run_id" in state:
				await services.sql.fail_agent_run(state["agent_run_id"], str(exc))
		except Exception:
			logger.exception("Failed to record ingestion failure")
		return {"error": str(exc)}

	graph = StateGraph(IngestionState)
	graph.add_node("load", load_node)
	graph.add_node("parse", parse_node)
	graph.add_node("chunk", chunk_node)
	graph.add_node("embed", embed_node)
	graph.add_node("upsert_pinecone", upsert_pinecone_node)
	graph.add_node("upsert_postgres", upsert_postgres_node)

	graph.add_edge(START, "load")
	graph.add_edge("load", "parse")
	graph.add_edge("parse", "chunk")
	graph.add_edge("chunk", "embed")
	graph.add_edge("embed", "upsert_pinecone")
	graph.add_edge("upsert_pinecone", "upsert_postgres")
	graph.add_edge("upsert_postgres", END)

	compiled = graph.compile()

	# Attach a graph-level error handler (LangGraph catches exceptions
	# at invoke-time; we wrap run helpers below to call this).
	compiled._ingestion_error_handler = error_handler  # type: ignore[attr-defined]
	return compiled


async def run_ingestion_job(
	*,
	file_path: str,
	sql_store: SQLStore,
	vector_store: VectorStore,
	extraction_agent: ExtractionAgent,
	embeddings: Optional[Embeddings] = None,
	document_id: uuid.UUID | None = None,
	chunk_strategy: str = "recursive",
) -> IngestionState:
	"""Convenience helper to run the compiled graph with one call."""
	graph = build_document_ingestion_graph(
		sql_store=sql_store,
		vector_store=vector_store,
		extraction_agent=extraction_agent,
		embeddings=embeddings,
	)
	initial: IngestionState = {"file_path": file_path, "chunk_strategy": chunk_strategy}
	if document_id is not None:
		initial["document_id"] = document_id

	try:
		return await graph.ainvoke(initial)
	except Exception as exc:
		# Best-effort: update DB statuses even if the graph failed mid-flight.
		handler = getattr(graph, "_ingestion_error_handler", None)
		if handler is not None:
			await handler(initial, exc)
		raise
