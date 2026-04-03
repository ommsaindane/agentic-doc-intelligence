"""app.workflows.qa_graph

Final user-facing QA workflow orchestrator (LangGraph).

Flow
----
receive_query → retrieve (parallel via Send) → merge_evidence → analyze → verify →
	if fail: re-retrieve
	if pass: format_response → END

Key LangGraph primitives used
-----------------------------
- `StateGraph` with a `TypedDict` state.
- `Annotated[..., operator.add]` reducers for list fields that multiple
  parallel nodes append to (e.g., `evidence_bundles`).
- `Send` fan-out for per-document retrieval in parallel.
- `Command` to update state + choose next node.
- Optional checkpointing (caller can inject a checkpointer such as `MemorySaver`
	or a Postgres-backed saver). If a checkpointer is used, the caller must
	provide a `thread_id` via LangGraph's `configurable` config.
"""

from __future__ import annotations

import logging
import operator
import uuid
from typing import Any, Annotated, Optional, TypedDict

from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from app.agents.analysis_agent import AnalysisAgent, AnalysisResult
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.verifier_agent import VerificationResult, VerifierAgent
from app.storage.schemas import DocumentStatus
from app.storage.sql_store import SQLStore

logger = logging.getLogger(__name__)


class QAGraphState(TypedDict, total=False):
	# Inputs
	query: str
	document_ids: list[str]  # optional scope
	debug: bool

	# Retrieval config
	retrieval_round: int
	max_rounds: int
	semantic_k: int
	bm25_k: int

	# Parallel retrieval outputs (reduced with operator.add)
	# Each entry: {"document_id": str, "docs": [{"page_content": str, "metadata": dict}, ...]}
	evidence_bundles: Annotated[list[dict[str, Any]], operator.add]

	# Working evidence (serializable)
	merged_evidence: list[dict[str, Any]]

	# Agent outputs (serializable)
	analysis: dict[str, Any]
	verification: dict[str, Any]

	# Final output
	response: dict[str, Any]


def _doc_to_dict(doc: Document) -> dict[str, Any]:
	return {"page_content": doc.page_content, "metadata": dict(doc.metadata or {})}


def _dict_to_doc(obj: dict[str, Any]) -> Document:
	return Document(page_content=obj.get("page_content", ""), metadata=obj.get("metadata") or {})


def _dedupe_docs_by_chunk_id(docs: list[Document]) -> list[Document]:
	seen: set[str] = set()
	out: list[Document] = []
	for d in docs:
		md = d.metadata or {}
		cid = md.get("chunk_id") or md.get("id")
		key = str(cid) if cid is not None else d.page_content[:80]
		if key in seen:
			continue
		seen.add(key)
		out.append(d)
	return out


async def _bm25_corpus_for_doc(sql: SQLStore, document_id: str) -> list[Document]:
	"""Build a BM25 corpus from Postgres chunks for a given document id."""
	chunks = await sql.get_chunks_by_document(uuid.UUID(document_id))
	corpus: list[Document] = []
	for c in chunks:
		corpus.append(
			Document(
				page_content=c.content,
				metadata={
					"chunk_id": str(c.id),
					"document_id": str(c.document_id),
					"page_number": c.page_number,
					"element_type": c.element_type,
					"chunk_index": c.chunk_index,
				},
			)
		)
	return corpus


def build_qa_graph(
	*,
	sql_store: SQLStore,
	retrieval_agent: RetrievalAgent,
	analysis_agent: AnalysisAgent,
	verifier_agent: VerifierAgent,
	checkpointer: Any | None = None,
):
	"""Build and compile the QA graph with injected services."""

	async def receive_query(state: QAGraphState) -> Command:
		query = state["query"]
		retrieval_round = int(state.get("retrieval_round", 0))
		max_rounds = int(state.get("max_rounds", 2))

		# Expand retrieval parameters each round.
		base_semantic_k = int(state.get("semantic_k", 10))
		base_bm25_k = int(state.get("bm25_k", 10))
		semantic_k = base_semantic_k + retrieval_round * 6
		bm25_k = base_bm25_k + retrieval_round * 6

		# Determine relevant document IDs.
		document_ids = state.get("document_ids")
		if not document_ids:
			docs = await sql_store.list_documents(status=DocumentStatus.COMPLETE)
			document_ids = [str(d.id) for d in docs]

		if not document_ids:
			# No docs to search — still return a response path.
			return Command(
				update={
					"evidence_bundles": [],
					"merged_evidence": [],
					"analysis": {"answer": "No documents are available for retrieval.", "reasoning": "", "key_claims": [], "contradictions": [], "citations": [], "limits": ["No ingested documents."]},
					"verification": {"is_grounded": False, "citations": [], "rejected_claims": ["No evidence"], "verified_claims": [], "revised_answer": "No documents are available for retrieval.", "missing_evidence": ["Ingest documents first."]},
				},
				goto="format_response",
			)

		if retrieval_round >= max_rounds:
			# Safety: stop reretrieval.
			return Command(goto="merge_evidence")

		sends = [
			Send(
				"retrieve",
				{
					"query": query,
					"document_id": doc_id,
					"semantic_k": semantic_k,
					"bm25_k": bm25_k,
				},
			)
			for doc_id in document_ids
		]

		logger.info(
			"QA receive_query: %d docs fan-out (round=%d, semantic_k=%d, bm25_k=%d)",
			len(sends),
			retrieval_round,
			semantic_k,
			bm25_k,
		)

		return Command(
			update={
				"document_ids": document_ids,
				"retrieval_round": retrieval_round,
				"semantic_k": semantic_k,
				"bm25_k": bm25_k,
			},
			goto=sends,
		)

	async def retrieve(state: dict[str, Any]) -> dict[str, Any]:
		"""Per-document retrieval worker (target of Send fan-out)."""
		query: str = state["query"]
		document_id: str = state["document_id"]
		semantic_k: int = int(state.get("semantic_k", 12))
		bm25_k: int = int(state.get("bm25_k", 12))

		# Pinecone metadata filter: restrict to one doc.
		pinecone_filter = {"document_id": {"$eq": document_id}}

		# BM25 corpus from Postgres chunks.
		bm25_corpus = await _bm25_corpus_for_doc(sql_store, document_id)

		docs = await retrieval_agent.retrieve(
			query,
			pinecone_filter=pinecone_filter,
			bm25_corpus=bm25_corpus,
			semantic_k=semantic_k,
			bm25_k=bm25_k,
			max_results=8,
		)

		# Add deterministic ranks for debug visibility.
		for rank, d in enumerate(docs, start=1):
			d.metadata = dict(d.metadata or {})
			d.metadata["_retrieval_rank"] = rank

		bundle = {
			"document_id": document_id,
			"docs": [_doc_to_dict(d) for d in docs],
		}
		return {"evidence_bundles": [bundle]}

	async def merge_evidence(state: QAGraphState) -> dict[str, Any]:
		bundles = state.get("evidence_bundles", [])
		all_docs: list[Document] = []
		for b in bundles:
			for d in b.get("docs", []):
				all_docs.append(_dict_to_doc(d))

		all_docs = _dedupe_docs_by_chunk_id(all_docs)
		logger.info("Merged evidence: %d bundles → %d unique docs", len(bundles), len(all_docs))
		return {"merged_evidence": [_doc_to_dict(d) for d in all_docs]}

	async def analyze(state: QAGraphState) -> dict[str, Any]:
		query = state["query"]
		evidence_docs = [_dict_to_doc(d) for d in state.get("merged_evidence", [])]
		result: AnalysisResult = await analysis_agent.run(question=query, evidence_docs=evidence_docs)
		return {"analysis": result.model_dump()}

	async def verify(state: QAGraphState) -> Command:
		query = state["query"]
		evidence_docs = [_dict_to_doc(d) for d in state.get("merged_evidence", [])]
		analysis = AnalysisResult.model_validate(state.get("analysis") or {})

		verification: VerificationResult = await verifier_agent.verify_analysis_result(
			question=query,
			analysis=analysis,
			evidence_docs=evidence_docs,
		)

		# Gate: reretrieve if not grounded.
		is_grounded = bool(verification.is_grounded)
		round_idx = int(state.get("retrieval_round", 0))
		max_rounds = int(state.get("max_rounds", 2))

		if (not is_grounded) and (round_idx + 1 < max_rounds):
			logger.info("Verifier failed grounding; reretrieving (next_round=%d)", round_idx + 1)
			return Command(
				update={
					"verification": verification.model_dump(),
					"retrieval_round": round_idx + 1,
				},
				goto="receive_query",
			)

		# Either grounded or out of rounds — proceed to formatting.
		return Command(
			update={"verification": verification.model_dump()},
			goto="format_response",
		)

	async def format_response(state: QAGraphState) -> dict[str, Any]:
		"""Return the final response the API will send back to the user."""

		analysis = state.get("analysis") or {}
		verification = state.get("verification") or {}

		# Prefer the verifier's revised answer since it removes unsupported claims.
		final_text = verification.get("revised_answer") or analysis.get("answer") or ""
		citations = verification.get("citations") or []
		grounded = bool(verification.get("is_grounded", False))

		response: dict[str, Any] = {
			"answer": final_text,
			"is_grounded": grounded,
			"citations": citations,
			"rejected_claims": verification.get("rejected_claims", []),
			"missing_evidence": verification.get("missing_evidence", []),
		}

		if bool(state.get("debug", False)):
			response["debug"] = {
				"reranker_enabled": bool(getattr(retrieval_agent, "_reranker", None)),
				"evidence_bundles": state.get("evidence_bundles", []),
				"merged_evidence": state.get("merged_evidence", []),
				"retrieval_round": int(state.get("retrieval_round", 0)),
			}

		return {"response": response}

	graph = StateGraph(QAGraphState)
	graph.add_node("receive_query", receive_query)
	graph.add_node("retrieve", retrieve)
	graph.add_node("merge_evidence", merge_evidence)
	graph.add_node("analyze", analyze)
	graph.add_node("verify", verify)
	graph.add_node("format_response", format_response)

	graph.add_edge(START, "receive_query")
	graph.add_edge("retrieve", "merge_evidence")
	graph.add_edge("merge_evidence", "analyze")
	graph.add_edge("analyze", "verify")
	graph.add_edge("format_response", END)

	# Note: verifier routing is handled via `Command.goto`.
	if checkpointer is None:
		return graph.compile()
	return graph.compile(checkpointer=checkpointer)


async def run_qa_graph(
	*,
	query: str,
	sql_store: SQLStore,
	retrieval_agent: RetrievalAgent,
	analysis_agent: AnalysisAgent,
	verifier_agent: VerifierAgent,
	document_ids: Optional[list[str]] = None,
	max_rounds: int = 2,
	semantic_k: int = 10,
	bm25_k: int = 10,
	debug: bool = False,
	checkpointer: Any | None = None,
	thread_id: Optional[str] = None,
) -> dict[str, Any]:
	"""Convenience wrapper to execute the compiled QA graph."""
	if (checkpointer is not None) and (not thread_id):
		raise ValueError("thread_id is required when a checkpointer is provided")

	checkpointer_to_use: Any | None = None
	config: dict[str, Any] = {}
	if thread_id:
		checkpointer_to_use = checkpointer or MemorySaver()
		config = {"configurable": {"thread_id": thread_id}}

	graph = build_qa_graph(
		sql_store=sql_store,
		retrieval_agent=retrieval_agent,
		analysis_agent=analysis_agent,
		verifier_agent=verifier_agent,
		checkpointer=checkpointer_to_use,
	)

	initial: QAGraphState = {
		"query": query,
		"document_ids": document_ids or [],
		"debug": debug,
		"retrieval_round": 0,
		"max_rounds": max_rounds,
		"semantic_k": semantic_k,
		"bm25_k": bm25_k,
		"evidence_bundles": [],
	}

	final_state: QAGraphState = await graph.ainvoke(initial, config=config)
	return final_state.get("response") or {}

