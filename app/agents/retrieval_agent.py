"""Retrieval Agent — hybrid evidence retrieval for analytical QA.

Requirements implemented
------------------------
- Use `langchain_pinecone.PineconeVectorStore` directly for semantic search.
- For cross-document retrieval, combine semantic search with keyword BM25
  search (`langchain_community.retrievers.BM25Retriever`).
- Wrap the hybrid retriever in a contextual compression step that uses an
  LLM to **re-rank** and **compress** chunks, dramatically reducing noise.

Why hybrid + compression
------------------------
Semantic retrieval is great for paraphrases, but can miss exact entities and
figures. BM25 boosts literal matches. Contextual compression then removes
irrelevant parts and reorders by query relevance before handing context to
the analysis agent.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  LLM compression schema
# ═══════════════════════════════════════════════════════════════════


class CompressedChunk(BaseModel):
	"""A single compressed chunk selected as evidence."""

	chunk_id: str = Field(description="Chunk UUID (from metadata.chunk_id).")
	score: float = Field(ge=0.0, le=1.0, description="Relevance score 0..1.")
	compressed_text: str = Field(
		description="Only the sentences/phrases relevant to the query."
	)


class CompressionResult(BaseModel):
	"""LLM output: top evidence chunks after rerank + compression."""

	results: list[CompressedChunk] = Field(default_factory=list)


COMPRESSION_PROMPT = ChatPromptTemplate.from_messages(
	[
		(
			"system",
			"You are a retrieval compression model. Given a user query and a set "
			"of candidate chunks, do two things: (1) drop irrelevant chunks, "
			"(2) for kept chunks, extract ONLY the spans relevant to answering the "
			"query. Return JSON matching the schema exactly.\n\n{format_instructions}",
		),
		(
			"human",
			"Query: {query}\n\n"
			"Candidate chunks (JSON array):\n{candidates}",
		),
	]
)


# ═══════════════════════════════════════════════════════════════════
#  Hybrid retrieval (semantic + BM25)
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class HybridWeights:
	semantic: float = 0.6
	bm25: float = 0.4


class RetrievalAgent:
	"""Hybrid retriever with optional contextual compression."""

	def __init__(
		self,
		*,
		embeddings,
		llm: Optional[BaseChatModel] = None,
		pinecone_index_name: Optional[str] = None,
	):
		self.embeddings = embeddings
		self.index_name = pinecone_index_name or os.environ["PINECONE_INDEX_NAME"]

		# PineconeVectorStore reads auth from env (PINECONE_API_KEY).
		self.vector_store = PineconeVectorStore(
			index_name=self.index_name,
			embedding=self.embeddings,
		)

		self.llm = llm or ChatOpenAI(
			model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
			temperature=0,
			api_key=os.environ["OPENAI_API_KEY"],
		)

		self._compression_parser = PydanticOutputParser(pydantic_object=CompressionResult)
		self._compress_chain = COMPRESSION_PROMPT | self.llm | self._compression_parser

	# ── Pinecone semantic retrieval ────────────────────────────────

	async def _semantic_search(
		self,
		query: str,
		*,
		k: int,
		filter: Optional[dict] = None,
	) -> list[tuple[Document, float]]:
		"""Return semantic hits with scores if supported."""

		# Prefer with-score retrieval when available.
		try:
			return await self.vector_store.asimilarity_search_with_score(
				query,
				k=k,
				**({"filter": filter} if filter else {}),
			)
		except Exception:
			docs = await self.vector_store.asimilarity_search(
				query,
				k=k,
				**({"filter": filter} if filter else {}),
			)
			# Approximate scores by rank.
			return [(d, 1.0 / (i + 1)) for i, d in enumerate(docs)]

	# ── BM25 keyword retrieval ─────────────────────────────────────

	def _build_bm25(
		self,
		corpus_documents: Iterable[Document],
		*,
		k: int,
	) -> BM25Retriever:
		retriever = BM25Retriever.from_documents(list(corpus_documents))
		retriever.k = k
		return retriever

	# ── Hybrid ensemble ────────────────────────────────────────────

	async def hybrid_retrieve(
		self,
		query: str,
		*,
		semantic_k: int = 12,
		bm25_k: int = 12,
		pinecone_filter: Optional[dict] = None,
		bm25_corpus: Optional[list[Document]] = None,
		weights: HybridWeights = HybridWeights(),
	) -> list[Document]:
		"""
		Retrieve candidate chunks using hybrid semantic + keyword search.

		Parameters
		----------
		pinecone_filter:
			Pinecone metadata filter (e.g. {"document_id": {"$in": [...]}}).
		bm25_corpus:
			Documents to build BM25 over. For cross-document retrieval this
			should be the union of chunk Documents for the relevant documents.
		"""

		semantic_hits = await self._semantic_search(
			query,
			k=semantic_k,
			filter=pinecone_filter,
		)

		semantic_docs: list[Document] = []
		for doc, score in semantic_hits:
			# Keep a semantic score for later merging/rerank.
			doc.metadata = dict(doc.metadata or {})
			doc.metadata["_semantic_score"] = float(score)
			semantic_docs.append(doc)

		bm25_docs: list[Document] = []
		if bm25_corpus:
			bm25 = self._build_bm25(bm25_corpus, k=bm25_k)
			bm25_docs = bm25.get_relevant_documents(query)
			for i, d in enumerate(bm25_docs):
				d.metadata = dict(d.metadata or {})
				d.metadata["_bm25_score"] = float(1.0 / (i + 1))

		# Merge by chunk_id when possible.
		merged: dict[str, Document] = {}

		def key_for(d: Document) -> str:
			md = d.metadata or {}
			return str(md.get("chunk_id") or md.get("id") or hash(d.page_content))

		for d in semantic_docs + bm25_docs:
			k = key_for(d)
			if k not in merged:
				merged[k] = d
			else:
				# Combine scores if doc appears in both.
				existing = merged[k]
				md = dict(existing.metadata or {})
				md.update(d.metadata or {})
				existing.metadata = md
				merged[k] = existing

		# Compute combined score for ordering.
		scored: list[tuple[Document, float]] = []
		for d in merged.values():
			md = d.metadata or {}
			semantic_score = float(md.get("_semantic_score", 0.0))
			bm25_score = float(md.get("_bm25_score", 0.0))
			combined = weights.semantic * semantic_score + weights.bm25 * bm25_score
			md["_hybrid_score"] = combined
			d.metadata = md
			scored.append((d, combined))

		scored.sort(key=lambda x: x[1], reverse=True)
		return [d for d, _ in scored]

	# ── Contextual compression (LLM rerank + extract) ───────────────

	async def contextual_compress(
		self,
		*,
		query: str,
		documents: list[Document],
		max_candidates: int = 20,
		max_results: int = 8,
		max_chars_per_candidate: int = 1500,
	) -> list[Document]:
		"""
		Re-rank and compress candidate chunks using an LLM.

		Returns Documents whose `page_content` is the compressed text,
		while preserving original metadata (chunk_id/document_id/page_number).
		"""
		if not documents:
			return []

		candidates = documents[:max_candidates]

		payload: list[dict[str, Any]] = []
		for d in candidates:
			md = d.metadata or {}
			chunk_id = md.get("chunk_id") or md.get("id")
			payload.append(
				{
					"chunk_id": str(chunk_id) if chunk_id is not None else "",
					"document_id": str(md.get("document_id", "")),
					"page_number": md.get("page_number"),
					"element_type": md.get("element_type"),
					"chunk_index": md.get("chunk_index"),
					"text": d.page_content[:max_chars_per_candidate],
				}
			)

		candidates_json = json.dumps(payload, ensure_ascii=False)
		result: CompressionResult = await self._compress_chain.ainvoke(
			{
				"query": query,
				"candidates": candidates_json,
				"format_instructions": self._compression_parser.get_format_instructions(),
			}
		)

		# Build a map from chunk_id → original Document to preserve metadata.
		by_id: dict[str, Document] = {}
		for d in candidates:
			md = d.metadata or {}
			cid = md.get("chunk_id") or md.get("id")
			if cid is not None:
				by_id[str(cid)] = d

		compressed_docs: list[tuple[Document, float]] = []
		for item in result.results:
			original = by_id.get(item.chunk_id)
			if original is None:
				continue
			md = dict(original.metadata or {})
			md["_compression_score"] = item.score
			md["_compressed"] = True
			compressed_docs.append(
				(
					Document(page_content=item.compressed_text, metadata=md),
					item.score,
				)
			)

		compressed_docs.sort(key=lambda x: x[1], reverse=True)
		return [d for d, _ in compressed_docs[:max_results]]

	# ── Public API ──────────────────────────────────────────────────

	async def retrieve(
		self,
		query: str,
		*,
		pinecone_filter: Optional[dict] = None,
		bm25_corpus: Optional[list[Document]] = None,
		semantic_k: int = 12,
		bm25_k: int = 12,
		compress: bool = True,
		max_results: int = 8,
	) -> list[Document]:
		"""Hybrid retrieval with optional contextual compression."""

		candidates = await self.hybrid_retrieve(
			query,
			semantic_k=semantic_k,
			bm25_k=bm25_k,
			pinecone_filter=pinecone_filter,
			bm25_corpus=bm25_corpus,
		)

		if not compress:
			return candidates[:max_results]

		try:
			return await self.contextual_compress(
				query=query,
				documents=candidates,
				max_results=max_results,
			)
		except Exception:
			# Fail open: retrieval still works even if compression fails.
			logger.exception("Contextual compression failed; returning raw candidates")
			return candidates[:max_results]

