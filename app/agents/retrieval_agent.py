"""Retrieval Agent — hybrid evidence retrieval for analytical QA.

This repository uses **deterministic hybrid retrieval only**:

- Pinecone semantic search (vector similarity)
- BM25 keyword search over a Postgres-derived corpus

Results are merged and ordered deterministically (no LLM rerank/compression).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever

load_dotenv()

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Hybrid retrieval (semantic + BM25)
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class HybridWeights:
	semantic: float = 0.6
	bm25: float = 0.4


class RetrievalAgent:
	"""Hybrid retriever (semantic + BM25), no LLM compression."""

	def __init__(
		self,
		*,
		embeddings,
		pinecone_index_name: Optional[str] = None,
	):
		self.embeddings = embeddings
		self.index_name = pinecone_index_name or os.environ["PINECONE_INDEX_NAME"]

		# PineconeVectorStore reads auth from env (PINECONE_API_KEY).
		self.vector_store = PineconeVectorStore(
			index_name=self.index_name,
			embedding=self.embeddings,
		)

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
			bm25_docs = bm25.invoke(query)
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

	# ── Public API ──────────────────────────────────────────────────

	async def retrieve(
		self,
		query: str,
		*,
		pinecone_filter: Optional[dict] = None,
		bm25_corpus: Optional[list[Document]] = None,
		semantic_k: int = 12,
		bm25_k: int = 12,
		max_results: int = 8,
	) -> list[Document]:
		"""Hybrid retrieval (deterministic)."""

		candidates = await self.hybrid_retrieve(
			query,
			semantic_k=semantic_k,
			bm25_k=bm25_k,
			pinecone_filter=pinecone_filter,
			bm25_corpus=bm25_corpus,
		)

		return candidates[:max_results]

