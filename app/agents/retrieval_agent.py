"""Retrieval Agent — hybrid evidence retrieval for analytical QA.

Default behavior is **deterministic hybrid retrieval**:

- Pinecone semantic search (vector similarity)
- BM25 keyword search over a Postgres-derived corpus

Optionally, a deterministic **cross-encoder reranker** can be enabled via env
(`ENABLE_RERANKER=true`) to reorder the top-K merged candidates.
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


def _env_bool(name: str, default: bool = False) -> bool:
	val = os.getenv(name)
	if val is None:
		return default
	val = val.strip().lower()
	return val in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
	val = os.getenv(name)
	if val is None:
		return default
	try:
		return int(val)
	except ValueError as exc:
		raise ValueError(f"{name} must be an int (got {val!r})") from exc


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
		enable_reranker: Optional[bool] = None,
	):
		self.embeddings = embeddings
		self.index_name = pinecone_index_name or os.environ["PINECONE_INDEX_NAME"]

		self.enable_observability = _env_bool("ENABLE_OBSERVABILITY", False)
		self.enable_reranker = (
			bool(enable_reranker)
			if enable_reranker is not None
			else _env_bool("ENABLE_RERANKER", False)
		)
		self._reranker = None
		self._reranker_model_name = os.getenv("RERANKER_MODEL") or "BAAI/bge-reranker-base"
		self._reranker_device = os.getenv("RERANKER_DEVICE") or "cpu"
		self._reranker_batch_size = _env_int("RERANKER_BATCH_SIZE", 16)
		self._reranker_max_length = _env_int("RERANKER_MAX_LENGTH", 512)

		# PineconeVectorStore reads auth from env (PINECONE_API_KEY).
		self.vector_store = PineconeVectorStore(
			index_name=self.index_name,
			embedding=self.embeddings,
		)

		if self.enable_reranker:
			# Fail-fast if enabled and model cannot be loaded.
			try:
				from app.reranking.cross_encoder_reranker import CrossEncoderReranker
			except Exception as exc:
				raise RuntimeError(
					"Reranker is enabled (ENABLE_RERANKER=true) but required dependencies are missing. "
					"Install optional extras: `pip install .[reranker]` (or add torch+transformers)."
				) from exc

			self._reranker = CrossEncoderReranker(
				model_name=self._reranker_model_name,
				device=self._reranker_device,
				batch_size=self._reranker_batch_size,
				max_length=self._reranker_max_length,
			)

			logger.info(
				"Reranker enabled: model=%s device=%s batch_size=%d max_length=%d",
				self._reranker_model_name,
				self._reranker_device,
				self._reranker_batch_size,
				self._reranker_max_length,
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

		if self._reranker is None:
			return candidates[:max_results]

		rerank_top_k = _env_int("RERANKER_TOP_K", 50)
		rerank_top_n = _env_int("RERANKER_TOP_N", max_results)
		if rerank_top_k <= 0:
			raise ValueError("RERANKER_TOP_K must be > 0")
		if rerank_top_n <= 0:
			raise ValueError("RERANKER_TOP_N must be > 0")

		pool = candidates[: min(rerank_top_k, len(candidates))]
		results = self._reranker.rerank(query=query, candidates=pool)

		def _chunk_key(d: Document) -> str:
			md = d.metadata or {}
			return str(md.get("chunk_id") or md.get("id") or hash(d.page_content))

		reranked: list[Document] = []
		for r in results:
			r.doc.metadata = dict(r.doc.metadata or {})
			r.doc.metadata["_rerank_score"] = float(r.score)
			reranked.append(r.doc)

		# Deterministic tie-breaking: rerank_score → hybrid_score → chunk_id.
		reranked.sort(
			key=lambda d: (
				-float((d.metadata or {}).get("_rerank_score", 0.0)),
				-float((d.metadata or {}).get("_hybrid_score", 0.0)),
				_chunk_key(d),
			)
		)

		limit = min(max_results, rerank_top_n)
		if self.enable_observability:
			logger.info(
				"Retrieval rerank: candidates=%d pool=%d limit=%d",
				len(candidates),
				len(pool),
				limit,
			)

		return reranked[:limit]

