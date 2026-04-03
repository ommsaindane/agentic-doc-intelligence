from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.documents import Document

from app.agents.retrieval_agent import RetrievalAgent
from app.embeddings.embedding import get_embedding_model
from app.storage.sql_store import AsyncSessionLocal, SQLStore

load_dotenv()


@dataclass(frozen=True)
class Case:
    query: str
    document_ids: list[str]
    relevant_chunk_ids: set[str]


def _read_cases(path: Path) -> list[Case]:
    cases: list[Case] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            query = str(obj["query"])
            document_ids = [str(x) for x in obj.get("document_ids") or []]
            relevant_chunk_ids = {str(x) for x in obj.get("relevant_chunk_ids") or []}
            if not query.strip():
                raise ValueError(f"Case line {line_no}: query must be non-empty")
            if not document_ids:
                raise ValueError(f"Case line {line_no}: document_ids must be non-empty")
            if not relevant_chunk_ids:
                raise ValueError(f"Case line {line_no}: relevant_chunk_ids must be non-empty")
            cases.append(
                Case(
                    query=query,
                    document_ids=document_ids,
                    relevant_chunk_ids=relevant_chunk_ids,
                )
            )
    return cases


async def _bm25_corpus(sql: SQLStore, document_ids: Iterable[str]) -> list[Document]:
    corpus: list[Document] = []
    for doc_id in document_ids:
        chunks = await sql.get_chunks_by_document(uuid.UUID(doc_id))
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


def _first_relevant_rank(docs: list[Document], relevant_ids: set[str]) -> int | None:
    for i, d in enumerate(docs, start=1):
        cid = str((d.metadata or {}).get("chunk_id") or "")
        if cid in relevant_ids:
            return i
    return None


async def _run() -> int:
    parser = argparse.ArgumentParser(description="Benchmark retrieval with/without cross-encoder reranking")
    parser.add_argument(
        "--cases",
        required=True,
        help="Path to JSONL benchmark cases (each line: query, document_ids, relevant_chunk_ids)",
    )
    parser.add_argument("--k", type=int, default=8, help="Evaluate Recall@k and ranks in top-k")
    parser.add_argument("--semantic-k", type=int, default=12, help="Semantic retrieval k")
    parser.add_argument("--bm25-k", type=int, default=12, help="BM25 retrieval k")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.exists():
        raise FileNotFoundError(f"Cases file not found: {cases_path}")

    # Hard requirement: embeddings + Pinecone must be configured to run retrieval.
    embeddings = get_embedding_model()

    baseline = RetrievalAgent(embeddings=embeddings, enable_reranker=False)
    reranked = RetrievalAgent(embeddings=embeddings, enable_reranker=True)

    cases = _read_cases(cases_path)

    async with AsyncSessionLocal() as session:
        sql = SQLStore(session)

        baseline_rr = 0.0
        rerank_rr = 0.0
        baseline_recall = 0
        rerank_recall = 0

        for idx, case in enumerate(cases, start=1):
            pinecone_filter = {"document_id": {"$in": case.document_ids}}
            corpus = await _bm25_corpus(sql, case.document_ids)

            base_docs = await baseline.retrieve(
                case.query,
                pinecone_filter=pinecone_filter,
                bm25_corpus=corpus,
                semantic_k=args.semantic_k,
                bm25_k=args.bm25_k,
                max_results=args.k,
            )
            rerank_docs = await reranked.retrieve(
                case.query,
                pinecone_filter=pinecone_filter,
                bm25_corpus=corpus,
                semantic_k=args.semantic_k,
                bm25_k=args.bm25_k,
                max_results=args.k,
            )

            base_rank = _first_relevant_rank(base_docs, case.relevant_chunk_ids)
            rerank_rank = _first_relevant_rank(rerank_docs, case.relevant_chunk_ids)

            if base_rank is not None:
                baseline_recall += 1
                baseline_rr += 1.0 / base_rank
            if rerank_rank is not None:
                rerank_recall += 1
                rerank_rr += 1.0 / rerank_rank

            print(
                json.dumps(
                    {
                        "case": idx,
                        "baseline_rank": base_rank,
                        "rerank_rank": rerank_rank,
                        "baseline_top_chunk_id": (base_docs[0].metadata or {}).get("chunk_id") if base_docs else None,
                        "rerank_top_chunk_id": (rerank_docs[0].metadata or {}).get("chunk_id") if rerank_docs else None,
                    }
                )
            )

        n = max(len(cases), 1)
        print("\n=== Summary ===")
        print(f"Cases: {len(cases)}")
        print(f"Recall@{args.k} baseline: {baseline_recall}/{len(cases)} = {baseline_recall / n:.3f}")
        print(f"Recall@{args.k} rerank:   {rerank_recall}/{len(cases)} = {rerank_recall / n:.3f}")
        print(f"MRR@{args.k} baseline: {baseline_rr / n:.3f}")
        print(f"MRR@{args.k} rerank:   {rerank_rr / n:.3f}")

    return 0


def main() -> None:
    # Avoid any implicit asyncio loop reuse.
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
