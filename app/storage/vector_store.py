# storage/vector_store.py

import logging
import os
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from .schemas import ChunkModel

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Wraps all Pinecone operations for the application.

    Nothing outside this file imports Pinecone directly.  Every vector
    carries enough metadata for useful filtered searches without round-
    tripping to Postgres, but full chunk text stays in Postgres — Pinecone
    isn't designed for large text blobs and we avoid duplicating data.

    Metadata stored per vector:
        chunk_id, document_id, element_type, chunk_index,
        page_number, doc_type
    """

    def __init__(self, embeddings: Embeddings):
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index_name = os.environ["PINECONE_INDEX_NAME"]
        self.embeddings = embeddings

        # LangChain wrapper around the Pinecone index — gives access to
        # .similarity_search(), .as_retriever(), and other high-level
        # retrieval patterns.
        self.store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
        )
        logger.info("VectorStore initialised — index: %s", self.index_name)

    # ── Upsert ──────────────────────────────────────────────────────

    async def upsert_chunks(
        self,
        chunks: list[ChunkModel],
        doc_type: Optional[str] = None,
    ) -> list[str]:
        """
        Convert ChunkModel rows into LangChain Documents and upsert them
        to Pinecone.  We use the chunk's UUID as the Pinecone record ID
        so the two systems stay perfectly in sync.

        Metadata is deliberately minimal — IDs and key filtering fields
        only.  Full content lives in Postgres.
        """
        documents: list[Document] = []
        ids: list[str] = []

        for chunk in chunks:
            doc = Document(
                page_content=chunk.content,
                metadata={
                    "chunk_id": str(chunk.id),
                    "document_id": str(chunk.document_id),
                    "element_type": chunk.element_type or "NarrativeText",
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number or 0,
                    "doc_type": doc_type or "unknown",
                },
            )
            documents.append(doc)
            ids.append(str(chunk.id))

        # add_documents handles embedding generation + upserting in one call
        await self.store.aadd_documents(documents, ids=ids)
        logger.info("Upserted %d vectors for doc_type=%s", len(ids), doc_type)
        return ids

    # ── Retrieval ───────────────────────────────────────────────────

    def as_retriever(self, k: int = 8, filter: Optional[dict] = None):
        """
        Return a LangChain retriever for use in the Retrieval Agent or
        composed into an EnsembleRetriever.

        The *filter* parameter scopes retrieval to specific documents —
        essential for cross-document analysis where you want evidence
        from a known set of sources.

        Example filter::

            {"document_id": {"$in": ["id1", "id2"]}}
        """
        return self.store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k,
                **({"filter": filter} if filter else {}),
            },
        )

    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 8,
        filter: Optional[dict] = None,
    ) -> list[tuple[Document, float]]:
        """
        Return (Document, score) pairs — useful when the Retrieval Agent
        needs confidence scores or the Verifier Agent wants to rank
        evidence by relevance.
        """
        return await self.store.asimilarity_search_with_score(
            query,
            k=k,
            **({"filter": filter} if filter else {}),
        )

    # ── Deletion ────────────────────────────────────────────────────

    async def delete_document_vectors(self, document_id: str) -> None:
        """
        Delete all vectors belonging to a document.

        Needed when a user deletes or re-ingests a document.  Pinecone
        supports metadata-filtered deletion, which is why we store
        ``document_id`` in every vector's metadata.
        """
        index = self.pc.Index(self.index_name)
        index.delete(filter={"document_id": {"$eq": document_id}})
        logger.info("Deleted vectors for document %s", document_id)