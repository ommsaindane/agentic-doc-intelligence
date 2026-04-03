"""
Document chunking module.

Provides two strategies:

1. **Recursive character splitting** (`RecursiveCharacterTextSplitter`) –
   the general-purpose default.  Fast, deterministic, no model required.
2. **Semantic splitting** (`SemanticChunker` from `langchain_experimental`) –
   splits on embedding-similarity boundaries so each chunk preserves a
   coherent argument unit.  Better for analytical QA but requires an
   embeddings model.

Both splitters set ``add_start_index=True`` so every chunk carries the byte
offset into the original element text — useful for citation highlighting.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE: int = 1_000
DEFAULT_CHUNK_OVERLAP: int = 200


# ── Recursive character chunking ────────────────────────────────────
def chunk_recursive(
    documents: Iterable[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split documents with `RecursiveCharacterTextSplitter`.

    Parameters
    ----------
    documents : Iterable[Document]
        Source documents (e.g. from ``ingestion.load_document``).
    chunk_size : int
        Maximum characters per chunk.
    chunk_overlap : int
        Character overlap between consecutive chunks.

    Returns
    -------
    list[Document]
        Chunked documents with ``start_index`` in metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    docs = list(documents)  # materialise the generator once
    chunks = splitter.split_documents(docs)
    logger.info(
        "Recursive chunking: %d docs → %d chunks (size=%d, overlap=%d)",
        len(docs), len(chunks), chunk_size, chunk_overlap,
    )
    return chunks


# ── Semantic chunking ───────────────────────────────────────────────
def chunk_semantic(
    documents: Iterable[Document],
    embeddings: Embeddings,
    *,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float | None = None,
) -> list[Document]:
    """
    Split documents with `SemanticChunker` (embedding-similarity boundaries).

    Produces semantically coherent chunks — much better for analytical QA
    because each chunk preserves a complete argument / reasoning unit.

    Parameters
    ----------
    documents : Iterable[Document]
        Source documents (e.g. from ``ingestion.load_document``).
    embeddings : Embeddings
        Any LangChain-compatible embeddings model.
    breakpoint_threshold_type : str
        How to detect semantic boundaries.
        One of ``"percentile"`` | ``"standard_deviation"`` | ``"interquartile"``
        | ``"gradient"``.
    breakpoint_threshold_amount : float | None
        Threshold value; if ``None`` the chunker uses its built-in default.

    Returns
    -------
    list[Document]
        Chunked documents that respect semantic boundaries.
    """
    from langchain_experimental.text_splitter import SemanticChunker

    kwargs: dict = {
        "embeddings": embeddings,
        "breakpoint_threshold_type": breakpoint_threshold_type,
        "add_start_index": True,
    }
    if breakpoint_threshold_amount is not None:
        kwargs["breakpoint_threshold_amount"] = breakpoint_threshold_amount

    splitter = SemanticChunker(**kwargs)

    docs = list(documents)
    chunks = splitter.split_documents(docs)
    logger.info(
        "Semantic chunking: %d docs → %d chunks (threshold_type=%s)",
        len(docs), len(chunks), breakpoint_threshold_type,
    )
    return chunks


# ── Section-aware chunking ─────────────────────────────────────────
def chunk_section_aware(
    documents: Iterable[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Chunk documents by section metadata, without splitting tables.

    Expected input
    --------------
    Elements should carry (best-effort) metadata keys:
      - `section_title` / `section_level`
      - `layout_type` (e.g. "text", "table")

    Behavior
    --------
    - Accumulates contiguous non-table elements under the same `section_title`.
    - Emits each table element as a single chunk (no table splitting).
    - Applies the existing recursive splitter to text groups so we still get
      `start_index` and overlap behavior.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    chunks: list[Document] = []
    buffer_parts: list[str] = []
    buffer_pages: list[int] = []
    buffer_meta: dict = {}

    def flush_buffer() -> None:
        nonlocal buffer_parts, buffer_pages, buffer_meta, chunks
        if not buffer_parts:
            buffer_pages = []
            buffer_meta = {}
            return

        text = "\n\n".join(p for p in buffer_parts if p.strip()).strip()
        if not text:
            buffer_parts = []
            buffer_pages = []
            buffer_meta = {}
            return

        meta = dict(buffer_meta)
        meta.pop("bbox", None)
        if buffer_pages:
            meta["page_numbers"] = sorted({p for p in buffer_pages if isinstance(p, int)})
            meta["page_number"] = meta.get("page_number") or meta["page_numbers"][0]

        # Normalize element/layout type for downstream storage.
        layout_type = meta.get("layout_type") or meta.get("element_type")
        if isinstance(layout_type, str) and layout_type:
            meta["layout_type"] = layout_type
            meta["element_type"] = meta.get("element_type") or layout_type
        else:
            meta["layout_type"] = meta.get("layout_type") or "text"
            meta["element_type"] = meta.get("element_type") or "text"

        doc = Document(page_content=text, metadata=meta)
        chunks.extend(splitter.split_documents([doc]))

        buffer_parts = []
        buffer_pages = []
        buffer_meta = {}

    docs = list(documents)
    for doc in docs:
        content = (doc.page_content or "").strip()
        if not content:
            continue

        meta = doc.metadata or {}
        section_title = meta.get("section_title")
        section_level = meta.get("section_level")
        layout_type = (meta.get("layout_type") or meta.get("element_type") or "text")
        if isinstance(layout_type, str):
            layout_type = layout_type.lower()
        else:
            layout_type = "text"

        # Tables are emitted as single chunks.
        if layout_type == "table":
            flush_buffer()

            table_meta = dict(meta)
            table_meta.pop("bbox", None)
            table_meta["layout_type"] = meta.get("layout_type") or "table"
            table_meta["element_type"] = meta.get("element_type") or "table"
            chunks.append(Document(page_content=content, metadata=table_meta))
            continue

        # Start a new section group when section_title changes.
        if buffer_meta:
            if section_title != buffer_meta.get("section_title") or section_level != buffer_meta.get(
                "section_level"
            ):
                flush_buffer()

        if not buffer_meta:
            buffer_meta = {
                "source": meta.get("source"),
                "page_number": meta.get("page_number"),
                "section_title": section_title,
                "section_level": section_level,
                "layout_type": meta.get("layout_type") or "text",
                "element_type": meta.get("element_type") or "text",
                "layout_aware": bool(meta.get("layout_aware")),
            }

        page_number = meta.get("page_number")
        if isinstance(page_number, int):
            buffer_pages.append(page_number)
        buffer_parts.append(content)

    flush_buffer()

    logger.info(
        "Section-aware chunking: %d docs → %d chunks (size=%d, overlap=%d)",
        len(docs),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


# ── Convenience dispatcher ──────────────────────────────────────────
def chunk_documents(
    documents: Iterable[Document],
    *,
    strategy: str = "recursive",
    embeddings: Embeddings | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float | None = None,
) -> list[Document]:
    """
    Unified entry point – pick a chunking strategy by name.

    Parameters
    ----------
    documents : Iterable[Document]
        Source documents to chunk.
    strategy : str
        ``"recursive"`` (default) or ``"semantic"``.
    embeddings : Embeddings | None
        Required when ``strategy="semantic"``.
    chunk_size : int
        For recursive strategy only.
    chunk_overlap : int
        For recursive strategy only.
    breakpoint_threshold_type : str
        For semantic strategy only.
    breakpoint_threshold_amount : float | None
        For semantic strategy only.

    Returns
    -------
    list[Document]
        Chunked documents.
    """
    if strategy == "recursive":
        return chunk_recursive(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    if strategy == "semantic":
        if embeddings is None:
            raise ValueError(
                "An embeddings model is required for semantic chunking. "
                "Pass an `Embeddings` instance via the `embeddings` parameter."
            )
        return chunk_semantic(
            documents,
            embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )

    if strategy == "section_aware":
        return chunk_section_aware(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    raise ValueError(
        f"Unknown chunking strategy '{strategy}'. "
        "Choose 'recursive', 'semantic', or 'section_aware'."
    )
