"""app.ingestion.ingestion

Document ingestion module.

This repository ingests **PDFs only** and extracts text deterministically using
PyMuPDF (`pymupdf`, imported as `fitz`). No OCR is performed.

Loading yields one `langchain_core.documents.Document` per page with metadata
required for citations:

- `source`: absolute file path
- `page_number`: 1-based page number
- `element_type`: set to "page"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

from langchain_core.documents import Document
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ── Supported extensions ────────────────────────────────────────────
SUPPORTED_EXTENSIONS: set[str] = {
    ".pdf",
}


def _validate_path(path: Path) -> Path:
    """Resolve and validate a file path."""
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Path is not a file: {resolved}")
    if resolved.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{resolved.suffix}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return resolved


def load_document(file_path: str | Path) -> Generator[Document, None, None]:
    """
    Lazily load a single PDF document using PyMuPDF.

    Yields one `langchain_core.documents.Document` per page.

    Parameters
    ----------
    file_path : str | Path
        Path to the document to ingest.

    Yields
    ------
    Document
        One document per PDF page.
    """
    path = _validate_path(Path(file_path))
    logger.info("Loading document: %s", path)

    page_count = 0
    try:
        with fitz.open(str(path)) as pdf:
            if pdf.page_count <= 0:
                raise ValueError(f"PDF has no pages: {path}")

            for idx in range(pdf.page_count):
                page = pdf.load_page(idx)
                text = (page.get_text("text") or "").strip()

                # Deterministic ingestion: if a page has no extractable text,
                # still yield an empty page so citations remain consistent.
                yield Document(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "page_number": idx + 1,
                        "element_type": "page",
                    },
                )
                page_count += 1
    except Exception as exc:
        raise ValueError(f"Failed to parse PDF with PyMuPDF: {path}. Error: {exc}") from exc

    logger.info("Finished loading %s – %d page(s) extracted.", path.name, page_count)


def load_directory(
    dir_path: str | Path,
    *,
    glob: str = "**/*",
    recursive: bool = True,
) -> Generator[Document, None, None]:
    """
    Walk a directory and lazily ingest every supported file found.

    Parameters
    ----------
    dir_path : str | Path
        Root directory to scan.
    glob : str
        Glob pattern for file discovery (default ``"**/*"``).
    recursive : bool
        Whether the glob pattern is applied recursively (default True).

    Yields
    ------
    Document
        One document chunk per structural element, across all files.
    """
    root = Path(dir_path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Directory not found: {root}")

    pattern = glob if recursive else glob.replace("**/", "")
    files = sorted(
        p for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        logger.warning("No supported files found in %s", root)
        return

    logger.info("Found %d supported file(s) in %s", len(files), root)

    for file in files:
        yield from load_document(file)
