"""
Document ingestion module.

Uses `langchain-unstructured` with `UnstructuredLoader` (strategy="hi_res")
for layout-aware parsing with automatic OCR. All loading uses `lazy_load()`
to stream pages as a generator – safe for large documents.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader

logger = logging.getLogger(__name__)

# ── Supported extensions ────────────────────────────────────────────
SUPPORTED_EXTENSIONS: set[str] = {
    ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp",
    ".docx", ".doc", ".pptx", ".xlsx", ".html", ".txt", ".md",
    ".eml", ".msg",
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
    Lazily load a single document using Unstructured hi-res strategy.

    Yields `langchain_core.documents.Document` objects one element at a
    time so arbitrarily large files never blow up memory.

    Each yielded Document carries rich metadata added by Unstructured
    (page number, element type, coordinates, etc.) plus the source path.

    Parameters
    ----------
    file_path : str | Path
        Path to the document to ingest.

    Yields
    ------
    Document
        One document chunk per structural element detected by Unstructured.
    """
    path = _validate_path(Path(file_path))
    logger.info("Loading document: %s", path)

    loader = UnstructuredLoader(
        file_path=str(path),
        strategy="hi_res",          # layout detection + OCR
        mode="elements",            # one Document per structural element
    )

    element_count = 0
    for doc in loader.lazy_load():
        # Ensure every chunk carries a source reference for citation
        doc.metadata.setdefault("source", str(path))
        element_count += 1
        yield doc

    logger.info("Finished loading %s – %d elements extracted.", path.name, element_count)


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
