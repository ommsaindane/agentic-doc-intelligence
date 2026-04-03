"""app.ingestion.ingestion

Document ingestion module.

Default ingestion is **PDF-only** and extracts text deterministically using
PyMuPDF (`pymupdf`, imported as `fitz`). No OCR is performed.

An optional layout-aware loader is also provided. It uses PyMuPDF4LLM (and the
PyMuPDF-Layout module, if installed) to yield smaller structural elements with
extra metadata (e.g. `layout_type`, `section_title`). OCR remains strict opt-in
and is never applied unless explicitly enabled by the caller.

Loading yields `langchain_core.documents.Document` elements with citation
metadata:

- `source`: absolute file path
- `page_number`: 1-based page number
- `element_type`: e.g. "page" (default loader) or a layout class like "text" / "table"
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any, Generator

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


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _require_pymupdf4llm() -> Any:
    try:
        import pymupdf4llm  # type: ignore

        return pymupdf4llm
    except Exception as exc:
        raise RuntimeError(
            "Layout-aware ingestion requires the optional dependency 'pymupdf4llm' "
            "(and typically 'pymupdf-layout'). Install them to use layout_aware parsing."
        ) from exc


def load_document_layout_aware(
    file_path: str | Path,
    *,
    enable_ocr: bool = False,
    ocr_language: str = "eng",
) -> Generator[Document, None, None]:
    """Layout-aware PDF loader using PyMuPDF4LLM.

    Yields one `Document` per detected layout boundary box (reading order).
    Each element includes:
      - `layout_type`: a box class like "text", "table", "picture", ...
      - `section_title` / `section_level`: best-effort from Markdown headings

    OCR is strict opt-in via `enable_ocr`.
    """
    path = _validate_path(Path(file_path))
    logger.info("Loading document (layout-aware): %s", path)

    pymupdf4llm = _require_pymupdf4llm()

    # If PyMuPDF-Layout is installed, PyMuPDF4LLM will use it by default.
    # This is a global toggle inside pymupdf4llm.
    try:
        pymupdf4llm.use_layout(True)
    except Exception:
        # If the layout module isn't present, pymupdf4llm may still work.
        pass

    try:
        page_chunks = pymupdf4llm.to_markdown(
            str(path),
            page_chunks=True,
            use_ocr=bool(enable_ocr),
            ocr_language=str(ocr_language or "eng"),
            show_progress=False,
            write_images=False,
            embed_images=False,
            page_separators=False,
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to parse PDF with PyMuPDF4LLM: {path}. Error: {exc}"
        ) from exc

    # PyMuPDF4LLM returns JSON strings for to_json(), but to_markdown(page_chunks=True)
    # returns a list of per-page dicts.
    if isinstance(page_chunks, str):
        try:
            page_chunks = json.loads(page_chunks)
        except Exception as exc:
            raise ValueError(
                "Unexpected PyMuPDF4LLM output type: expected list[dict] or JSON string."
            ) from exc

    if not isinstance(page_chunks, list):
        raise ValueError(
            f"Unexpected PyMuPDF4LLM output type: {type(page_chunks)} (expected list)."
        )

    current_section_title: str | None = None
    current_section_level: int | None = None

    emitted = 0
    for page in page_chunks:
        if not isinstance(page, dict):
            continue

        page_md = (page.get("text") or "")
        page_meta = page.get("metadata") or {}
        page_number = page_meta.get("page_number")

        boxes = page.get("page_boxes") or []
        if isinstance(boxes, list) and boxes:
            try:
                boxes = sorted(
                    (b for b in boxes if isinstance(b, dict)),
                    key=lambda b: int(b.get("index", 0)),
                )
            except Exception:
                boxes = [b for b in boxes if isinstance(b, dict)]
        else:
            boxes = []

        # Fallback: if no boxes are available, yield the whole page markdown.
        if not boxes:
            text = page_md.strip()
            yield Document(
                page_content=text,
                metadata={
                    "source": str(path),
                    "page_number": page_number,
                    "element_type": "page",
                    "layout_type": "page",
                    "section_title": current_section_title,
                    "section_level": current_section_level,
                    "layout_aware": True,
                },
            )
            emitted += 1
            continue

        for box in boxes:
            pos = box.get("pos")
            if not (
                isinstance(pos, (list, tuple))
                and len(pos) == 2
                and isinstance(pos[0], int)
                and isinstance(pos[1], int)
            ):
                continue

            start, stop = pos
            if start < 0 or stop <= start:
                continue
            if start > len(page_md):
                continue
            stop = min(stop, len(page_md))

            snippet = page_md[start:stop].strip()
            if not snippet:
                continue

            # Best-effort: update current section from any markdown heading.
            for line in snippet.splitlines():
                m = _HEADING_RE.match(line.strip())
                if m:
                    current_section_level = len(m.group(1))
                    current_section_title = m.group(2).strip()
                    break

            layout_type = box.get("class") or "text"
            if not isinstance(layout_type, str):
                layout_type = "text"

            meta: dict[str, Any] = {
                "source": str(path),
                "page_number": page_number,
                "element_type": layout_type,
                "layout_type": layout_type,
                "section_title": current_section_title,
                "section_level": current_section_level,
                "layout_aware": True,
            }
            if "bbox" in box:
                meta["bbox"] = box.get("bbox")

            yield Document(page_content=snippet, metadata=meta)
            emitted += 1

    logger.info("Finished loading %s – %d element(s) extracted.", path.name, emitted)


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
