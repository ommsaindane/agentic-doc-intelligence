from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import requests
import streamlit as st


@dataclass(frozen=True)
class ApiError(Exception):
    status_code: int
    body: str

    def __str__(self) -> str:  # pragma: no cover
        return f"API error {self.status_code}: {self.body}"


def _api_post_json(base_url: str, path: str, payload: dict[str, Any], *, timeout_s: int = 60) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    resp = requests.post(url, json=payload, timeout=timeout_s)
    if resp.status_code // 100 != 2:
        raise ApiError(resp.status_code, resp.text)
    return resp.json()


def _api_get(base_url: str, path: str, *, timeout_s: int = 30) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    resp = requests.get(url, timeout=timeout_s)
    if resp.status_code // 100 != 2:
        raise ApiError(resp.status_code, resp.text)
    return resp.json()


def _api_ingest_upload(
    base_url: str,
    *,
    filename: str,
    file_bytes: bytes,
    chunk_strategy: str,
    layout_aware: bool,
    enable_ocr: bool,
    ocr_language: str,
    timeout_s: int = 120,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/ingest/upload"
    files = {"file": (filename, file_bytes, "application/octet-stream")}
    data = {
        "chunk_strategy": chunk_strategy,
        "layout_aware": str(layout_aware).lower(),
        "enable_ocr": str(enable_ocr).lower(),
        "ocr_language": ocr_language,
    }
    resp = requests.post(url, files=files, data=data, timeout=timeout_s)
    if resp.status_code // 100 != 2:
        raise ApiError(resp.status_code, resp.text)
    return resp.json()


def _load_documents(base_url: str) -> list[dict[str, Any]]:
    obj = _api_get(base_url, "/documents")
    return list(obj.get("documents") or [])


def _get_document(base_url: str, document_id: str) -> dict[str, Any]:
    return _api_get(base_url, f"/documents/{document_id}")


def _chunks_by_ids(base_url: str, chunk_ids: list[str]) -> list[dict[str, Any]]:
    if not chunk_ids:
        return []
    return list(_api_post_json(base_url, "/chunks/by_ids", {"chunk_ids": chunk_ids}, timeout_s=60) or [])


st.set_page_config(page_title="Agentic Doc Intelligence", layout="wide")

st.title("Agentic Doc Intelligence")

# ── Session state ─────────────────────────────────────────────────
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = "http://127.0.0.1:8000"
if "documents" not in st.session_state:
    st.session_state.documents = []
if "last_ingest_id" not in st.session_state:
    st.session_state.last_ingest_id = None
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "last_query_response" not in st.session_state:
    st.session_state.last_query_response = None


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("Backend")
    st.session_state.api_base_url = st.text_input(
        "API base URL",
        value=st.session_state.api_base_url,
        help="FastAPI service base URL (e.g. http://127.0.0.1:8000)",
    )

    st.header("QA")
    debug = st.toggle("Debug", value=False, help="Show retrieved chunks and rerank info (if enabled)")

    st.header("Refresh")
    if st.button("Load documents"):
        try:
            st.session_state.documents = _load_documents(st.session_state.api_base_url)
        except Exception as exc:
            st.error(str(exc))


# ── Ingestion ─────────────────────────────────────────────────────
st.subheader("Ingestion")

col_a, col_b = st.columns([2, 1])

with col_a:
    uploaded = st.file_uploader("Upload a document", type=None)

with col_b:
    chunk_strategy = st.selectbox("Chunk strategy", ["recursive", "semantic", "section_aware"], index=0)
    layout_aware = st.checkbox("Layout-aware parsing", value=False)
    enable_ocr = st.checkbox("Enable OCR", value=False)
    ocr_language = st.text_input("OCR language", value="eng", disabled=not enable_ocr)

if uploaded is not None:
    ingest_cols = st.columns([1, 1, 2])
    with ingest_cols[0]:
        do_ingest = st.button("Ingest", type="primary")
    with ingest_cols[1]:
        auto_poll = st.checkbox("Auto-poll", value=True, help="Poll status briefly after scheduling")

    if do_ingest:
        try:
            with st.spinner("Uploading + scheduling ingestion..."):
                resp = _api_ingest_upload(
                    st.session_state.api_base_url,
                    filename=uploaded.name,
                    file_bytes=uploaded.getvalue(),
                    chunk_strategy=chunk_strategy,
                    layout_aware=layout_aware,
                    enable_ocr=enable_ocr,
                    ocr_language=ocr_language,
                )
            st.session_state.last_ingest_id = resp.get("document_id")
            st.success(f"Scheduled ingestion: {st.session_state.last_ingest_id}")
            st.session_state.documents = _load_documents(st.session_state.api_base_url)
        except Exception as exc:
            st.error(str(exc))

    # Status block
    if st.session_state.last_ingest_id:
        status_cols = st.columns([1, 1, 3])
        with status_cols[0]:
            if st.button("Refresh status"):
                try:
                    doc = _get_document(st.session_state.api_base_url, st.session_state.last_ingest_id)
                    st.info(f"Status: {doc.get('status')}")
                except Exception as exc:
                    st.error(str(exc))

        if auto_poll:
            try:
                doc = _get_document(st.session_state.api_base_url, st.session_state.last_ingest_id)
                status = str(doc.get("status") or "")
                if status.lower() in {"pending", "processing"}:
                    with st.spinner("Polling ingestion status..."):
                        for _ in range(20):
                            time.sleep(0.5)
                            doc = _get_document(st.session_state.api_base_url, st.session_state.last_ingest_id)
                            status = str(doc.get("status") or "")
                            if status.lower() not in {"pending", "processing"}:
                                break
                st.caption(f"Last ingestion status: {status}")
            except Exception as exc:
                st.warning(f"Status polling failed: {exc}")

# Document table
if st.session_state.documents:
    st.caption("Documents")
    st.dataframe(st.session_state.documents, use_container_width=True, hide_index=True)


# ── QA ────────────────────────────────────────────────────────────
st.subheader("QA")

docs = st.session_state.documents or []
options = [d.get("document_id") for d in docs if d.get("document_id")]

selected_doc_ids = st.multiselect(
    "Scope to document IDs (optional)",
    options=options,
    default=[st.session_state.last_ingest_id] if st.session_state.last_ingest_id in options else [],
)

query_text = st.text_area("Question", value=st.session_state.last_query or "", height=80)

ask = st.button("Ask", type="primary")
if ask:
    st.session_state.last_query = query_text
    try:
        with st.spinner("Running QA..."):
            resp = _api_post_json(
                st.session_state.api_base_url,
                "/query",
                {
                    "query": query_text,
                    "document_ids": selected_doc_ids,
                    "max_rounds": 2,
                    "semantic_k": 10,
                    "bm25_k": 10,
                    "debug": bool(debug),
                },
                timeout_s=180,
            )
        st.session_state.last_query_response = resp
    except Exception as exc:
        st.error(str(exc))

resp = st.session_state.last_query_response
if resp:
    st.markdown("**Answer**")
    st.write(resp.get("answer") or "")

    st.markdown("**Grounding**")
    st.write({
        "is_grounded": resp.get("is_grounded"),
        "missing_evidence": resp.get("missing_evidence"),
        "rejected_claims": resp.get("rejected_claims"),
    })

    citations = list(resp.get("citations") or [])
    if citations:
        st.markdown("**Citations**")
        st.write(citations)

        try:
            chunks = _chunks_by_ids(st.session_state.api_base_url, citations)
        except Exception as exc:
            st.error(f"Failed to fetch chunk snippets: {exc}")
            chunks = []

        if chunks:
            st.markdown("**Source snippets**")
            # Preserve citation order
            by_id = {c.get("chunk_id"): c for c in chunks}
            for cid in citations:
                c = by_id.get(cid)
                if not c:
                    continue
                page = c.get("page_number")
                title = f"{cid}" + (f" (page {page})" if page is not None else "")
                with st.expander(title):
                    st.caption(
                        {
                            "document_id": c.get("document_id"),
                            "element_type": c.get("element_type"),
                            "chunk_index": c.get("chunk_index"),
                            "section_title": c.get("section_title"),
                            "layout_type": c.get("layout_type"),
                        }
                    )
                    content = c.get("content") or ""
                    st.text(content[:1500] + ("\n…" if len(content) > 1500 else ""))

    if debug and resp.get("debug"):
        dbg = resp.get("debug") or {}
        with st.expander("Debug: retrieved chunks"):
            st.write({"reranker_enabled": dbg.get("reranker_enabled"), "retrieval_round": dbg.get("retrieval_round")})

            bundles = dbg.get("evidence_bundles") or []
            rows: list[dict[str, Any]] = []
            for b in bundles:
                doc_id = b.get("document_id")
                for d in b.get("docs") or []:
                    md = (d.get("metadata") or {})
                    rows.append(
                        {
                            "document_id": doc_id,
                            "rank": md.get("_retrieval_rank"),
                            "chunk_id": md.get("chunk_id"),
                            "page": md.get("page_number"),
                            "hybrid_score": md.get("_hybrid_score"),
                            "rerank_score": md.get("_rerank_score"),
                            "preview": (d.get("page_content") or "")[:160].replace("\n", " "),
                        }
                    )

            if rows:
                if not dbg.get("reranker_enabled"):
                    # Hide rerank_score column noise when reranker isn't on.
                    for r in rows:
                        r.pop("rerank_score", None)
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.write("No debug retrieval bundles available.")
