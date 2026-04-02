---
name: Agentic Doc Intelligence
description: Project-wide rules for FastAPI + LangGraph doc ingestion/QA (strict + deterministic; PyMuPDF parsing; no contextual compression)
applyTo: '**'
---

# Agentic Doc Intelligence — Workspace Instructions

Use these instructions when generating code, refactoring, reviewing changes, or answering implementation questions in this workspace.

## Project context (keep it short)

- This is a **FastAPI** service with two primary flows:
	- **Ingestion**: load local documents → chunk → classify/extract → store chunks in **Postgres** and embeddings/filters in **Pinecone**.
	- **QA**: retrieve evidence (semantic + keyword) → analyze → verify grounding → return grounded answer with citations.
- Workflows are orchestrated with **LangGraph**.

# Coding Guidelines.

* Use Python only 
* Use as many tools and libraries as needed (e.g., Langchain(important), PyPDF2, OpenAI API, etc.) 
* Search the web using MCP if you need to find libraries or solutions 
* Prioritize use of libraries over custom code

## Non‑negotiable behavior (user preferences)

- Prefer **strict and deterministic runtime behavior**.
	- Use `temperature=0` for LLM calls.
	- Use structured outputs (Pydantic models / `with_structured_output`) whenever possible.
- **Fail fast** on missing or invalid critical configuration.
	- Missing `OPENAI_API_KEY` or inability to access the configured model must be treated as an error (do not “limp along”).
- Avoid operational “fallbacks” and silent degradation:
	- No default inputs that hide missing user parameters.
	- No “OCR fallback” paths.
	- No silent exception swallowing.
	- No retry/continue-on-fail loops that hide errors.
- Answers may use “Not Available” only when **evidence is insufficient**, and must explicitly state what evidence is missing.


## Architectural rules

### API layer

- FastAPI endpoints are the user-facing contract.
- Background ingestion must:
	- create/track a stable `document_id`
	- set document status transitions deterministically (`pending → processing → complete` or `failed`)
	- record failures clearly (including error messages) and stop.

### Workflow layer (LangGraph)

- Workflows should be explicit and serializable:
	- state should contain JSON-serializable data where possible.
	- avoid storing live service objects in state except via closures.
- QA workflow behavior:
	- fan-out retrieval per document scope
	- merge and dedupe evidence by `chunk_id`
	- analyze → verify
	- optional bounded re-retrieval rounds based on verifier grounding outcome.

### Agent layer (LLM-backed)

- **Extraction agent**
	- Classify a document to a closed `DocType` enum.
	- Extract structured entities/facts into a typed Pydantic model.
- **Analysis agent**
	- Produce an evidence-grounded answer.
	- Every key claim must map to at least one `chunk_id` citation.
	- Parallel sub-analyses are OK if deterministic and typed.
- **Verifier agent**
	- Treat drafts as untrusted.
	- Enforce grounding: unsupported claims must be rejected and removed/qualified.
	- Output must include `is_grounded`, `citations` (chunk IDs), `rejected_claims`, `missing_evidence`, and a `revised_answer`.

### Retrieval layer

- Keep hybrid retrieval, but without any LLM compression:
	- Pinecone semantic hits (with metadata filters)
	- BM25 hits over Postgres chunks
	- deterministic merge and scoring
- Ensure retrieval always preserves citation metadata: `chunk_id`, `document_id`, `page_number`, `element_type`, `chunk_index`.

### Storage layer

- Postgres stores:
	- documents (status, doc_type, metadata)
	- chunks (text, offsets, page_number, element_type, metadata)
	- agent_runs (workflow audit trail)
- Pinecone stores:
	- vectors keyed by `chunk_id` with minimal metadata needed for filtering.

## Error handling rules

- Prefer raising explicit exceptions with clear messages.
- Never catch exceptions just to continue execution.
- If you must catch exceptions to attach context, re-raise and ensure the failure is recorded (e.g., set document status to `failed`).

## Configuration rules

- Use `.env` via `python-dotenv` for local dev.
- Required configuration must be validated early:
	- `OPENAI_API_KEY`
	- `DATABASE_URL`
	- `PINECONE_API_KEY`
	- `PINECONE_INDEX_NAME`
- If a refactor changes required env vars or supported file types, update README.

## When making changes

- Keep changes **minimal and targeted**.
- Do not introduce new frameworks or new UX/APIs unless explicitly requested.
- If you remove dependencies (Unstructured/OCR stack, contextual compression), also:
	- delete dead code paths
	- remove unused env vars/options
	- update `pyproject.toml` and README accordingly
	- ensure ingestion and QA still function end-to-end.

## Quick review checklist (use during PR/code review)

- Does the change preserve deterministic behavior (`temperature=0`, typed outputs)?
- Does it fail fast on missing OpenAI config (no silent degrade)?
- Is contextual compression fully removed (no prompts/parsers/flags)?
- Is document parsing implemented via PyMuPDF (no Unstructured/pytesseract/pdfplumber usage)?
- Are citations traceable to `chunk_id` and (when possible) page numbers?

---

# .env configuration 

The .env file has been set up with the following variables:

```env
OPENAI_API_KEY=""
OPENAI_MODEL=gpt-5-mini
PINECONE_API_KEY=
PINECONE_INDEX_NAME=agentic-doc-intel
DATABASE_URL=
```

The model has following limits:

TPM=60000
RPM=3
RPD=200
TPD=200000

---