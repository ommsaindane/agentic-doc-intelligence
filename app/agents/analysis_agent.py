"""Analysis Agent — evidence-grounded reasoning over retrieved chunks.

This module is intentionally LCEL-first:

	prompt | model | output_parser

It also uses `RunnableParallel` to run multiple sub-analyses concurrently
(summary + key claims + contradictions), then merges the results into a
final, citation-backed analytical output.

Input assumptions
-----------------
The caller provides a list of `langchain_core.documents.Document` objects
as evidence. Each Document should include metadata for citations:

	chunk_id, document_id, page_number, element_type, chunk_index

This aligns with the metadata written to Pinecone in `VectorStore` and
stored in Postgres via `ChunkModel`.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Output schemas (strict, typed)
# ═══════════════════════════════════════════════════════════════════


class Citation(BaseModel):
	"""Evidence pointer for verifiable output."""

	chunk_id: str
	document_id: Optional[str] = None
	page_number: Optional[int] = None
	quote: str = Field(
		description="Short quote/span from evidence supporting the claim."
	)


class KeyClaim(BaseModel):
	claim: str
	support_chunk_ids: list[str] = Field(default_factory=list)
	confidence: float = Field(ge=0.0, le=1.0)


class Contradiction(BaseModel):
	description: str
	chunk_ids: list[str] = Field(default_factory=list)


class SubSummary(BaseModel):
	summary: str
	salient_points: list[str] = Field(default_factory=list)


class SubClaims(BaseModel):
	claims: list[KeyClaim] = Field(default_factory=list)


class SubContradictions(BaseModel):
	contradictions: list[Contradiction] = Field(default_factory=list)


class AnalysisResult(BaseModel):
	"""Final analytical answer produced by the analysis agent."""

	answer: str = Field(description="Direct answer to the user question.")
	reasoning: str = Field(
		description="Concise explanation grounded in the provided evidence."
	)
	key_claims: list[KeyClaim] = Field(default_factory=list)
	contradictions: list[Contradiction] = Field(default_factory=list)
	citations: list[Citation] = Field(
		default_factory=list,
		description="Citations for the answer/claims (must reference chunk_id).",
	)
	limits: list[str] = Field(
		default_factory=list,
		description="Known gaps/uncertainties due to missing evidence.",
	)


# ═══════════════════════════════════════════════════════════════════
#  Evidence formatting
# ═══════════════════════════════════════════════════════════════════


def _format_evidence(
	docs: list[Document],
	*,
	max_docs: int = 12,
	max_chars_per_doc: int = 1200,
) -> str:
	"""Render evidence into a deterministic, citation-friendly string."""
	lines: list[str] = []
	for i, d in enumerate(docs[:max_docs]):
		md = d.metadata or {}
		chunk_id = md.get("chunk_id") or md.get("id") or f"chunk_{i}"
		document_id = md.get("document_id")
		page_number = md.get("page_number")
		element_type = md.get("element_type")
		chunk_index = md.get("chunk_index")

		header = (
			f"[chunk_id={chunk_id} | doc_id={document_id} | page={page_number} | "
			f"element_type={element_type} | chunk_index={chunk_index}]"
		)
		text = (d.page_content or "").strip().replace("\u0000", "")
		if len(text) > max_chars_per_doc:
			text = text[:max_chars_per_doc] + "…"
		lines.append(header)
		lines.append(text)
		lines.append("")
	return "\n".join(lines).strip()


# ═══════════════════════════════════════════════════════════════════
#  Prompts
# ═══════════════════════════════════════════════════════════════════


SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
	[
		(
			"system",
			"You are a careful analyst. Summarize the evidence in a way that is "
			"useful for downstream reasoning. Only use provided evidence.\n\n"
			"{format_instructions}",
		),
		(
			"human",
			"Question: {question}\n\nEvidence:\n{evidence}",
		),
	]
)


CLAIMS_PROMPT = ChatPromptTemplate.from_messages(
	[
		(
			"system",
			"Extract key factual/analytical claims from the evidence that help answer "
			"the question. Each claim MUST include supporting chunk_ids. Only use the "
			"evidence provided.\n\n{format_instructions}",
		),
		(
			"human",
			"Question: {question}\n\nEvidence:\n{evidence}",
		),
	]
)


CONTRADICTIONS_PROMPT = ChatPromptTemplate.from_messages(
	[
		(
			"system",
			"Identify contradictions or tensions within the evidence relevant to the question. "
			"If none, return an empty list. Use chunk_ids.\n\n{format_instructions}",
		),
		(
			"human",
			"Question: {question}\n\nEvidence:\n{evidence}",
		),
	]
)


SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
	[
		(
			"system",
			"You are a verification-first analysis model. Answer the question using ONLY the evidence. "
			"If the evidence is insufficient, state what is missing.\n\n"
			"Rules:\n"
			"- Every key claim must be backed by at least one chunk_id.\n"
			"- Citations must include chunk_id and a short supporting quote.\n"
			"- Do not invent facts, numbers, or named entities not present in evidence.\n\n"
			"{format_instructions}",
		),
		(
			"human",
			"Question: {question}\n\n"
			"Evidence:\n{evidence}\n\n"
			"Summary:\n{summary}\n\n"
			"Key claims:\n{claims}\n\n"
			"Contradictions:\n{contradictions}",
		),
	]
)


def _default_llm() -> ChatOpenAI:
	return ChatOpenAI(
		model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
		temperature=0,
		api_key=os.environ["OPENAI_API_KEY"],
	)


# ═══════════════════════════════════════════════════════════════════
#  Chain builder
# ═══════════════════════════════════════════════════════════════════


def build_analysis_chain(llm: BaseChatModel):
	"""Return the full LCEL runnable for analysis."""

	summary_parser = PydanticOutputParser(pydantic_object=SubSummary)
	claims_parser = PydanticOutputParser(pydantic_object=SubClaims)
	contradictions_parser = PydanticOutputParser(pydantic_object=SubContradictions)
	final_parser = PydanticOutputParser(pydantic_object=AnalysisResult)

	summary_chain = SUMMARY_PROMPT | llm | summary_parser
	claims_chain = CLAIMS_PROMPT | llm | claims_parser
	contradictions_chain = CONTRADICTIONS_PROMPT | llm | contradictions_parser

	parallel = RunnableParallel(
		summary=summary_chain,
		claims=claims_chain,
		contradictions=contradictions_chain,
	)

	def _merge(inputs: dict[str, Any]) -> dict[str, Any]:
		# inputs includes: question, evidence, summary, claims, contradictions
		return {
			"question": inputs["question"],
			"evidence": inputs["evidence"],
			"summary": inputs["summary"].summary,
			"claims": [c.model_dump() for c in inputs["claims"].claims],
			"contradictions": [c.model_dump() for c in inputs["contradictions"].contradictions],
			"format_instructions": final_parser.get_format_instructions(),
		}

	synthesis_chain = SYNTHESIS_PROMPT | llm | final_parser

	# Prepare inputs → run parallel subtasks → merge → synthesize
	chain = (
		RunnableLambda(lambda x: {
			"question": x["question"],
			"evidence": _format_evidence(x["evidence_docs"]),
			"format_instructions": "",  # per-subchain instructions injected below
		})
		| RunnableLambda(
			lambda x: {
				"question": x["question"],
				"evidence": x["evidence"],
				"summary": {
					"question": x["question"],
					"evidence": x["evidence"],
					"format_instructions": summary_parser.get_format_instructions(),
				},
				"claims": {
					"question": x["question"],
					"evidence": x["evidence"],
					"format_instructions": claims_parser.get_format_instructions(),
				},
				"contradictions": {
					"question": x["question"],
					"evidence": x["evidence"],
					"format_instructions": contradictions_parser.get_format_instructions(),
				},
			}
		)
		| RunnableParallel(
			question=RunnableLambda(lambda x: x["question"]),
			evidence=RunnableLambda(lambda x: x["evidence"]),
			summary=RunnableLambda(lambda x: x["summary"]) | summary_chain,
			claims=RunnableLambda(lambda x: x["claims"]) | claims_chain,
			contradictions=RunnableLambda(lambda x: x["contradictions"]) | contradictions_chain,
		)
		| RunnableLambda(_merge)
		| synthesis_chain
	)

	return chain


class AnalysisAgent:
	"""Thin wrapper around the LCEL analysis chain."""

	def __init__(self, llm: Optional[BaseChatModel] = None):
		self.llm = llm or _default_llm()
		self.chain = build_analysis_chain(self.llm)

	async def run(self, *, question: str, evidence_docs: list[Document]) -> AnalysisResult:
		logger.info("AnalysisAgent running on %d evidence docs", len(evidence_docs))
		return await self.chain.ainvoke({"question": question, "evidence_docs": evidence_docs})

