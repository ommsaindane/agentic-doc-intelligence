"""Verifier Agent — grounding + citation enforcement.

This agent checks whether a drafted answer is actually supported by the
retrieved evidence chunks.

Implementation notes
--------------------
- Uses `model.with_structured_output()` to force a strict verification
  schema: `is_grounded`, `citations`, `rejected_claims`, etc.
- The prompt requires every claim to include explicit `source_chunk_ids`
  mapping back to your Postgres `chunks` table.

About `LLMChainFilter` / `CitationAnnotation`
--------------------------------------------
Your environment currently doesn't expose `langchain.retrievers.*` and
`langchain_core.CitationAnnotation` is not present in langchain_core==1.2.16.
So we implement an equivalent LLM-based filtering/verifying step that outputs
chunk IDs directly.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.agents.analysis_agent import AnalysisResult

load_dotenv()

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Schemas
# ═══════════════════════════════════════════════════════════════════


class VerifiedClaim(BaseModel):
	claim: str = Field(description="A concrete claim from the draft answer.")
	is_supported: bool = Field(description="Whether the evidence supports this claim.")
	source_chunk_ids: list[str] = Field(
		default_factory=list,
		description="Chunk IDs that directly support the claim.",
	)
	support_quotes: list[str] = Field(
		default_factory=list,
		description="Short direct quotes from the supporting chunks.",
	)
	notes: Optional[str] = Field(
		default=None,
		description="Short explanation if unsupported or partially supported.",
	)


class VerificationResult(BaseModel):
	is_grounded: bool
	citations: list[str] = Field(
		default_factory=list,
		description="List of chunk_ids used as citations.",
	)
	rejected_claims: list[str] = Field(
		default_factory=list,
		description="Claims from the draft that were not supported by evidence.",
	)
	verified_claims: list[VerifiedClaim] = Field(default_factory=list)
	revised_answer: str = Field(
		description=(
			"Rewritten answer that removes or qualifies unsupported claims and "
			"is fully grounded in the evidence."
		)
	)
	missing_evidence: list[str] = Field(
		default_factory=list,
		description="What evidence would be needed to support rejected claims.",
	)


# ═══════════════════════════════════════════════════════════════════
#  Prompt + evidence formatting
# ═══════════════════════════════════════════════════════════════════


def _format_evidence(
	docs: list[Document],
	*,
	max_docs: int = 16,
	max_chars_per_doc: int = 1400,
) -> str:
	"""Format evidence with chunk_id headers for citation enforcement."""
	parts: list[str] = []
	for i, d in enumerate(docs[:max_docs]):
		md = d.metadata or {}
		chunk_id = md.get("chunk_id") or md.get("id") or f"chunk_{i}"
		document_id = md.get("document_id")
		page_number = md.get("page_number")
		element_type = md.get("element_type")
		chunk_index = md.get("chunk_index")

		parts.append(
			f"[chunk_id={chunk_id} | doc_id={document_id} | page={page_number} | "
			f"element_type={element_type} | chunk_index={chunk_index}]"
		)

		text = (d.page_content or "").strip().replace("\u0000", "")
		if len(text) > max_chars_per_doc:
			text = text[:max_chars_per_doc] + "…"
		parts.append(text)
		parts.append("")
	return "\n".join(parts).strip()


VERIFIER_PROMPT = ChatPromptTemplate.from_messages(
	[
		(
			"system",
			"You are a verification-first fact checker. You MUST only accept claims that are supported "
			"by the provided evidence chunks.\n\n"
			"Rules:\n"
			"- Treat the draft answer as untrusted.\n"
			"- For each claim, decide supported/unsupported.\n"
			"- If supported, include source_chunk_ids and short direct quotes.\n"
			"- If unsupported, add it to rejected_claims and say what evidence would be needed.\n"
			"- The revised_answer must contain NO unsupported claims.\n"
			"- citations must be chunk_ids only (strings).\n",
		),
		(
			"human",
			"Question: {question}\n\n"
			"Draft answer to verify:\n{draft_answer}\n\n"
			"Draft claims (if provided, verify these explicitly):\n{draft_claims}\n\n"
			"Evidence chunks:\n{evidence}",
		),
	]
)


def _default_llm() -> ChatOpenAI:
	return ChatOpenAI(
		model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
		temperature=0,
		api_key=os.environ["OPENAI_API_KEY"],
	)


class VerifierAgent:
	"""Verifies grounding and enforces claim→chunk citations."""

	def __init__(self, llm: Optional[BaseChatModel] = None):
		self.llm = llm or _default_llm()
		self.chain = VERIFIER_PROMPT | self.llm.with_structured_output(VerificationResult)

	async def verify(
		self,
		*,
		question: str,
		draft_answer: str,
		evidence_docs: list[Document],
		draft_claims: Optional[list[str]] = None,
	) -> VerificationResult:
		"""Verify a draft answer against evidence."""

		formatted_evidence = _format_evidence(evidence_docs)
		result: VerificationResult = await self.chain.ainvoke(
			{
				"question": question,
				"draft_answer": draft_answer,
				"draft_claims": "\n".join(draft_claims or []),
				"evidence": formatted_evidence,
			}
		)

		logger.info(
			"Verifier: grounded=%s, citations=%d, rejected_claims=%d",
			result.is_grounded,
			len(result.citations),
			len(result.rejected_claims),
		)
		return result

	async def verify_analysis_result(
		self,
		*,
		question: str,
		analysis: AnalysisResult,
		evidence_docs: list[Document],
	) -> VerificationResult:
		"""Verify an `AnalysisResult` from `AnalysisAgent`."""
		claim_texts = [c.claim for c in analysis.key_claims]
		return await self.verify(
			question=question,
			draft_answer=analysis.answer,
			evidence_docs=evidence_docs,
			draft_claims=claim_texts,
		)

	@staticmethod
	def filter_supported_documents(
		*,
		evidence_docs: list[Document],
		keep_chunk_ids: list[str],
	) -> list[Document]:
		"""
		Equivalent to a chain filter: keep only evidence docs whose chunk_id
		is in the verifier-selected citations list.

		This gives you the same practical effect as `LLMChainFilter`.
		"""
		keep = set(str(x) for x in keep_chunk_ids)
		filtered: list[Document] = []
		for d in evidence_docs:
			md = d.metadata or {}
			cid = md.get("chunk_id") or md.get("id")
			if cid is not None and str(cid) in keep:
				filtered.append(d)
		return filtered

