"""
Extraction Agent — structured knowledge extraction from document chunks.

Two responsibilities:
1. **Classification** — determine the document type (financial_report,
   legal_contract, invoice, etc.) using ``model.with_structured_output()``
   with a typed Enum.
2. **Entity / fact extraction** — pull structured fields (dates, parties,
   amounts, key clauses, etc.) into a validated Pydantic model using
   ``PydanticOutputParser`` so downstream agents receive clean, typed data.

Both operations run against the first N chunks of a document (the "head")
to keep token usage low while still capturing title pages, headers, and
introductory sections that carry the richest classification signals.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────
# How many leading chunks to feed the classifier / extractor.
# Enough to capture title pages and summaries without blowing up context.
HEAD_CHUNKS = 10


# ═══════════════════════════════════════════════════════════════════
#  Schemas — these are what the LLM is forced to return
# ═══════════════════════════════════════════════════════════════════

class DocType(str, Enum):
    """Closed set of document categories the system understands."""
    FINANCIAL_REPORT = "financial_report"
    LEGAL_CONTRACT = "legal_contract"
    INVOICE = "invoice"
    RESEARCH_PAPER = "research_paper"
    POLICY_DOCUMENT = "policy_document"
    TECHNICAL_MANUAL = "technical_manual"
    EMAIL_CORRESPONDENCE = "email_correspondence"
    OTHER = "other"


class ClassificationResult(BaseModel):
    """Schema for the classification step — model.with_structured_output()."""
    doc_type: DocType = Field(
        description="The category that best describes this document."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score between 0 and 1.",
    )
    reasoning: str = Field(
        description="Brief explanation of why this category was chosen.",
    )


class ExtractedEntity(BaseModel):
    """A single named entity or key fact extracted from the document."""
    name: str = Field(description="Entity name or field label.")
    value: str = Field(description="Extracted value.")
    entity_type: str = Field(
        description=(
            "Category of the entity, e.g. 'date', 'monetary_amount', "
            "'person', 'organization', 'clause', 'metric'."
        ),
    )
    page_or_chunk: Optional[int] = Field(
        default=None,
        description="Page number or chunk index where this entity was found.",
    )


class ExtractionResult(BaseModel):
    """Full extraction output returned to the orchestrator / verifier."""
    doc_type: DocType = Field(
        description="Document classification.",
    )
    title: Optional[str] = Field(
        default=None,
        description="Document title if identifiable.",
    )
    summary: str = Field(
        description="2-4 sentence summary of the document's purpose and content.",
    )
    entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description="Structured entities and key facts extracted.",
    )
    key_dates: list[str] = Field(
        default_factory=list,
        description="Important dates mentioned (ISO-8601 preferred).",
    )
    key_parties: list[str] = Field(
        default_factory=list,
        description="People or organisations that are primary actors.",
    )
    monetary_values: list[str] = Field(
        default_factory=list,
        description="Dollar amounts, totals, or financial figures.",
    )


# ═══════════════════════════════════════════════════════════════════
#  Prompts
# ═══════════════════════════════════════════════════════════════════

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a document classification specialist.  Analyze the "
        "provided text and determine the document type.  Return your "
        "answer strictly in the requested structured format.",
    ),
    (
        "human",
        "Classify the following document content:\n\n{content}",
    ),
])


# The extraction prompt embeds the Pydantic schema description via
# {format_instructions} so the LLM knows exactly what JSON to produce.
EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert information extraction agent.  Extract all "
        "structured knowledge from the document content below.  Follow "
        "the output schema precisely.\n\n{format_instructions}",
    ),
    (
        "human",
        "Document type: {doc_type}\n\n"
        "Document content:\n\n{content}",
    ),
])


# ═══════════════════════════════════════════════════════════════════
#  Agent
# ═══════════════════════════════════════════════════════════════════

def _default_llm() -> ChatOpenAI:
    """Build the default LLM from environment variables."""
    return ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=0,          # deterministic extraction
        api_key=os.environ["OPENAI_API_KEY"],
    )


def _prepare_content(chunks: list[Document], max_chunks: int = HEAD_CHUNKS) -> str:
    """Concatenate the first *max_chunks* chunks into a single string."""
    selected = chunks[:max_chunks]
    return "\n\n".join(doc.page_content for doc in selected if doc.page_content.strip())


class ExtractionAgent:
    """
    Orchestrates classification → extraction over a list of document chunks.

    Usage::

        agent = ExtractionAgent()            # uses default OpenAI model
        result = await agent.run(chunks)     # ExtractionResult
    """

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm or _default_llm()

        # Classification: with_structured_output forces the LLM to return
        # a ClassificationResult — the cleanest pattern for enum-typed output.
        self._classifier = CLASSIFICATION_PROMPT | self.llm.with_structured_output(
            ClassificationResult
        )

        # Extraction: PydanticOutputParser validates the free-form JSON the
        # LLM produces against ExtractionResult and gives clear error messages
        # if a field is missing or has the wrong type.
        self._extraction_parser = PydanticOutputParser(pydantic_object=ExtractionResult)
        self._extractor = EXTRACTION_PROMPT | self.llm | self._extraction_parser

    # ── Public API ──────────────────────────────────────────────────

    async def classify(self, chunks: list[Document]) -> ClassificationResult:
        """Classify a document from its leading chunks."""
        content = _prepare_content(chunks)
        result: ClassificationResult = await self._classifier.ainvoke({"content": content})
        logger.info(
            "Classified as %s (confidence=%.2f): %s",
            result.doc_type.value, result.confidence, result.reasoning,
        )
        return result

    async def extract(
        self,
        chunks: list[Document],
        doc_type: DocType,
    ) -> ExtractionResult:
        """Extract structured entities given a known document type."""
        content = _prepare_content(chunks)
        result: ExtractionResult = await self._extractor.ainvoke({
            "content": content,
            "doc_type": doc_type.value,
            "format_instructions": self._extraction_parser.get_format_instructions(),
        })
        logger.info(
            "Extracted %d entities, %d dates, %d parties from %s",
            len(result.entities),
            len(result.key_dates),
            len(result.key_parties),
            doc_type.value,
        )
        return result

    async def run(self, chunks: list[Document]) -> ExtractionResult:
        """
        Full pipeline: classify → extract.

        Returns an `ExtractionResult` with the classification baked in
        so the caller gets everything in one shot.
        """
        classification = await self.classify(chunks)
        extraction = await self.extract(chunks, classification.doc_type)
        return extraction