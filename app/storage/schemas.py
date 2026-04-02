# storage/schemas.py

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional
from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, JSON, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship
from pydantic import BaseModel, Field

Base = declarative_base()


# ── Enums ───────────────────────────────────────────────────────────
# Using Python enums that are shared between SQLAlchemy and Pydantic
# keeps your valid values in one place rather than duplicated as strings.

class DocumentStatus(str, Enum):
    PENDING = "pending"       # uploaded but not yet processed
    PROCESSING = "processing" # ingestion pipeline is running
    COMPLETE = "complete"     # fully ingested, available for QA
    FAILED = "failed"         # ingestion failed, check agent_runs for why


class AgentRunStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


# ── SQLAlchemy ORM Models ──────────────────────────────────────────
# These define the actual Postgres tables. Notice that we store
    # doc_metadata as a JSON column — this is intentional.
# documents have wildly varying metadata (some have authors, some have
# page counts, some have neither), so a flexible JSON column beats
# adding 15 nullable columns you'll mostly never use.

class DocumentModel(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Legacy schema columns (may be required NOT NULL in existing DBs).
    source_type = Column(String(128), nullable=True)
    source_uri = Column(String(2048), nullable=True)
    title = Column(String(512), nullable=True)
    mime_type = Column(String(256), nullable=True)
    sha256 = Column(String(128), nullable=True)
    raw_text = Column(Text, nullable=True)
    # NOTE: This repository may already have a legacy `documents` table.
    # We keep these columns nullable so we can coexist with older rows.
    filename = Column(String(512), nullable=True)
    file_path = Column(String(1024), nullable=True)

    # This is the classification your Extraction Agent produces —
    # e.g. "financial_report", "legal_contract", "invoice"
    doc_type = Column(String(128), nullable=True)

    # Stored as a Postgres enum `documentstatus` (created/ensured at startup).
    # IMPORTANT: Persist enum *values* ("pending") not enum *names* ("PENDING").
    status = Column(
        SAEnum(
            DocumentStatus,
            name="documentstatus",
            values_callable=lambda enum: [e.value for e in enum],
        ),
        default=DocumentStatus.PENDING,
        nullable=False,
    )

    # Flexible metadata: page count, author, detected language, etc.
    # Store document-level metadata here (e.g., source).
    # The existing DB schema already uses a `metadata` JSON column.
    doc_metadata = Column("metadata", JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=True)

    # Relationship lets you do document.chunks in Python
    # without writing a JOIN manually
    chunks = relationship("ChunkModel", back_populates="document", cascade="all, delete-orphan")


class ChunkModel(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # This is the foreign key that links a chunk back to its parent document.
    # This is the critical bridge for citation: Pinecone stores chunk.id,
    # and from that ID you can always find the source document.
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)

    # Legacy schema uses column name `text`.
    content = Column("text", Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)  # position within the document

    # start_index comes from add_start_index=True in RecursiveCharacterTextSplitter.
    # It records the character offset in the original text where this chunk began,
    # which enables precise citation like "page 4, paragraph 2" if needed.
    start_index = Column(Integer, nullable=True)

    # Page number from parser metadata.
    # Essential for human-readable citations ("see page 12").
    page_number = Column(Integer, nullable=True)

    # What the parser tells us about this chunk's role in the document:
    # "NarrativeText", "Table", "Title", "ListItem", etc.
    # Your Analysis Agent should treat Table chunks very differently from NarrativeText.
    element_type = Column(String(64), nullable=True)

    # Pinecone vector ID — the same UUID stored as the Pinecone record ID.
    # When Pinecone returns results, you use this to fetch the full chunk from Postgres.
    # Keeping them identical (not just linked) makes lookups trivial.
    pinecone_id = Column(String(256), nullable=True)

    # Legacy schema already uses a `metadata` JSON column.
    chunk_metadata = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    document = relationship("DocumentModel", back_populates="chunks")


class AgentRunModel(Base):
    __tablename__ = "agent_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Which LangGraph workflow triggered this run
    workflow = Column(String(128), nullable=False)  # "ingestion" or "qa"

    # The LangGraph thread_id — this connects an agent_run to a full
    # LangGraph checkpoint trace. If something goes wrong, you can
    # replay the exact graph state using this ID.
    thread_id = Column(String(256), nullable=True)

    status = Column(SAEnum(AgentRunStatus), default=AgentRunStatus.RUNNING)

    # For ingestion runs, which document was being processed
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True)

    # The user's original question (for QA runs)
    input_query = Column(Text, nullable=True)

    # Store the final structured output — the verifier's approved response
    # including citations. JSON is fine here; you're not querying into this.
    output = Column(JSON, nullable=True)

    # If status is FAILED, what went wrong
    error_message = Column(Text, nullable=True)

    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # How many tokens did this run consume? Critical for cost tracking.
    total_tokens = Column(Integer, nullable=True)


# ── Pydantic Models ────────────────────────────────────────────────
# These are what your FastAPI endpoints and agents work with at runtime.
# They validate incoming data and shape outgoing responses.
# Notice they mirror the ORM models but are independent — this is intentional.
# Your API layer shouldn't be tightly coupled to your database layer.


class DocumentCreate(BaseModel):
    """Inbound payload when a user uploads / registers a document."""
    filename: str
    file_path: str
    doc_metadata: dict = Field(default_factory=dict)


class DocumentOut(BaseModel):
    """Outbound representation of a document."""
    id: str
    filename: str
    file_path: str
    doc_type: Optional[str] = None
    status: DocumentStatus
    doc_metadata: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class ChunkOut(BaseModel):
    """Outbound representation of a single chunk."""
    id: str
    document_id: str
    content: str
    chunk_index: int
    element_type: Optional[str] = None
    start_index: Optional[int] = None
    page_number: Optional[int] = None
    chunk_metadata: dict = Field(default_factory=dict)

    model_config = {"from_attributes": True}


class AgentRunOut(BaseModel):
    """Outbound representation of an agent run (ingestion or QA)."""
    id: str
    workflow: str
    thread_id: Optional[str] = None
    status: AgentRunStatus
    document_id: Optional[str] = None
    input_query: Optional[str] = None
    output: Optional[dict] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_tokens: Optional[int] = None

    model_config = {"from_attributes": True}