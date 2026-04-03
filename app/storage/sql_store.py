# storage/sql_store.py

import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Sequence

from dotenv import load_dotenv
from sqlalchemy import delete, select, update
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .schemas import (
    AgentRunModel,
    AgentRunStatus,
    ChunkModel,
    DocumentModel,
    DocumentStatus,
)

load_dotenv()

# ── Engine setup ────────────────────────────────────────────────────
# The .env stores a sync URL (postgresql://…). We swap the scheme to
# postgresql+asyncpg:// because async SQLAlchemy needs the asyncpg driver.
# This is the only place the conversion happens.
_sync_url: str = os.environ["DATABASE_URL"]
_async_url: str = _sync_url.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = create_async_engine(
    _async_url,
    echo=False,
    pool_size=10,       # tune based on expected concurrency
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def ensure_schema() -> None:
    """Best-effort, idempotent schema shim for legacy databases.

    Why this exists
    --------------
    This repo may be pointed at a Postgres DB that already contains legacy
    `documents`/`chunks` tables. SQLAlchemy's `create_all()` won't add columns
    to existing tables, so we ensure the minimal set of columns this code
    requires using `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`.

    This does NOT drop data.
    """
    async with engine.begin() as conn:
        # Enum used by DocumentModel.status.
        # Some legacy DBs may already have a `documentstatus` type with UPPERCASE
        # labels. If it's unused, we recreate it with lowercase labels to match
        # the application's `DocumentStatus` values.
        await conn.execute(
            text(
                """
                DO $$
                DECLARE
                    type_labels text[];
                    type_in_use bool;
                BEGIN
                    IF EXISTS (SELECT 1 FROM pg_type WHERE typname = 'documentstatus') THEN
                        SELECT COALESCE(array_agg(e.enumlabel ORDER BY e.enumsortorder), ARRAY[]::text[])
                        INTO type_labels
                        FROM pg_type t
                        JOIN pg_enum e ON t.oid = e.enumtypid
                        WHERE t.typname = 'documentstatus';

                        SELECT EXISTS (
                            SELECT 1
                            FROM information_schema.columns
                            WHERE table_schema = 'public' AND udt_name = 'documentstatus'
                        ) INTO type_in_use;

                        IF (NOT type_in_use) AND (type_labels = ARRAY['PENDING','PROCESSING','COMPLETE','FAILED']) THEN
                            DROP TYPE documentstatus;
                        END IF;
                    END IF;

                    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'documentstatus') THEN
                        CREATE TYPE documentstatus AS ENUM ('pending','processing','complete','failed');
                    END IF;
                END $$;
                """
            )
        )

        # Documents table: add columns used by the application.
        await conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS filename varchar(512);"))
        await conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_path varchar(1024);"))
        await conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS doc_type varchar(128);"))
        await conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS updated_at timestamptz;"))
        await conn.execute(
            text(
                """
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS status documentstatus NOT NULL DEFAULT 'pending';
                """
            )
        )

        # Chunks table: add columns used for citations/filtering.
        await conn.execute(text("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS start_index integer;"))
        await conn.execute(text("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS page_number integer;"))
        await conn.execute(text("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS element_type varchar(64);"))
        await conn.execute(text("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS section_title varchar(512);"))
        await conn.execute(text("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS section_level integer;"))
        await conn.execute(text("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS layout_type varchar(64);"))
        await conn.execute(text("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS pinecone_id varchar(256);"))


class SQLStore:
    """
    Service class wrapping all Postgres operations.

    Using a class rather than bare functions lets you inject
    a session for testing — you can pass a mock session in tests
    without touching the real database.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    # ── Documents ───────────────────────────────────────────────────

    async def create_document(
        self,
        filename: str,
        file_path: str,
        doc_metadata: Optional[dict] = None,
    ) -> DocumentModel:
        doc = DocumentModel(
            id=uuid.uuid4(),
            # Legacy required fields (safe defaults).
            source_type="file",
            source_uri=file_path,
            title=filename,
            raw_text="",
            filename=filename,
            file_path=file_path,
            status=DocumentStatus.PENDING,
            doc_metadata=doc_metadata or {},
        )
        self.session.add(doc)
        await self.session.commit()
        await self.session.refresh(doc)
        return doc

    async def get_document(self, document_id: uuid.UUID) -> Optional[DocumentModel]:
        result = await self.session.execute(
            select(DocumentModel).where(DocumentModel.id == document_id)
        )
        return result.scalar_one_or_none()

    async def list_documents(
        self,
        status: Optional[DocumentStatus] = None,
    ) -> Sequence[DocumentModel]:
        stmt = select(DocumentModel).order_by(DocumentModel.created_at.desc())
        if status is not None:
            stmt = stmt.where(DocumentModel.status == status)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def update_document_status(
        self,
        document_id: uuid.UUID,
        status: DocumentStatus,
        doc_type: Optional[str] = None,
        doc_metadata: Optional[dict] = None,
    ) -> None:
        values: dict = {"status": status, "updated_at": datetime.utcnow()}
        if doc_type is not None:
            values["doc_type"] = doc_type
        if doc_metadata is not None:
            values["doc_metadata"] = doc_metadata

        await self.session.execute(
            update(DocumentModel)
            .where(DocumentModel.id == document_id)
            .values(**values)
        )
        await self.session.commit()

    async def delete_document(self, document_id: uuid.UUID) -> None:
        """Delete a document and its chunks (cascaded by the ORM relationship)."""
        await self.session.execute(
            delete(DocumentModel).where(DocumentModel.id == document_id)
        )
        await self.session.commit()

    # ── Chunks ──────────────────────────────────────────────────────

    async def save_chunks(self, chunks: list[dict]) -> list[ChunkModel]:
        """
        Bulk insert chunks for a document.  Always use bulk insert rather
        than inserting one chunk at a time — for a 200-page document you
        might have 400+ chunks, and individual inserts would be very slow.
        """
        models = [ChunkModel(**chunk) for chunk in chunks]
        self.session.add_all(models)
        await self.session.commit()
        return models

    async def get_chunks_by_ids(self, chunk_ids: list[str]) -> Sequence[ChunkModel]:
        """
        Fetch chunks by their Pinecone IDs.

        This is what your Retrieval Agent calls after getting results from
        Pinecone.  Pinecone returns IDs and scores; this fetches the actual
        content and metadata so agents have full context for reasoning and
        the verifier can check citations.
        """
        result = await self.session.execute(
            select(ChunkModel).where(ChunkModel.pinecone_id.in_(chunk_ids))
        )
        return result.scalars().all()

    async def get_chunks_by_document(
        self,
        document_id: uuid.UUID,
    ) -> Sequence[ChunkModel]:
        """Return all chunks for a document, ordered by position."""
        result = await self.session.execute(
            select(ChunkModel)
            .where(ChunkModel.document_id == document_id)
            .order_by(ChunkModel.chunk_index)
        )
        return result.scalars().all()

    # ── Agent runs ──────────────────────────────────────────────────

    async def create_agent_run(
        self,
        workflow: str,
        document_id: Optional[uuid.UUID] = None,
        query: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> AgentRunModel:
        run = AgentRunModel(
            id=uuid.uuid4(),
            workflow=workflow,
            document_id=document_id,
            input_query=query,
            thread_id=thread_id,
            status=AgentRunStatus.RUNNING,
        )
        self.session.add(run)
        await self.session.commit()
        await self.session.refresh(run)
        return run

    async def complete_agent_run(
        self,
        run_id: uuid.UUID,
        output: dict,
        total_tokens: int = 0,
    ) -> None:
        await self.session.execute(
            update(AgentRunModel)
            .where(AgentRunModel.id == run_id)
            .values(
                status=AgentRunStatus.SUCCESS,
                output=output,
                total_tokens=total_tokens,
                completed_at=datetime.utcnow(),
            )
        )
        await self.session.commit()

    async def fail_agent_run(self, run_id: uuid.UUID, error: str) -> None:
        await self.session.execute(
            update(AgentRunModel)
            .where(AgentRunModel.id == run_id)
            .values(
                status=AgentRunStatus.FAILED,
                error_message=error,
                completed_at=datetime.utcnow(),
            )
        )
        await self.session.commit()


# ── FastAPI dependency injection ────────────────────────────────────

@asynccontextmanager
async def get_db():
    """Yields a SQLStore wrapped around a managed session."""
    async with AsyncSessionLocal() as session:
        try:
            yield SQLStore(session)
        except Exception:
            await session.rollback()
            raise