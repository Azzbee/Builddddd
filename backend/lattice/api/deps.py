"""Dependency container and auth.

The container holds the wired services. It defaults to in-memory backends so the
API boots and is fully exercised in tests without Neo4j/Postgres/LLM keys. In
production, ``build_container`` swaps in Neo4j/pgvector/LiteLLM based on config.
Tests override the container via :func:`set_container`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fastapi import Header, HTTPException, status

from lattice.config import Settings, get_settings
from lattice.core.llm import LiteLLMClient, LLMClient
from lattice.db.aux_stores import (
    DigestStore,
    InMemoryDigestStore,
    InMemoryWatchStore,
    WatchStore,
)
from lattice.db.blobs import BlobStore, InMemoryBlobStore
from lattice.db.cards import CorpusStore, InMemoryCardStore, InMemoryJobStore, JobStore
from lattice.db.vector import InMemoryVectorStore, VectorStore
from lattice.embeddings.chunks import ChunkEmbedder
from lattice.graph.store import FakeGraphStore, GraphStore
from lattice.ingestion.grobid_client import GrobidClient
from lattice.ingestion.service import IngestionService, Parser
from lattice.rag.agent import RagAgent
from lattice.rag.tools import Toolbox


@dataclass
class Container:
    settings: Settings
    llm: LLMClient
    vectors: VectorStore
    cards: CorpusStore
    jobs: JobStore
    graph: GraphStore
    chunk_embedder: ChunkEmbedder
    ingestion: IngestionService
    watch: WatchStore
    digests: DigestStore
    blobs: BlobStore

    def make_agent(self) -> RagAgent:
        toolbox = Toolbox(
            graph=self.graph,
            vectors=self.vectors,
            cards=self.cards,
            embedder=self.chunk_embedder,
            workspace_id=self.settings.workspace_id,
            top_k=self.settings.rag.top_k_chunks,
        )
        return RagAgent(self.llm, toolbox, self.settings.rag)


# Shared persistent resources (created once at startup when settings.persistent).
_persist_pool: object | None = None
_persist_graph: GraphStore | None = None


def build_container(settings: Settings | None = None, workspace_id: str | None = None) -> Container:
    settings = settings or get_settings()
    if workspace_id is not None and workspace_id != settings.workspace_id:
        settings = settings.model_copy(update={"workspace_id": workspace_id})
    ws = settings.workspace_id
    chunk_embedder = ChunkEmbedder(dim=settings.embedding.chunk_dim)

    vectors: VectorStore
    cards: CorpusStore
    jobs: JobStore
    graph: GraphStore
    watch: WatchStore
    digests: DigestStore
    blobs: BlobStore
    reader = None
    if settings.persistent and _persist_pool is not None and _persist_graph is not None:
        from lattice.db.blobs import PgBlobStore
        from lattice.db.pg_stores import (
            PgCardStore,
            PgDigestStore,
            PgJobStore,
            PgWatchStore,
        )
        from lattice.db.vector import PgVectorStore
        from lattice.graph.reader import Neo4jGraphReader

        vectors = PgVectorStore(_persist_pool, ws, settings.rag.hybrid_vector_weight)
        cards = PgCardStore(_persist_pool, ws)
        jobs = PgJobStore(_persist_pool, ws)
        watch = PgWatchStore(_persist_pool, ws)
        digests = PgDigestStore(_persist_pool, ws)
        blobs = PgBlobStore(_persist_pool, ws)
        graph = _persist_graph
        reader = Neo4jGraphReader(_persist_graph)
    else:
        vectors = InMemoryVectorStore(hybrid_vector_weight=settings.rag.hybrid_vector_weight)
        cards = InMemoryCardStore()
        jobs = InMemoryJobStore()
        watch = InMemoryWatchStore()
        digests = InMemoryDigestStore()
        blobs = InMemoryBlobStore()
        graph = FakeGraphStore()

    llm: LLMClient
    parser: Parser
    enricher = None
    text_extractor: Callable[[bytes], str] | None = None
    if settings.demo_mode:
        from lattice.demo import DemoLLM, DemoParser

        llm = DemoLLM()
        parser = DemoParser()
        text_extractor = lambda data: "Synthetic demo text. " * 60  # noqa: E731
    else:
        from lattice.enrichment.service import CompositeEnricher

        llm = LiteLLMClient()
        parser = GrobidClient(settings.grobid)
        # Free SPECTER2 vectors + reference graph from S2/OpenAlex (best-effort).
        enricher = CompositeEnricher(settings.enrichment)
    ingestion = IngestionService(
        settings=settings,
        llm=llm,
        parser=parser,
        vectors=vectors,
        cards=cards,
        blobs=blobs,
        graph=graph,
        reader=reader,
        enricher=enricher,
        chunk_embedder=chunk_embedder,
        text_extractor=text_extractor,
    )
    return Container(
        settings=settings,
        llm=llm,
        vectors=vectors,
        cards=cards,
        jobs=jobs,
        graph=graph,
        chunk_embedder=chunk_embedder,
        ingestion=ingestion,
        watch=watch,
        digests=digests,
        blobs=blobs,
    )


async def init_persistence(settings: Settings | None = None) -> None:
    """Create the shared Postgres pool + Neo4j store and apply schemas. Idempotent."""
    global _persist_pool, _persist_graph
    settings = settings or get_settings()
    if not settings.persistent or _persist_pool is not None:
        return
    from pathlib import Path

    from lattice.db.vector import create_pg_pool
    from lattice.graph.schema import apply_schema
    from lattice.graph.store import Neo4jGraphStore

    pool = await create_pg_pool(
        settings.postgres.dsn, min_size=settings.postgres.pool_min, max_size=settings.postgres.pool_max
    )
    schema_sql = (Path(__file__).parent.parent / "db" / "schema.sql").read_text()
    async with pool.acquire() as conn:
        await conn.execute(schema_sql)
    neo4j = Neo4jGraphStore(settings.neo4j)
    await apply_schema(neo4j)
    _persist_pool = pool
    _persist_graph = neo4j


async def shutdown_persistence() -> None:
    global _persist_pool, _persist_graph
    pool = _persist_pool
    if pool is not None and hasattr(pool, "close"):
        await pool.close()
    graph = _persist_graph
    if graph is not None and hasattr(graph, "close"):
        await graph.close()
    _persist_pool = None
    _persist_graph = None


# A test override (takes precedence over the registry) plus a per-workspace
# registry so each corpus gets fully isolated in-memory stores (extra #7).
_override: Container | None = None
_registry: dict[str, Container] = {}


def get_container(x_workspace_id: str | None = Header(default=None)) -> Container:
    """FastAPI dependency: resolve the container for the request's workspace.

    A test override wins if set. Otherwise containers are created and cached per
    ``X-Workspace-Id`` header (defaulting to the configured workspace), giving each
    corpus isolated storage and graph state.
    """
    if _override is not None:
        return _override
    # Guard against non-FastAPI callers receiving the Header() sentinel.
    header = x_workspace_id if isinstance(x_workspace_id, str) else None
    ws = header or get_settings().workspace_id
    if ws not in _registry:
        _registry[ws] = build_container(workspace_id=ws)
    return _registry[ws]


def set_container(container: Container | None) -> None:
    """Override the container for every workspace (used by tests)."""
    global _override
    _override = container


def list_workspaces() -> list[str]:
    return sorted(_registry)


async def require_auth(authorization: str | None = Header(default=None)) -> None:
    """Single-user bearer-token auth. No-op when no token is configured (dev)."""
    settings = get_settings()
    if not settings.auth_token:
        return
    expected = f"Bearer {settings.auth_token}"
    if authorization != expected:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing bearer token")
