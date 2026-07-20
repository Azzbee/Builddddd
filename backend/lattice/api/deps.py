"""Dependency container and auth.

The container holds the wired services. It defaults to in-memory backends so the
API boots and is fully exercised in tests without Neo4j/Postgres/LLM keys. In
production, ``build_container`` swaps in Neo4j/pgvector/LiteLLM based on config.
Tests override the container via :func:`set_container`.
"""

from __future__ import annotations

import asyncio
import hmac
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol, cast

from fastapi import Header, HTTPException, status

from lattice.config import Settings, get_settings, validate_workspace_id
from lattice.core.llm import LiteLLMClient, LLMClient
from lattice.db.aux_stores import (
    DigestStore,
    InMemoryDigestStore,
    InMemoryWatchStore,
    WatchStore,
)
from lattice.db.blobs import BlobStore, InMemoryBlobStore
from lattice.db.cards import CorpusStore, InMemoryCardStore, InMemoryJobStore, JobStore
from lattice.db.ingest_artifacts import (
    IngestArtifactStore,
    InMemoryIngestArtifactStore,
)
from lattice.db.vector import InMemoryVectorStore, VectorStore
from lattice.embeddings.base import make_text_embedder
from lattice.embeddings.chunks import AspectEmbedder, ChunkEmbedder
from lattice.embeddings.specter2 import Specter2Embedder
from lattice.graph.store import FakeGraphStore, GraphStore
from lattice.ingestion.dispatch import (
    ArqIngestionDispatcher,
    IngestionDispatcher,
    InlineIngestionDispatcher,
    JobQueue,
)
from lattice.ingestion.grobid_client import GrobidClient
from lattice.ingestion.hybrid_parser import HybridParser
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
    artifacts: IngestArtifactStore = field(default_factory=InMemoryIngestArtifactStore)
    dispatcher: IngestionDispatcher | None = None

    def __post_init__(self) -> None:
        # Manual containers in tests and integrations must share the same durable
        # job state as their ingestion service, just like build_container does.
        self.ingestion.jobs = self.jobs
        self.ingestion.artifacts = self.artifacts
        if self.dispatcher is None:
            self.dispatcher = InlineIngestionDispatcher(self.ingestion)

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
_persist_redis: object | None = None


class _PgConnection(Protocol):
    async def fetchval(self, query: str) -> object: ...


class _PgAcquireContext(Protocol):
    async def __aenter__(self) -> _PgConnection: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool | None: ...


class _PgPool(Protocol):
    def acquire(self) -> _PgAcquireContext: ...


class _RedisPool(Protocol):
    async def ping(self) -> object: ...


async def persistence_health(settings: Settings | None = None) -> dict[str, bool]:
    """Probe every dependency required by a persistent API process."""
    settings = settings or get_settings()
    if not settings.persistent:
        return {"postgres": True, "neo4j": True, "redis": True}

    async def postgres_ready() -> bool:
        if _persist_pool is None:
            return False
        async with cast(_PgPool, _persist_pool).acquire() as connection:
            return await connection.fetchval("SELECT 1") == 1

    async def neo4j_ready() -> bool:
        if _persist_graph is None:
            return False
        rows = await _persist_graph.execute("RETURN 1 AS ok")
        return bool(rows and rows[0].get("ok") == 1)

    async def redis_ready() -> bool:
        if _persist_redis is None:
            return False
        return bool(await cast(_RedisPool, _persist_redis).ping())

    async def bounded(probe: Callable[[], Awaitable[bool]]) -> bool:
        try:
            return await asyncio.wait_for(probe(), timeout=settings.readiness_timeout_s)
        except Exception:
            return False

    postgres, neo4j, redis = await asyncio.gather(
        bounded(postgres_ready),
        bounded(neo4j_ready),
        bounded(redis_ready),
    )
    return {"postgres": postgres, "neo4j": neo4j, "redis": redis}


def build_container(settings: Settings | None = None, workspace_id: str | None = None) -> Container:
    settings = settings or get_settings()
    if workspace_id is not None and workspace_id != settings.workspace_id:
        settings = settings.model_copy(update={"workspace_id": workspace_id})
    ws = settings.workspace_id

    # Resolve the embedding backend: real local models in prod (with safe fallback),
    # the hashing fallback in demo/dev/test so the suite stays offline and fast.
    emb = settings.embedding
    prefer_local = emb.backend == "local" or (
        emb.backend == "auto" and settings.environment == "prod" and not settings.demo_mode
    )
    chunk_backend = make_text_embedder(emb.chunk_st_model, emb.chunk_dim, prefer_local=prefer_local)
    paper_backend = (
        make_text_embedder(emb.paper_st_model, emb.paper_dim, prefer_local=prefer_local)
        if prefer_local
        else None
    )
    chunk_embedder = ChunkEmbedder(backend=chunk_backend, dim=chunk_backend.dim)
    aspect_embedder = AspectEmbedder(backend=chunk_backend, dim=chunk_backend.dim)
    specter = Specter2Embedder(local_backend=paper_backend, dim=emb.paper_dim)

    vectors: VectorStore
    cards: CorpusStore
    jobs: JobStore
    artifacts: IngestArtifactStore
    graph: GraphStore
    watch: WatchStore
    digests: DigestStore
    blobs: BlobStore
    reader = None
    if settings.persistent and _persist_pool is not None and _persist_graph is not None:
        from lattice.db.blobs import PgBlobStore
        from lattice.db.ingest_artifacts import PgIngestArtifactStore
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
        artifacts = PgIngestArtifactStore(_persist_pool, ws)
        watch = PgWatchStore(_persist_pool, ws)
        digests = PgDigestStore(_persist_pool, ws)
        blobs = PgBlobStore(_persist_pool, ws)
        graph = _persist_graph
        reader = Neo4jGraphReader(_persist_graph)
    else:
        vectors = InMemoryVectorStore(hybrid_vector_weight=settings.rag.hybrid_vector_weight)
        cards = InMemoryCardStore()
        jobs = InMemoryJobStore()
        artifacts = InMemoryIngestArtifactStore()
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
        from lattice.ingestion.vision_fallback import LiteLLMVisionModel

        llm = LiteLLMClient()
        vision = (
            LiteLLMVisionModel(
                settings.docling.vision_model,
                timeout=settings.docling.vision_timeout_s,
            )
            if settings.docling.vision_enabled
            else None
        )
        parser = HybridParser(GrobidClient(settings.grobid), settings.docling, vision=vision)
        # Free SPECTER2 vectors + reference graph from S2/OpenAlex (best-effort).
        enricher = CompositeEnricher(settings.enrichment)
    ingestion = IngestionService(
        settings=settings,
        llm=llm,
        parser=parser,
        vectors=vectors,
        cards=cards,
        jobs=jobs,
        artifacts=artifacts,
        blobs=blobs,
        graph=graph,
        reader=reader,
        enricher=enricher,
        specter=specter,
        chunk_embedder=chunk_embedder,
        aspect_embedder=aspect_embedder,
        text_extractor=text_extractor,
    )
    dispatcher: IngestionDispatcher = InlineIngestionDispatcher(ingestion)
    if settings.persistent and _persist_redis is not None:
        dispatcher = ArqIngestionDispatcher(ingestion, cast(JobQueue, _persist_redis))
    return Container(
        settings=settings,
        llm=llm,
        vectors=vectors,
        cards=cards,
        jobs=jobs,
        artifacts=artifacts,
        graph=graph,
        chunk_embedder=chunk_embedder,
        ingestion=ingestion,
        watch=watch,
        digests=digests,
        blobs=blobs,
        dispatcher=dispatcher,
    )


async def init_persistence(settings: Settings | None = None) -> None:
    """Create the shared Postgres pool + Neo4j store and apply schemas. Idempotent."""
    global _persist_pool, _persist_graph, _persist_redis
    settings = settings or get_settings()
    if not settings.persistent or _persist_pool is not None:
        return
    from pathlib import Path

    from arq import create_pool
    from arq.connections import RedisSettings

    from lattice.db.vector import create_pg_pool
    from lattice.graph.schema import apply_schema
    from lattice.graph.store import Neo4jGraphStore

    pool = await create_pg_pool(
        settings.postgres.dsn,
        min_size=settings.postgres.pool_min,
        max_size=settings.postgres.pool_max,
    )
    schema_sql = (Path(__file__).parent.parent / "db" / "schema.sql").read_text()
    async with pool.acquire() as conn:
        await conn.execute(schema_sql)
    neo4j = Neo4jGraphStore(settings.neo4j)
    await apply_schema(neo4j)
    _persist_pool = pool
    _persist_graph = neo4j
    _persist_redis = await create_pool(RedisSettings.from_dsn(settings.redis.url))


async def shutdown_persistence() -> None:
    global _persist_pool, _persist_graph, _persist_redis
    pool = _persist_pool
    if pool is not None and hasattr(pool, "close"):
        await pool.close()
    graph = _persist_graph
    if graph is not None and hasattr(graph, "close"):
        await graph.close()
    redis = _persist_redis
    if redis is not None and hasattr(redis, "aclose"):
        await redis.aclose()
    _persist_pool = None
    _persist_graph = None
    _persist_redis = None


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
    settings = get_settings()
    try:
        ws = validate_workspace_id(header or settings.workspace_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    if ws not in _registry:
        if len(_registry) >= settings.max_workspaces:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                f"workspace limit reached ({settings.max_workspaces})",
            )
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
    if authorization is None or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing bearer token")
