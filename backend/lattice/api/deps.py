"""Dependency container and auth.

The container holds the wired services. It defaults to in-memory backends so the
API boots and is fully exercised in tests without Neo4j/Postgres/LLM keys. In
production, ``build_container`` swaps in Neo4j/pgvector/LiteLLM based on config.
Tests override the container via :func:`set_container`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import Header, HTTPException, status

from lattice.config import Settings, get_settings
from lattice.core.llm import LiteLLMClient, LLMClient
from lattice.db.cards import InMemoryCardStore, InMemoryJobStore
from lattice.db.vector import InMemoryVectorStore
from lattice.embeddings.chunks import ChunkEmbedder
from lattice.graph.store import FakeGraphStore, GraphStore
from lattice.ingestion.grobid_client import GrobidClient
from lattice.ingestion.service import IngestionService
from lattice.rag.agent import RagAgent
from lattice.rag.tools import Toolbox


@dataclass
class Container:
    settings: Settings
    llm: LLMClient
    vectors: InMemoryVectorStore
    cards: InMemoryCardStore
    jobs: InMemoryJobStore
    graph: GraphStore
    chunk_embedder: ChunkEmbedder
    ingestion: IngestionService
    _watch_queue: list[dict[str, object]] = field(default_factory=list)
    _digests: list[dict[str, object]] = field(default_factory=list)

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


def build_container(settings: Settings | None = None) -> Container:
    settings = settings or get_settings()
    llm: LLMClient = LiteLLMClient()
    vectors = InMemoryVectorStore(hybrid_vector_weight=settings.rag.hybrid_vector_weight)
    cards = InMemoryCardStore()
    jobs = InMemoryJobStore()
    graph: GraphStore = FakeGraphStore()
    chunk_embedder = ChunkEmbedder(dim=settings.embedding.chunk_dim)
    parser = GrobidClient(settings.grobid)
    ingestion = IngestionService(
        settings=settings,
        llm=llm,
        parser=parser,
        vectors=vectors,
        cards=cards,
        graph=graph,
        chunk_embedder=chunk_embedder,
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
    )


_container: Container | None = None


def get_container() -> Container:
    global _container
    if _container is None:
        _container = build_container()
    return _container


def set_container(container: Container | None) -> None:
    """Override the container (used by tests)."""
    global _container
    _container = container


async def require_auth(authorization: str | None = Header(default=None)) -> None:
    """Single-user bearer-token auth. No-op when no token is configured (dev)."""
    settings = get_settings()
    if not settings.auth_token:
        return
    expected = f"Bearer {settings.auth_token}"
    if authorization != expected:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing bearer token")
