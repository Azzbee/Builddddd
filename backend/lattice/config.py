"""Central configuration. Every knob lives here.

Settings are loaded from environment variables (prefix ``LATTICE_``) and an
optional ``.env`` file. Nested settings use a double-underscore delimiter, e.g.
``LATTICE_SIMILARITY__ALPHA=0.4``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SimilarityWeights(BaseSettings):
    """Coefficients for the composite edge-weight function (see docs/SIMILARITY.md).

    ``w(i,j) = sigma(alpha*S_sem + beta*S_meth + gamma*S_cit + delta*S_data) * T(i,j)``
    """

    alpha: float = 0.40  # thematic / semantic (SPECTER2)
    beta: float = 0.25  # methodological proximity
    gamma: float = 0.25  # citation structure
    delta: float = 0.10  # dataset overlap

    tau: float = 0.35  # edge materialization threshold
    knn_cap: int = 15  # max RELATED_TO edges per paper
    candidate_k: int = 50  # ANN candidates considered per new paper

    recency_boost: float = 0.15  # multiplicative boost for edges < recency_days old
    recency_days: int = 90

    sigmoid_gain: float = 4.0  # steepness of the squashing sigmoid
    sigmoid_midpoint: float = 0.5  # input value mapped to 0.5 output

    # Weight of section-level methodology cosine vs Jaccard tag overlap in S_meth.
    meth_section_weight: float = 0.6

    @field_validator("alpha", "beta", "gamma", "delta")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("similarity weights must be non-negative")
        return v


class ExtractionSettings(BaseSettings):
    primary_model: str = "claude-haiku-4-5-20251001"
    escalation_model: str = "claude-sonnet-4-6"
    fallback_model: str = "mistral/mistral-large-latest"
    prompt_version: str = "papercard_v1"
    max_repair_attempts: int = 2
    # Below this self-reported/heuristic confidence, escalate to the stronger model.
    escalation_confidence: float = 0.55
    # Below this confidence after escalation, flag needs_review.
    review_confidence: float = 0.40
    max_input_chars: int = 120_000


class EmbeddingSettings(BaseSettings):
    paper_model: str = "specter2"  # citation-informed paper similarity
    chunk_model: str = "bge-m3"  # configurable: qwen3-embedding, voyage-3, ...
    paper_dim: int = 768
    chunk_dim: int = 1024
    prefer_s2_precomputed: bool = True  # use Semantic Scholar SPECTER2 if available
    batch_size: int = 16


class GrobidSettings(BaseSettings):
    url: str = "http://grobid:8070"
    timeout_s: float = 120.0
    consolidate_citations: int = 1
    consolidate_header: int = 1


class DoclingSettings(BaseSettings):
    enabled: bool = True
    table_min_confidence: float = 0.5
    # Token-overlap agreement above which GROBID/Docling regions are auto-accepted.
    reconcile_threshold: float = 0.9


class Neo4jSettings(BaseSettings):
    uri: str = "bolt://neo4j:7687"
    user: str = "neo4j"
    password: str = "lattice-dev-password"
    database: str = "neo4j"


class PostgresSettings(BaseSettings):
    dsn: str = "postgresql://lattice:lattice@postgres:5432/lattice"
    pool_min: int = 1
    pool_max: int = 10


class RedisSettings(BaseSettings):
    url: str = "redis://redis:6379/0"


class EnrichmentSettings(BaseSettings):
    s2_api_key: str | None = None
    s2_base_url: str = "https://api.semanticscholar.org/graph/v1"
    openalex_base_url: str = "https://api.openalex.org"
    openalex_mailto: str = "adamzeraiki@gmail.com"  # polite-pool access
    crossref_base_url: str = "https://api.crossref.org"
    arxiv_base_url: str = "http://export.arxiv.org/api/query"
    cache_ttl_s: int = 60 * 60 * 24 * 7
    max_retries: int = 4
    backoff_base_s: float = 1.0


class RagSettings(BaseSettings):
    agent_model: str = "claude-sonnet-4-6"
    router_model: str = "claude-haiku-4-5-20251001"
    max_tool_calls: int = 8
    low_confidence_floor: float = 0.35  # below this the agent answers "I don't know"
    hybrid_vector_weight: float = 0.6  # vs BM25 in hybrid retrieval fusion
    top_k_chunks: int = 12


class CostSettings(BaseSettings):
    per_job_usd_cap: float = 1.00
    daily_usd_cap: float = 50.00
    target_per_paper_usd: float = 0.15


class WatcherSettings(BaseSettings):
    arxiv_categories: list[str] = Field(default_factory=lambda: ["q-fin.ST", "cs.LG", "econ.EM"])
    similarity_floor: float = 0.45  # min similarity-to-corpus to enqueue for approval
    poll_interval_s: int = 60 * 60 * 6


class Settings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_prefix="LATTICE_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["dev", "test", "prod"] = "dev"
    demo_mode: bool = False  # offline demo corpus + deterministic models at startup
    workspace_id: str = "default"  # multi-tenant ready: every row carries this
    log_level: str = "INFO"
    log_json: bool = True
    data_dir: str = "/data"
    auth_token: str | None = None  # single-user bearer token
    rate_limit_per_min: int = 240  # per-client request cap; 0 disables

    similarity: SimilarityWeights = Field(default_factory=SimilarityWeights)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    grobid: GrobidSettings = Field(default_factory=GrobidSettings)
    docling: DoclingSettings = Field(default_factory=DoclingSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    enrichment: EnrichmentSettings = Field(default_factory=EnrichmentSettings)
    rag: RagSettings = Field(default_factory=RagSettings)
    cost: CostSettings = Field(default_factory=CostSettings)
    watcher: WatcherSettings = Field(default_factory=WatcherSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton settings instance."""
    return Settings()
