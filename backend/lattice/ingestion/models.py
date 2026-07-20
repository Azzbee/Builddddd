"""Data models for ingestion: parsed documents, regions, and job state.

The job moves through an explicit state machine so any stage can crash and resume
from the last persisted state. Linking (graph write) is the final step and touches
the graph last, so a crash before it leaves no paper in the graph; its writes are
idempotent, so a re-run of a partially-linked paper converges (it is not a single
cross-store transaction).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from lattice.core.hashing import normalize_arxiv, normalize_doi


class SourceType(StrEnum):
    FILE = "file"
    ARXIV = "arxiv"
    DOI = "doi"


class RegionType(StrEnum):
    PROSE = "prose"
    TABLE = "table"
    FORMULA = "formula"
    FIGURE = "figure"
    ALGORITHM = "algorithm"


class JobStage(StrEnum):
    """Happy-path pipeline stages. ``stage`` is always the last *completed* stage,
    so the resume point is unambiguous. Off-ramps live in ``JobStatus``."""

    RECEIVED = "received"
    PARSING = "parsing"
    EXTRACTING = "extracting"
    ENRICHING = "enriching"
    EMBEDDING = "embedding"
    LINKING = "linking"
    DONE = "done"


class JobStatus(StrEnum):
    """Orthogonal run status. A job at stage=EXTRACTING with status=PAUSED resumes
    by running the ENRICHING handler next."""

    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"  # e.g. cost cap hit; resumable from ``stage``
    FAILED = "failed"  # resumable from ``stage`` on retry
    DUPLICATE = "duplicate"
    SUCCEEDED = "succeeded"


#: Linear happy-path order used to compute progress and resume points.
STAGE_ORDER: list[JobStage] = [
    JobStage.RECEIVED,
    JobStage.PARSING,
    JobStage.EXTRACTING,
    JobStage.ENRICHING,
    JobStage.EMBEDDING,
    JobStage.LINKING,
    JobStage.DONE,
]


def next_stage(stage: JobStage) -> JobStage | None:
    """Return the stage following ``stage`` on the happy path, or None at DONE."""
    idx = STAGE_ORDER.index(stage)
    if idx >= len(STAGE_ORDER) - 1:
        return None
    return STAGE_ORDER[idx + 1]


class ParsedReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw: str = ""
    title: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    year: int | None = None
    authors: list[str] = Field(default_factory=list)

    def model_post_init(self, _ctx: object) -> None:
        object.__setattr__(self, "doi", normalize_doi(self.doi))
        object.__setattr__(self, "arxiv_id", normalize_arxiv(self.arxiv_id))


class TableArtifact(BaseModel):
    """A structured table extracted by Docling TableFormer (cells, not just MD)."""

    model_config = ConfigDict(extra="forbid")

    table_id: str
    caption: str | None = None
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    page: int | None = None
    parse_confidence: float = 1.0


class FormulaArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    formula_id: str
    latex: str
    region_anchor: str | None = None
    parse_confidence: float = 1.0


class FigureArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    figure_id: str
    caption: str | None = None
    alt_text: str | None = None  # vision-LLM grounded description
    page: int | None = None


class ParsedSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str
    title: str
    text: str
    level: int = 1
    region_type: RegionType = RegionType.PROSE
    page: int | None = None
    parse_confidence: float = 1.0


class ParsedDocument(BaseModel):
    """The reconciled output of GROBID + Docling (+ vision fallback)."""

    model_config = ConfigDict(extra="forbid")

    title: str = ""
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    language: str = "en"
    sections: list[ParsedSection] = Field(default_factory=list)
    references: list[ParsedReference] = Field(default_factory=list)
    tables: list[TableArtifact] = Field(default_factory=list)
    formulas: list[FormulaArtifact] = Field(default_factory=list)
    figures: list[FigureArtifact] = Field(default_factory=list)
    #: Min parse confidence across sections; gates extraction hedging.
    overall_confidence: float = 1.0

    def section_by_keywords(self, *keywords: str) -> ParsedSection | None:
        """Find the first section whose title contains any keyword (case-insensitive)."""
        kws = [k.lower() for k in keywords]
        for s in self.sections:
            tl = s.title.lower()
            if any(k in tl for k in kws):
                return s
        return None

    def full_text(self) -> str:
        parts: list[str] = []
        if self.abstract:
            parts.append(f"# Abstract\n{self.abstract}")
        for s in self.sections:
            parts.append(f"# {s.title}\n{s.text}")
        return "\n\n".join(parts)


class IngestJob(BaseModel):
    """Persisted job state. The single source of truth for resumability."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    workspace_id: str = "default"
    source_type: SourceType
    source_ref: str  # filename, arxiv id, or doi
    content_hash: str | None = None
    paper_id: str | None = None
    stage: JobStage = JobStage.RECEIVED
    status: JobStatus = JobStatus.QUEUED
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    attempts: int = 0
    cost_usd: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def progress(self) -> float:
        """Fraction of the happy path completed, in [0, 1]."""
        if self.status == JobStatus.DUPLICATE:
            return 1.0
        return STAGE_ORDER.index(self.stage) / (len(STAGE_ORDER) - 1)

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.DUPLICATE,
        )
