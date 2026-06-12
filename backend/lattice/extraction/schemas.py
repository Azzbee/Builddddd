"""The PaperCard: the structured extraction target and heart of Lattice.

These Pydantic models are filled by the LLM via structured output with a
validate-and-repair loop. Identity fields (title, authors, ids) come from GROBID
and enrichment, not the LLM; intellectual-content fields are LLM-extracted.

Every field that supports it carries an ``evidence_location`` so the UI can
deep-link into the PDF and the agent can cite precisely.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lattice.core.hashing import normalize_text


class PaperType(StrEnum):
    EMPIRICAL = "empirical"
    THEORETICAL = "theoretical"
    SURVEY = "survey"
    BENCHMARK = "benchmark"
    POSITION = "position"
    METHODS = "methods"
    DATASET = "dataset"
    UNKNOWN = "unknown"


class Author(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    affiliation: str | None = None
    s2_id: str | None = None
    orcid: str | None = None

    @property
    def normalized_name(self) -> str:
        return normalize_text(self.name)


class DatasetRef(BaseModel):
    """A dataset used or introduced by the paper."""

    model_config = ConfigDict(extra="forbid")

    name: str
    source: str | None = None
    size: str | None = None  # free text, e.g. "1.2M rows, 2000-2023 daily"
    is_public: bool | None = None
    url: str | None = None
    evidence_location: str | None = None

    @property
    def normalized_name(self) -> str:
        return normalize_text(self.name)


class Result(BaseModel):
    """A single key result / claim, kept granular for citation grounding and
    promotion to first-class Claim nodes in the graph."""

    model_config = ConfigDict(extra="forbid")

    claim: str
    metric: str | None = None
    value: str | None = None
    baseline_comparison: str | None = None
    #: Section/page/table anchor, e.g. "Table 3, row LSTM" or "Sec 5.2, p.8".
    evidence_location: str = ""


class ReproSignals(BaseModel):
    """Reproducibility signals extracted from the paper."""

    model_config = ConfigDict(extra="forbid")

    code_available: bool | None = None
    data_available: bool | None = None
    hyperparams_reported: bool | None = None
    code_url: str | None = None


class Methodology(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approach_summary: str
    #: High-level family, e.g. ["statistical", "deep learning", "econometric"].
    method_family: list[str] = Field(default_factory=list)
    #: Specific techniques, e.g. ["LSTM", "attention", "VAR"].
    techniques: list[str] = Field(default_factory=list)
    evaluation_protocol: str | None = None
    baselines: list[str] = Field(default_factory=list)
    reproducibility: ReproSignals = Field(default_factory=ReproSignals)

    @field_validator("method_family", "techniques", "baselines")
    @classmethod
    def _dedupe_strip(cls, v: list[str]) -> list[str]:
        seen: dict[str, str] = {}
        for item in v:
            item = item.strip()
            if item:
                seen.setdefault(normalize_text(item), item)
        return list(seen.values())


class PaperCard(BaseModel):
    """The full structured representation of a paper.

    Split into identity (deterministic, from parsing/enrichment) and intellectual
    content (LLM-extracted). ``LLMPaperCardContent`` below is the subset the LLM
    is asked to produce, keeping the extraction surface small and validatable.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Identity (GROBID + enrichment, not LLM) ---
    paper_id: str
    title: str
    authors: list[Author] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    s2_paper_id: str | None = None
    abstract: str | None = None

    # --- LLM-extracted intellectual content ---
    problem_statement: str = ""
    research_questions: list[str] = Field(default_factory=list)
    methodology: Methodology
    datasets: list[DatasetRef] = Field(default_factory=list)
    key_results: list[Result] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    contributions: list[str] = Field(default_factory=list)
    future_work: list[str] = Field(default_factory=list)

    # --- Classification ---
    paper_type: PaperType = PaperType.UNKNOWN
    domains: list[str] = Field(default_factory=list)
    methods_taxonomy: list[str] = Field(default_factory=list)

    # --- Extraction metadata ---
    extraction_model: str = ""
    extraction_version: str = ""
    confidence: float = 0.0
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)

    @field_validator("year")
    @classmethod
    def _plausible_year(cls, v: int | None) -> int | None:
        if v is not None and not (1800 <= v <= 2100):
            return None
        return v

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @property
    def normalized_methods(self) -> set[str]:
        """Normalized method tags used by entity resolution and S_meth."""
        tags = list(self.methods_taxonomy) + list(self.methodology.techniques)
        return {normalize_text(t) for t in tags if t.strip()}

    @property
    def normalized_datasets(self) -> set[str]:
        return {d.normalized_name for d in self.datasets if d.name.strip()}


class LLMPaperCardContent(BaseModel):
    """The exact JSON shape requested from the LLM. Identity fields are excluded
    because they are filled deterministically; this keeps the model focused on
    intellectual content and shrinks the validation surface."""

    model_config = ConfigDict(extra="forbid")

    problem_statement: str
    research_questions: list[str] = Field(default_factory=list)
    methodology: Methodology
    datasets: list[DatasetRef] = Field(default_factory=list)
    key_results: list[Result] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    contributions: list[str] = Field(default_factory=list)
    future_work: list[str] = Field(default_factory=list)
    paper_type: PaperType = PaperType.UNKNOWN
    domains: list[str] = Field(default_factory=list)
    methods_taxonomy: list[str] = Field(default_factory=list)
    #: The model's own confidence in this extraction, 0-1.
    self_confidence: float = 0.5

    def to_card(self, *, identity: dict[str, object], meta: dict[str, object]) -> PaperCard:
        """Combine LLM content with deterministic identity + extraction metadata."""
        return PaperCard(
            **identity,
            problem_statement=self.problem_statement,
            research_questions=self.research_questions,
            methodology=self.methodology,
            datasets=self.datasets,
            key_results=self.key_results,
            limitations=self.limitations,
            contributions=self.contributions,
            future_work=self.future_work,
            paper_type=self.paper_type,
            domains=self.domains,
            methods_taxonomy=self.methods_taxonomy,
            **meta,
        )
