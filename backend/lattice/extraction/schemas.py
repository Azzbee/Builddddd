"""The PaperCard: the structured extraction target and heart of Lattice.

These Pydantic models are filled by the LLM via structured output with a
validate-and-repair loop. Identity fields (title, authors, ids) come from GROBID
and enrichment, not the LLM; intellectual-content fields are LLM-extracted.

Every field that supports it carries an ``evidence_location`` so the UI can
deep-link into the PDF and the agent can cite precisely.
"""

from __future__ import annotations

import types
from enum import StrEnum
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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

    @classmethod
    def _missing_(cls, value: object) -> PaperType | None:
        """Case/whitespace-insensitive lookup: LLMs emit "Empirical" as readily as
        "empirical"; the case carries no meaning, so accept it."""
        if isinstance(value, str):
            low = value.strip().lower()
            for member in cls:
                if member.value == low:
                    return member
        return None


def _accepts_none(annotation: Any) -> bool:
    """True if ``annotation`` permits None (``T | None`` / ``Optional[T]``)."""
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        return type(None) in get_args(annotation)
    return annotation is type(None)


def _list_element_type(annotation: Any) -> Any:
    """The element type of a ``list[...]`` annotation, or None."""
    if get_origin(annotation) is not list:
        return None
    args = get_args(annotation)
    return args[0] if args else None


def _clean_model_list(items: list[Any], elem: type[BaseModel]) -> list[Any]:
    """Normalize a list of LLM-produced entries for a nested model.

    * a bare string entry maps onto the element's single required field when that is
      unambiguous (``"datasets": ["LME Copper"]`` -> ``{"name": "LME Copper"}``);
    * ``null`` entries and dict entries whose required identity field is null or
      missing (a live local-model failure: ``{"name": null, "source": null, ...}``)
      are noise meaning "nothing here" - drop the entry, not the whole extraction.
      This matches the repo's hallucination-guard stance: prefer omission.
    """
    required = [n for n, f in elem.model_fields.items() if f.is_required()]
    out: list[Any] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, str) and len(required) == 1:
            item = {required[0]: item}
        if isinstance(item, dict) and any(item.get(r) is None for r in required):
            continue  # degenerate entry: required identity absent/null
        out.append(item)
    return out


class LLMTolerantModel(BaseModel):
    """Base for models populated from LLM output.

    Weak or cheap models reliably produce shape quirks that carry no semantic
    ambiguity, so we normalize them instead of failing the whole extraction into the
    repair loop (every case below was observed live from a local 7B model):

    * ``null`` for an "empty" field that does not accept None -> treat as absent so
      the field default applies. A *required* field sent as null still fails with a
      clear "Field required" error, which is what the repair prompt needs.
    * a bare string where a list is expected -> wrap it in a one-element list.
    * inside lists of nested models: ``null`` entries and entries whose required
      identity field is null/missing are dropped (see :func:`_clean_model_list`);
      a bare string entry maps onto the element's single required field.
    * ``null`` entries inside plain lists (e.g. ``["a", null]``) are dropped.
    * unknown extra keys -> ignored. (Internal models like ``PaperCard`` stay
      ``extra="forbid"``; tolerance is strictly an LLM-boundary property.)
    """

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _normalize_llm_output(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out: dict[str, Any] = {}
        for key, value in data.items():
            fld = cls.model_fields.get(key)
            if fld is None:
                continue  # unknown key from the model: drop it
            if value is None and not _accepts_none(fld.annotation):
                continue  # null-for-empty: let the default (or required error) apply
            elem = _list_element_type(fld.annotation)
            if elem is not None:
                if isinstance(value, str):
                    value = [value]  # bare scalar where a list belongs
                if isinstance(value, list):
                    if isinstance(elem, type) and issubclass(elem, BaseModel):
                        value = _clean_model_list(value, elem)
                    else:
                        value = [v for v in value if v is not None]
            out[key] = value
        return out


class Author(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    affiliation: str | None = None
    s2_id: str | None = None
    orcid: str | None = None

    @property
    def normalized_name(self) -> str:
        return normalize_text(self.name)


class DatasetRef(LLMTolerantModel):
    """A dataset used or introduced by the paper."""

    name: str
    source: str | None = None
    size: str | None = None  # free text, e.g. "1.2M rows, 2000-2023 daily"
    is_public: bool | None = None
    url: str | None = None
    evidence_location: str | None = None

    @property
    def normalized_name(self) -> str:
        return normalize_text(self.name)


class Result(LLMTolerantModel):
    """A single key result / claim, kept granular for citation grounding and
    promotion to first-class Claim nodes in the graph."""

    claim: str
    metric: str | None = None
    value: str | None = None
    baseline_comparison: str | None = None
    #: Section/page/table anchor, e.g. "Table 3, row LSTM" or "Sec 5.2, p.8".
    evidence_location: str = ""


class ReproSignals(LLMTolerantModel):
    """Reproducibility signals extracted from the paper."""

    code_available: bool | None = None
    data_available: bool | None = None
    hyperparams_reported: bool | None = None
    code_url: str | None = None


class Methodology(LLMTolerantModel):
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


class LLMPaperCardContent(LLMTolerantModel):
    """The exact JSON shape requested from the LLM. Identity fields are excluded
    because they are filled deterministically; this keeps the model focused on
    intellectual content and shrinks the validation surface."""

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
