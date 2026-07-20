"""Typed error hierarchy. Pipeline stages raise these so the job state machine
can map failures to actionable, resumable states rather than opaque tracebacks.
"""

from __future__ import annotations


class LatticeError(Exception):
    """Base class for all Lattice errors."""

    #: Stable machine-readable code surfaced in API responses and job state.
    code: str = "lattice_error"
    #: Whether retrying the same operation could plausibly succeed.
    retryable: bool = False


class ConfigError(LatticeError):
    code = "config_error"


# ---------------------------------------------------------------------------
# Ingestion / parsing
# ---------------------------------------------------------------------------
class IngestionError(LatticeError):
    code = "ingestion_error"


class CorruptedPdfError(IngestionError):
    code = "corrupted_pdf"


class ScannedPdfError(IngestionError):
    """PDF has no extractable text layer (likely scanned images)."""

    code = "scanned_pdf"


class PaywalledStubError(IngestionError):
    """Downloaded artifact is a paywall landing page, not the paper."""

    code = "paywalled_stub"


class DuplicateError(IngestionError):
    """The document already exists in the corpus (idempotent no-op)."""

    code = "duplicate"


class ParserError(IngestionError):
    code = "parser_error"


class ParserTimeoutError(ParserError):
    code = "parser_timeout"
    retryable = True


class ParserUnavailableError(ParserError):
    """A parsing dependency is missing (deployment/config error, not a bad file)."""

    code = "parser_unavailable"


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
class ExtractionError(LatticeError):
    code = "extraction_error"


class SchemaValidationError(ExtractionError):
    """LLM output failed schema validation after all repair attempts."""

    code = "schema_validation_failed"
    retryable = True


# ---------------------------------------------------------------------------
# Enrichment / external services
# ---------------------------------------------------------------------------
class EnrichmentError(LatticeError):
    code = "enrichment_error"
    retryable = True


class RateLimitedError(EnrichmentError):
    code = "rate_limited"
    retryable = True


class NotFoundError(EnrichmentError):
    """External service has no record for the identifier (terminal, not retryable)."""

    code = "not_found"
    retryable = False


# ---------------------------------------------------------------------------
# Cost control
# ---------------------------------------------------------------------------
class CostCapExceeded(LatticeError):
    """A configured spend cap was hit. Jobs PAUSE rather than fail."""

    code = "cost_cap_exceeded"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
class GraphError(LatticeError):
    code = "graph_error"


class CypherValidationError(GraphError):
    """A run_cypher request violated the read-only allowlist."""

    code = "cypher_not_allowed"
