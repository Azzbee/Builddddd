"""Structured PaperCard extraction with validate-and-repair.

Flow:
1. Render the versioned prompt with the parsed paper and identity.
2. Ask the LLM for JSON; validate against :class:`LLMPaperCardContent`.
3. On a parse/validation failure, feed the error back and retry (bounded).
4. Score confidence from the model's self-report, extraction completeness, and the
   parser's region confidence.
5. If confidence is low, escalate to a stronger model once. If still low, flag
   ``needs_review`` instead of pretending certainty.

Hallucination guards: every result must carry an evidence_location (enforced by
the prompt and re-checked here), and low parse confidence is surfaced to the model
so it hedges.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from lattice.config import ExtractionSettings
from lattice.core.cost import CostTracker
from lattice.core.errors import SchemaValidationError
from lattice.core.llm import LLMClient, LLMMessage
from lattice.core.logging import get_logger
from lattice.extraction.prompts import load_prompt, prompt_hash
from lattice.extraction.schemas import Author, LLMPaperCardContent, PaperCard
from lattice.extraction.skeleton import schema_skeleton

log = get_logger("extraction")

_SYSTEM = "You extract structured data from scientific papers and return only JSON."
_LOW_CONF_NOTICE = (
    "NOTE: parts of this paper were parsed with LOW confidence. Where the text "
    "looks garbled or incomplete, lower your self_confidence and prefer omission "
    "over guessing."
)


def render_prompt(
    version: str,
    *,
    title: str,
    authors: list[str],
    year: int | None,
    body: str,
    low_confidence: bool,
    max_chars: int,
) -> str:
    template = load_prompt(version)
    # str.format inserts a field's value verbatim (it does not re-scan the inserted
    # text for braces), so the skeleton's JSON braces pass through as-is. The body is
    # substituted the same way; only the template's own {..} fields are interpreted.
    return template.format(
        title=title or "(unknown)",
        authors=", ".join(authors) if authors else "(unknown)",
        year=year if year is not None else "(unknown)",
        body=body[:max_chars],
        low_confidence_notice=_LOW_CONF_NOTICE if low_confidence else "",
        schema=schema_skeleton(LLMPaperCardContent),
    )


def completeness(content: LLMPaperCardContent) -> float:
    """Fraction of high-value fields that are populated, in [0, 1]."""
    no_data_ok = content.paper_type in ("theoretical", "survey", "position")
    checks = [
        bool(content.problem_statement.strip()),
        bool(content.methodology.approach_summary.strip()),
        bool(content.methodology.method_family or content.methodology.techniques),
        bool(content.datasets) or no_data_ok,
        bool(content.key_results) or no_data_ok,
        bool(content.contributions),
    ]
    return sum(checks) / len(checks)


def score_confidence(content: LLMPaperCardContent, parse_confidence: float) -> float:
    """Blend self-report, completeness, and parser confidence into [0, 1]."""
    return round(
        0.5 * content.self_confidence + 0.3 * completeness(content) + 0.2 * parse_confidence,
        4,
    )


def _drop_unevidenced_results(content: LLMPaperCardContent) -> int:
    """Remove results lacking an evidence location. Returns how many were dropped."""
    kept = [r for r in content.key_results if r.evidence_location.strip()]
    dropped = len(content.key_results) - len(kept)
    content.key_results = kept
    return dropped


async def _call_with_repair(
    llm: LLMClient,
    model: str,
    user_prompt: str,
    max_repair: int,
    cost: CostTracker | None,
) -> LLMPaperCardContent:
    messages = [LLMMessage("system", _SYSTEM), LLMMessage("user", user_prompt)]
    last_error = ""
    for attempt in range(max_repair + 1):
        if cost is not None:
            cost.check()
        resp = await llm.complete(model, messages, json_mode=True, max_tokens=4096)
        if cost is not None:
            cost.record(model, resp.input_tokens, resp.output_tokens)
        try:
            data = resp.json()
            return LLMPaperCardContent.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)
            log.warning("extraction.repair", attempt=attempt, error=last_error[:200])
            messages.append(LLMMessage("assistant", resp.text))
            messages.append(
                LLMMessage(
                    "user",
                    "That output was invalid: "
                    f"{last_error[:800]}\nReturn corrected JSON only, matching the schema.",
                )
            )
    raise SchemaValidationError(f"extraction failed after {max_repair} repairs: {last_error[:300]}")


async def extract_paper_card(
    *,
    identity: dict[str, object],
    body: str,
    parse_confidence: float,
    llm: LLMClient,
    settings: ExtractionSettings,
    cost: CostTracker | None = None,
) -> PaperCard:
    """Extract a PaperCard for one paper. ``identity`` must include paper_id/title."""
    title = str(identity.get("title", ""))
    # Accept authors as Author instances or plain strings; normalize to Author.
    authors_value = identity.get("authors")
    authors_raw = authors_value if isinstance(authors_value, list) else []
    author_objs = [a if isinstance(a, Author) else Author(name=str(a)) for a in authors_raw]
    identity = {**identity, "authors": author_objs}
    authors = [a.name for a in author_objs]
    year_value = identity.get("year")
    year = year_value if isinstance(year_value, int) else None
    low_conf = parse_confidence < 0.75

    prompt = render_prompt(
        settings.prompt_version,
        title=title,
        authors=authors,
        year=year,
        body=body,
        low_confidence=low_conf,
        max_chars=settings.max_input_chars,
    )

    model = settings.primary_model
    content = await _call_with_repair(llm, model, prompt, settings.max_repair_attempts, cost)
    dropped = _drop_unevidenced_results(content)
    conf = score_confidence(content, parse_confidence)

    # Escalate once to a stronger model if confidence is low.
    escalated = False
    if conf < settings.escalation_confidence:
        log.info("extraction.escalate", paper_id=identity.get("paper_id"), conf=conf)
        model = settings.escalation_model
        escalated = True
        content = await _call_with_repair(llm, model, prompt, settings.max_repair_attempts, cost)
        dropped += _drop_unevidenced_results(content)
        conf = score_confidence(content, parse_confidence)

    reasons: list[str] = []
    if conf < settings.review_confidence:
        reasons.append(f"low_confidence={conf}")
    if low_conf:
        reasons.append(f"low_parse_confidence={parse_confidence}")
    if dropped:
        reasons.append(f"dropped_{dropped}_unevidenced_results")

    card = content.to_card(
        identity=identity,
        meta={
            "extraction_model": model,
            # Fold the schema skeleton into the hash: a schema change alters the
            # effective prompt, so it must alter the recorded version too.
            "extraction_version": (
                f"{settings.prompt_version}"
                f"@{prompt_hash(settings.prompt_version, schema_skeleton(LLMPaperCardContent))}"
            ),
            "confidence": conf,
            "needs_review": bool(reasons),
            "review_reasons": reasons,
        },
    )
    log.info(
        "extraction.done",
        paper_id=identity.get("paper_id"),
        confidence=conf,
        escalated=escalated,
        needs_review=card.needs_review,
    )
    return card
