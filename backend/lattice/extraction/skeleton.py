"""Generate an LLM-facing JSON skeleton from a Pydantic model.

Weak or cheap models (small local models, budget hosted tiers) frequently return
JSON of the wrong *shape* - wrapping everything under a `title` key, omitting
required fields, or renaming them - when the prompt only *describes* the schema in
prose. Showing them the exact structure fixes this. Rather than hand-write that
skeleton (which silently drifts from the model as fields change), we derive it from
the Pydantic model itself, so the prompt and the validator can never disagree.

The output is a fenced ```json block``` of the shape the model must produce, with
each leaf annotated by type and requiredness, e.g.::

    {
      "problem_statement": "<string>  (required)",
      "research_questions": ["<string>"],
      "methodology": {
        "approach_summary": "<string>  (required)",
        ...
      },
      "paper_type": "<one of: empirical|theoretical|...>",
      "self_confidence": 0.0
    }
"""

from __future__ import annotations

import enum
import types
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

_INDENT = "  "


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Return (inner_type, is_optional) for ``T | None`` / ``Optional[T]``.

    Handles both ``typing.Union`` and the PEP 604 ``X | None`` (``types.UnionType``).
    """
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        is_optional = len(args) != len(get_args(annotation))
        if len(args) == 1:
            return args[0], is_optional
        # Multiple non-None members: fall back to the first for the placeholder.
        return args[0], is_optional
    return annotation, False


def _placeholder(annotation: Any) -> Any:
    """A JSON-serializable placeholder value describing ``annotation``.

    Nested models recurse into a dict skeleton; lists render a one-element example;
    enums list their allowed values; scalars become a typed ``<...>`` token.
    """
    inner, _optional = _unwrap_optional(annotation)
    origin = get_origin(inner)

    if isinstance(inner, type) and issubclass(inner, BaseModel):
        return _model_skeleton(inner)

    if origin in (list, set, tuple):
        args = get_args(inner)
        elem = args[0] if args else str
        return [_placeholder(elem)]

    if origin is dict:
        return {"<key>": "<value>"}

    if isinstance(inner, type) and issubclass(inner, enum.Enum):
        allowed = "|".join(str(e.value) for e in inner)
        return f"<one of: {allowed}>"

    if inner is bool:
        return "<true|false|null>"
    if inner is int:
        return "<integer>"
    if inner is float:
        return 0.0
    if inner is str:
        return "<string>"
    return "<value>"


def _annotate_required(value: Any, field: FieldInfo) -> Any:
    """Append a `(required)` note to a leaf string placeholder for required fields."""
    if field.is_required() and isinstance(value, str) and value.startswith("<"):
        return f"{value}  (required)"
    return value


def _model_skeleton(model: type[BaseModel]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        placeholder = _placeholder(field.annotation)
        out[name] = _annotate_required(placeholder, field)
    return out


def schema_skeleton(model: type[BaseModel]) -> str:
    """Render ``model`` as a fenced JSON skeleton for embedding in an LLM prompt.

    Deterministic (no timestamps/random), so prompt hashing stays stable across runs.
    """
    import json

    skeleton = _model_skeleton(model)
    body = json.dumps(skeleton, indent=2, ensure_ascii=False)
    return f"```json\n{body}\n```"
