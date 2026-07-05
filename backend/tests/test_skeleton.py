"""The prompt schema skeleton is derived from the pydantic model, never hand-written.

These tests pin the contract: every field of LLMPaperCardContent appears, required
fields are marked, enums list their values, nesting recurses, and the emitted block
is itself valid JSON (so a model that copies the shape produces parseable output).
"""

from __future__ import annotations

import json

from lattice.extraction.schemas import LLMPaperCardContent
from lattice.extraction.skeleton import schema_skeleton
from pydantic import BaseModel


def _body(fenced: str) -> str:
    assert fenced.startswith("```json\n") and fenced.endswith("\n```")
    return fenced[len("```json\n") : -len("\n```")]


def test_skeleton_covers_every_field() -> None:
    out = schema_skeleton(LLMPaperCardContent)
    for name in LLMPaperCardContent.model_fields:
        assert f'"{name}"' in out, f"field {name} missing from skeleton"


def test_skeleton_marks_required_fields() -> None:
    parsed = json.loads(_body(schema_skeleton(LLMPaperCardContent)))
    assert "(required)" in parsed["problem_statement"]
    assert "(required)" in parsed["methodology"]["approach_summary"]
    assert "(required)" in parsed["datasets"][0]["name"]
    assert "(required)" in parsed["key_results"][0]["claim"]
    # Defaulted fields carry no required marker.
    assert "(required)" not in parsed["key_results"][0]["evidence_location"]


def test_skeleton_expands_enums_and_nests() -> None:
    parsed = json.loads(_body(schema_skeleton(LLMPaperCardContent)))
    for value in ("empirical", "theoretical", "survey", "benchmark", "unknown"):
        assert value in parsed["paper_type"]
    # Nested model recursion: reproducibility inside methodology.
    assert parsed["methodology"]["reproducibility"]["code_available"] == "<true|false|null>"
    # Lists render a single example element.
    assert isinstance(parsed["research_questions"], list) and len(parsed["research_questions"]) == 1


def test_skeleton_block_is_valid_json_and_deterministic() -> None:
    a = schema_skeleton(LLMPaperCardContent)
    b = schema_skeleton(LLMPaperCardContent)
    assert a == b  # prompt hashing depends on this
    json.loads(_body(a))  # must not raise


def test_skeleton_handles_generic_annotations() -> None:
    class Inner(BaseModel):
        x: int

    class Sample(BaseModel):
        req_str: str
        opt_str: str | None = None
        flag: bool | None = None
        num: float = 0.5
        count: int = 0
        tags: list[str] = []
        mapping: dict[str, str] = {}
        nested: Inner | None = None
        items: list[Inner] = []

    parsed = json.loads(_body(schema_skeleton(Sample)))
    assert parsed["req_str"].endswith("(required)")
    assert parsed["opt_str"] == "<string>"
    assert parsed["flag"] == "<true|false|null>"
    assert parsed["num"] == 0.0
    assert parsed["count"] == "<integer>"
    assert parsed["tags"] == ["<string>"]
    assert parsed["mapping"] == {"<key>": "<value>"}
    assert parsed["nested"] == {"x": "<integer>  (required)"}
    assert parsed["items"] == [{"x": "<integer>  (required)"}]
