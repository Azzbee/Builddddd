"""Versioned, hashed prompt loader.

Prompts are files so they are diffable and reviewable. Each PaperCard records the
prompt version that produced it; re-extraction is a one-command backfill keyed on
version. The content hash lets us detect drift between a recorded version and the
file on disk.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).parent


@lru_cache(maxsize=16)
def load_prompt(version: str) -> str:
    """Return the raw prompt template for a version (e.g. ``papercard_v1``)."""
    path = _DIR / f"{version}.md"
    if not path.exists():
        raise FileNotFoundError(f"unknown prompt version: {version}")
    return path.read_text(encoding="utf-8")


def prompt_hash(version: str) -> str:
    """Short content hash of a prompt version, for drift detection."""
    return hashlib.sha256(load_prompt(version).encode("utf-8")).hexdigest()[:12]


def available_versions() -> list[str]:
    return sorted(p.stem for p in _DIR.glob("*.md"))
