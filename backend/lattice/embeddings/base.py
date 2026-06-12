"""Embedding backend abstraction.

A :class:`TextEmbedder` turns text into vectors. Production backends are
sentence-transformers (bge-m3) or hosted APIs (Voyage/OpenAI). The
:class:`HashingEmbedder` is a deterministic, dependency-free fallback: it lets the
whole system run and be tested offline, and degrades gracefully when no model is
configured. It is not semantically strong, but it is stable and fast, which is the
right default for CI and local development.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class TextEmbedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def l2_normalize(vec: list[float]) -> list[float]:
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return vec
    return [float(x) for x in (arr / norm)]


class HashingEmbedder:
    """Hashed bag-of-words embedder. Deterministic, offline, L2-normalized.

    Uses TF weighting with a sub-linear scale and signed feature hashing so token
    collisions partly cancel rather than always add. Good enough for plumbing,
    tests, and a no-model fallback; swap in a real model for quality.
    """

    def __init__(self, dim: int = 768) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self._dim, dtype=np.float64)
        counts: dict[str, int] = {}
        for tok in _TOKEN.findall(text.lower()):
            counts[tok] = counts.get(tok, 0) + 1
        for tok, count in counts.items():
            digest = hashlib.sha1(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign * (1.0 + math.log(count))
        return l2_normalize(vec.tolist())

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class SentenceTransformerEmbedder:  # pragma: no cover - requires torch/model download
    """sentence-transformers backend (bge-m3 by default)."""

    def __init__(self, model_name: str = "BAAI/bge-m3", dim: int = 1024) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]
