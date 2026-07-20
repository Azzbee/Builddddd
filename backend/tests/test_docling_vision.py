from __future__ import annotations

from lattice.ingestion.docling_client import reconcile, token_overlap
from lattice.ingestion.vision_fallback import arbitrate_region


def test_token_overlap() -> None:
    assert token_overlap("the cat sat", "the cat sat") == 1.0
    assert token_overlap("", "") == 1.0
    assert token_overlap("cat", "") == 0.0
    assert 0.0 < token_overlap("the cat sat", "the cat ran") < 1.0


def test_reconcile_accepts_high_agreement_prefers_docling() -> None:
    d = reconcile("LSTM model results", "LSTM model results table", threshold=0.5)
    assert d.accept and not d.needs_vision
    assert d.text == "LSTM model results table"  # richer Docling text


def test_reconcile_escalates_on_disagreement() -> None:
    d = reconcile("alpha beta gamma", "completely different words here now", threshold=0.9)
    assert not d.accept and d.needs_vision
    assert d.confidence < 0.9


class _FakeVision:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    async def describe_image(self, image_png: bytes, prompt: str) -> str:
        self.calls += 1
        return self.reply


async def test_arbitrate_region_uses_model() -> None:
    model = _FakeVision("clean text")
    out = await arbitrate_region(model, b"png", "a", "b")
    assert out == "clean text" and model.calls == 1
