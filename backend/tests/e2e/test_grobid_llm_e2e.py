"""Live end-to-end ingestion: a real PDF through a real GROBID, then extraction.

Everything above GROBID has always been fixture-tested (``test_grobid_parse``
parses committed TEI; ``test_service_integration`` runs the pipeline against a
fake parser). Nothing verified that a real GROBID server, handed a real PDF,
returns TEI this codebase can actually parse. That is what this module does.

Two legs, gated separately, so the valuable half needs no secrets:

* **GROBID leg** - runs whenever ``LATTICE_E2E_GROBID_URL`` points at a GROBID.
  Real PDF bytes -> live ``processFulltextDocument`` -> ``parse_tei`` ->
  ``HybridParser`` -> chunking, embedding, card storage, graph writes. The model
  is scripted, so the run is deterministic and free.
* **LLM leg** - additionally requires ``LATTICE_E2E_LLM_MODEL`` and that
  provider's key in the environment. Same pipeline with the real
  ``LiteLLMClient``, asserting the extracted card actually derives from *this*
  paper rather than merely being well-formed.

Locally::

    docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.1
    LATTICE_E2E_GROBID_URL=http://localhost:8070 uv run pytest -m e2e -v

The fixture PDF is generated in-process (``tests/fixtures/paper_pdf.py``), so
there is no binary blob to review and no network fetch to flake on.
"""

from __future__ import annotations

import json
import os

import pytest

pytestmark = pytest.mark.e2e

GROBID_URL = os.environ.get("LATTICE_E2E_GROBID_URL")
if not GROBID_URL:
    pytest.skip("LATTICE_E2E_GROBID_URL not set", allow_module_level=True)

pytest.importorskip("lxml")

from lattice.config import DoclingSettings, GrobidSettings, Settings  # noqa: E402
from lattice.core.llm import LLMMessage, LLMResponse  # noqa: E402
from lattice.db.cards import InMemoryCardStore  # noqa: E402
from lattice.db.vector import InMemoryVectorStore  # noqa: E402
from lattice.embeddings.chunks import ChunkEmbedder  # noqa: E402
from lattice.embeddings.specter2 import Specter2Embedder  # noqa: E402
from lattice.graph.store import FakeGraphStore  # noqa: E402
from lattice.ingestion.grobid_client import GrobidClient  # noqa: E402
from lattice.ingestion.hybrid_parser import HybridParser  # noqa: E402
from lattice.ingestion.models import JobStatus  # noqa: E402
from lattice.ingestion.service import IngestionService  # noqa: E402

from tests.fixtures.paper_pdf import TITLE, build_sample_pdf  # noqa: E402

#: Set only when a provider key is available; the CI job sets it from a secret.
LLM_MODEL = os.environ.get("LATTICE_E2E_LLM_MODEL")

_SCRIPTED_CARD = json.dumps(
    {
        "problem_statement": "Copper prices are nonstationary, which limits linear forecasters.",
        "research_questions": ["Can recurrent models beat ARIMA on LME copper?"],
        "methodology": {
            "approach_summary": "Two-layer LSTM with attention",
            "method_family": ["deep learning"],
            "techniques": ["LSTM", "attention"],
            "baselines": ["ARIMA"],
        },
        "datasets": [{"name": "LME Copper", "is_public": True}],
        "key_results": [
            {
                "claim": "LSTM with attention beats ARIMA on copper price forecasting",
                "metric": "RMSE",
                "value": "0.12",
                "evidence_location": "Results",
            }
        ],
        "limitations": ["single metal", "ignores transaction costs"],
        "contributions": ["an attention-augmented LSTM forecaster"],
        "future_work": ["incorporate macroeconomic covariates", "test on other base metals"],
        "paper_type": "empirical",
        "domains": ["commodity markets"],
        "methods_taxonomy": ["LSTM", "attention"],
        "self_confidence": 0.9,
    }
)


class ScriptedLLM:
    """Deterministic stand-in so the GROBID leg needs no provider key."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, model: str, messages: list[LLMMessage], **kwargs: object):
        self.calls += 1
        return LLMResponse(text=_SCRIPTED_CARD, input_tokens=900, output_tokens=350, model=model)


def _parser() -> HybridParser:
    # Docling and the vision fallback are off: this job verifies the GROBID hop,
    # and both are already covered by test_docling_vision / test_hybrid_parser.
    grobid = GrobidClient(GrobidSettings(url=GROBID_URL or "", timeout_s=180.0))
    return HybridParser(grobid, DoclingSettings(enabled=False, vision_enabled=False))


def _service(llm: object) -> IngestionService:
    return IngestionService(
        settings=Settings(),
        llm=llm,  # type: ignore[arg-type]
        parser=_parser(),
        vectors=InMemoryVectorStore(),
        cards=InMemoryCardStore(),
        graph=FakeGraphStore(),
        specter=Specter2Embedder(dim=128),
        chunk_embedder=ChunkEmbedder(dim=256),
    )


# --------------------------------------------------------------------- GROBID leg
async def test_live_grobid_parses_the_paper_structure() -> None:
    """The raw GROBID hop: real PDF in, a ParsedDocument this codebase can use."""
    document = await _parser().process_fulltext(build_sample_pdf(), "copper_lstm.pdf")

    assert TITLE.lower() in document.title.lower()
    assert any("doe" in author.lower() for author in document.authors)
    assert document.abstract and "nonstationary" in document.abstract.lower()
    # Body sections came back with real prose, not empty shells.
    assert len(document.sections) >= 2
    assert any("lstm" in section.text.lower() for section in document.sections)
    # GROBID's reference model found the bibliography.
    assert document.references, "GROBID returned no references"
    # Page anchors arrived, which is what puts "p.8" on a chat citation. This is
    # the assertion that caught teiCoordinates being sent as one comma-joined
    # field: GROBID accepts that silently and stamps no coordinates at all.
    assert any(section.page == 1 for section in document.sections), (
        "no section carried a page anchor; teiCoordinates did not take effect"
    )


async def test_live_grobid_drives_the_full_ingest_pipeline() -> None:
    """Parse -> extract -> chunk -> embed -> store -> link, on live GROBID output."""
    llm = ScriptedLLM()
    svc = _service(llm)

    job = await svc.ingest_pdf("copper_lstm.pdf", build_sample_pdf())

    assert job.status == JobStatus.SUCCEEDED, job.error_message
    assert llm.calls >= 1, "extraction never reached the model"

    cards = await svc.cards.all_cards()
    assert len(cards) == 1
    card = cards[0]
    assert TITLE.lower() in card.title.lower()
    assert card.normalized_methods and card.normalized_datasets

    # Chunks came from the live TEI and are retrievable.
    hits = await svc.vectors.hybrid_search(
        "attention LSTM copper",
        svc.chunk_embedder.embed_texts(["attention LSTM copper"])[0],
        5,
        {"workspace_id": "default"},
    )
    assert hits, "no chunks were embedded from the parsed document"
    assert any(card.paper_id == hit.paper_id for hit in hits)

    # The graph was written.
    assert any("MERGE (p:Paper" in query for query, _params in svc.graph.calls)


# ------------------------------------------------------------------------ LLM leg
@pytest.mark.skipif(not LLM_MODEL, reason="LATTICE_E2E_LLM_MODEL not set (no provider key)")
async def test_live_grobid_and_live_model_produce_a_grounded_card() -> None:
    """The full production path: live GROBID, live model, no fixtures anywhere."""
    from lattice.core.llm import LiteLLMClient

    settings = Settings()
    settings.extraction.primary_model = LLM_MODEL or ""
    settings.extraction.escalation_model = LLM_MODEL or ""
    svc = _service(LiteLLMClient())
    svc.settings = settings

    job = await svc.ingest_pdf("copper_lstm.pdf", build_sample_pdf())
    assert job.status == JobStatus.SUCCEEDED, job.error_message

    (card,) = await svc.cards.all_cards()
    # The card has to be about *this* paper, not merely schema-valid.
    blob = json.dumps(card.model_dump(mode="json")).lower()
    assert "copper" in blob
    assert any(term in blob for term in ("lstm", "recurrent", "long short-term"))
    assert any(term in blob for term in ("arima", "baseline"))
    assert card.problem_statement.strip()
    assert card.key_results, "no key results extracted from a paper with a results table"
    assert card.confidence > 0
