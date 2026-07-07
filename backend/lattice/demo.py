"""Offline demo mode: a hand-crafted corpus that loads with no external services.

When ``LATTICE_DEMO=true`` the container is wired with a deterministic LLM and a
synthetic parser, and a small, deliberately interconnected corpus is ingested at
startup. Opening the web app then shows a live graph with communities,
contradictions, convergence, open problems, and momentum, with zero API keys,
Neo4j, Postgres, or GROBID. It is the fastest path to "holy shit, it works".
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from lattice.core.llm import LLMMessage, LLMResponse
from lattice.ingestion.models import ParsedDocument, ParsedReference, ParsedSection


@dataclass
class DemoPaper:
    filename: str
    title: str
    authors: list[str]
    year: int
    domain: str
    methods: list[str]
    datasets: list[str]
    problem: str
    results: list[tuple[str, str]]  # (claim, evidence_location)
    future_work: list[str]
    references: list[str]
    doi: str | None = None
    arxiv_id: str | None = None


# A commodity-forecasting corpus with two communities (deep-learning vs
# econometrics), a contradiction, convergent claims, and shared open problems.
DEMO_CORPUS: list[DemoPaper] = [
    DemoPaper(
        "lstm_copper.pdf", "LSTM Forecasting of LME Copper Prices",
        ["Jane Doe", "Carlos Ng"], 2019, "commodity markets",
        ["LSTM", "attention"], ["LME Copper"],
        "Daily copper price forecasting is hard due to nonstationarity and regime shifts.",
        [("LSTM with attention significantly improves forecasting accuracy over ARIMA", "Table 3")],
        ["test on other base metals", "incorporate macroeconomic covariates"],
        ["10.1/shared-arima", "10.1/lstm-orig"],
    ),
    DemoPaper(
        "gru_metals.pdf", "GRU Networks for Industrial Metal Prices",
        ["Wei Zhang"], 2020, "commodity markets",
        ["GRU", "LSTM"], ["LME Copper", "LME Aluminium"],
        "Recurrent models for multi-metal price forecasting remain underexplored.",
        [("GRU improves forecasting accuracy on industrial metals versus ARIMA", "Section 5")],
        ["incorporate macroeconomic covariates", "evaluate transaction costs"],
        ["10.1/shared-arima"],
    ),
    DemoPaper(
        "transformer_fx.pdf", "Transformer Models for Commodity Forecasting",
        ["Maria Silva", "Tomas Berg"], 2022, "commodity markets",
        ["transformer", "attention"], ["LME Copper", "COMEX Gold"],
        "Whether transformers beat recurrent models for commodity series is unsettled.",
        [("Transformers significantly improve forecasting accuracy over LSTM baselines", "Table 2")],
        ["reduce data requirements", "test on supply disruption events"],
        ["10.1/lstm-orig", "10.1/transformer-orig"],
    ),
    DemoPaper(
        "transformer_skeptic.pdf", "Are Transformers Effective for Commodity Time Series?",
        ["Olivia Park"], 2023, "commodity markets",
        ["transformer", "linear models"], ["LME Copper"],
        "Recent claims that transformers dominate time-series forecasting deserve scrutiny.",
        [("Transformers show no significant improvement in forecasting accuracy over LSTM baselines for commodity prices", "Table 4")],
        ["benchmark across more commodities", "study sample-size effects"],
        ["10.1/transformer-orig"],
    ),
    DemoPaper(
        "var_panel.pdf", "Vector Autoregression for Commodity Panels",
        ["Hans Mueller"], 2016, "econometrics",
        ["VAR"], ["World Bank Pink Sheet"],
        "High-dimensional commodity panels challenge classical VAR estimation.",
        [("Plain VAR underperforms on high-dimensional commodity panels", "Section 4")],
        ["apply regularization", "model nonlinear dynamics"],
        ["10.1/var-orig"],
    ),
    DemoPaper(
        "lasso_var.pdf", "Regularized VAR for Commodity Forecasting",
        ["Hans Mueller", "Greta Roth"], 2018, "econometrics",
        ["VAR", "LASSO", "ridge"], ["World Bank Pink Sheet"],
        "Regularization may rescue VAR on wide commodity panels.",
        [("Regularized VAR reduces forecast error versus plain VAR", "Table 1")],
        ["model nonlinear dynamics", "combine with machine learning"],
        ["10.1/var-orig"],
    ),
    DemoPaper(
        "garch_vol.pdf", "GARCH Volatility Models for Energy Commodities",
        ["Ahmed Hassan"], 2017, "econometrics",
        ["GARCH"], ["EIA Energy"],
        "Energy price volatility clustering is poorly captured by linear models.",
        [("GARCH captures volatility clustering better than rolling-window baselines", "Section 6")],
        ["combine with machine learning", "extend to agricultural commodities"],
        ["10.1/garch-orig"],
    ),
    DemoPaper(
        "wavelet_preprint.pdf", "Wavelet Decomposition for Commodity Price Forecasting",
        ["Elena Petrova", "Igor Volkov"], 2021, "commodity markets",
        ["wavelets", "LSTM"], ["LME Copper"],
        "Multi-scale decomposition may separate trend from noise in commodity prices.",
        [("Wavelet-LSTM hybrids improve forecasting accuracy over plain LSTM", "Table 2")],
        ["evaluate on longer horizons"],
        ["10.1/lstm-orig"],
        arxiv_id="2101.00001",
    ),
    DemoPaper(
        "wavelet_published.pdf", "Wavelet Decomposition for Commodity Price Forecasting",
        ["Elena Petrova", "Igor Volkov"], 2022, "commodity markets",
        ["wavelets", "LSTM"], ["LME Copper"],
        "Multi-scale decomposition may separate trend from noise in commodity prices.",
        [("Wavelet-LSTM hybrids improve forecasting accuracy over plain LSTM", "Table 2")],
        ["evaluate on longer horizons"],
        ["10.1/lstm-orig"],
        doi="10.5555/wavelet.2022",
    ),
    DemoPaper(
        "diffusion_supply.pdf", "Diffusion Models for Supply-Disruption Forecasting",
        ["Sofia Marin"], 2024, "commodity markets",
        ["diffusion models"], ["Custom Disruption Dataset"],
        "Forecasting prices around supply disruptions is largely unaddressed.",
        [("Diffusion models generate calibrated scenario forecasts under supply shocks", "Table 5")],
        ["scale to more commodities", "reduce inference cost"],
        ["10.1/diffusion-orig"],
    ),
]


def _document(p: DemoPaper) -> ParsedDocument:
    return ParsedDocument(
        title=p.title,
        authors=p.authors,
        abstract=f"{p.problem} We study {', '.join(p.methods)} on {', '.join(p.datasets)}.",
        year=p.year,
        doi=p.doi,
        arxiv_id=p.arxiv_id,
        sections=[
            ParsedSection(section_id="intro", title="Introduction", text=p.problem),
            ParsedSection(
                section_id="method", title="Methodology",
                text=f"We apply {', '.join(p.methods)} to {', '.join(p.datasets)} for price forecasting.",
            ),
            ParsedSection(
                section_id="results", title="Results",
                text=" ".join(c for c, _ in p.results) + " Baseline comparisons are reported.",
            ),
        ],
        references=[ParsedReference(raw=r, doi=r) for r in p.references],
    )


def _card_content(p: DemoPaper) -> str:
    return json.dumps(
        {
            "problem_statement": p.problem,
            "research_questions": [],
            "methodology": {
                "approach_summary": f"{p.methods[0]} based forecaster",
                "method_family": ["deep learning" if p.domain == "commodity markets" else "econometric"],
                "techniques": p.methods,
                "baselines": ["ARIMA"],
                "reproducibility": {"code_available": True},
            },
            "datasets": [{"name": d, "is_public": True} for d in p.datasets],
            "key_results": [{"claim": c, "evidence_location": ev} for c, ev in p.results],
            "limitations": ["single asset class"],
            "contributions": [f"a {p.methods[0]} approach to {p.domain}"],
            "future_work": p.future_work,
            "paper_type": "empirical",
            "domains": [p.domain],
            "methods_taxonomy": p.methods,
            "self_confidence": 0.9,
        }
    )


class DemoParser:
    """Returns the synthetic ParsedDocument for each demo filename."""

    def __init__(self) -> None:
        self._docs = {p.filename: _document(p) for p in DEMO_CORPUS}

    async def process_fulltext(self, pdf_bytes: bytes, filename: str = "paper.pdf") -> ParsedDocument:
        return self._docs.get(filename, _document(DEMO_CORPUS[0]))


class DemoLLM:
    """Deterministic LLM: serves canned card content for extraction, and a graceful
    message for the chat agent (real chat needs a provider)."""

    def __init__(self) -> None:
        self._content = {p.title: _card_content(p) for p in DEMO_CORPUS}

    async def complete(self, model: str, messages: list[LLMMessage], **kwargs: object) -> LLMResponse:
        system = next((m.content for m in messages if m.role == "system"), "")
        user = "\n".join(m.content for m in messages if m.role == "user")
        if "research assistant" in system:
            return LLMResponse(
                text=json.dumps(
                    {
                        "answer": "Demo mode is offline. Set ANTHROPIC_API_KEY and disable "
                        "LATTICE_DEMO to enable the chat agent.",
                        "citations": [],
                        "confidence": 0.0,
                    }
                ),
                model=model,
            )
        for title, content in self._content.items():
            if title in user:
                return LLMResponse(text=content, input_tokens=200, output_tokens=120, model=model)
        return LLMResponse(text=_card_content(DEMO_CORPUS[0]), model=model)


def demo_pdf_bytes(filename: str) -> bytes:
    return b"%PDF-1.7\n" + filename.encode() + b" " + b"x" * 4000


async def load_demo(container: object) -> int:
    """Ingest the demo corpus into a container and run contradiction analysis."""
    ingestion = container.ingestion  # type: ignore[attr-defined]
    jobs = container.jobs  # type: ignore[attr-defined]
    n = 0
    for p in DEMO_CORPUS:
        job = await ingestion.ingest_pdf(p.filename, demo_pdf_bytes(p.filename))
        await jobs.save(job)
        n += 1
    await ingestion.analyze_relations()  # heuristic NLI -> contradictions + supports
    return n
