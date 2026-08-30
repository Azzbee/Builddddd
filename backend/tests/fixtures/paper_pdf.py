"""Build a small, paper-shaped PDF in memory, with no third-party dependency.

The end-to-end test needs a real PDF for a real GROBID to parse. Committing a
binary blob would be opaque and unreviewable, and downloading one from arXiv in
CI would add a network flake to a job whose whole point is to be dependable. So
the fixture is generated: raw PDF syntax, uncompressed, one Helvetica text stream
laid out like a paper (title, authors, abstract, numbered sections, references),
so what GROBID sees is right here in the source.

The content is deliberately a plausible commodity-forecasting paper matching the
demo corpus's domain, so an LLM extraction run against it produces a PaperCard
whose fields can be asserted.
"""

from __future__ import annotations

TITLE = "Recurrent Neural Networks for Copper Price Forecasting"
AUTHOR = "Jane Doe"
ABSTRACT_MARKER = "nonstationary"

#: (font, size, leading-before, text). Rendered top-down as one text object.
_LINES: list[tuple[str, int, int, str]] = [
    ("F2", 17, 0, TITLE),
    ("F1", 11, 26, "Jane Doe, Carlos Ng"),
    ("F1", 10, 16, "Institute for Commodity Analytics"),
    ("F2", 12, 30, "Abstract"),
    ("F1", 10, 18, "Daily copper prices are nonstationary and prone to regime shifts, which"),
    ("F1", 10, 13, "limits classical linear forecasters. We train a long short-term memory"),
    ("F1", 10, 13, "network with an attention layer on eleven years of LME copper settlement"),
    ("F1", 10, 13, "prices and compare it against an ARIMA baseline. The recurrent model"),
    ("F1", 10, 13, "reduces out-of-sample RMSE from 0.21 to 0.12, a 43 percent improvement."),
    ("F2", 12, 26, "1. Introduction"),
    ("F1", 10, 18, "Forecasting industrial metal prices matters for hedging and inventory"),
    ("F1", 10, 13, "policy. Prices respond to macroeconomic demand, mine supply shocks and"),
    ("F1", 10, 13, "speculative flows, so the series is nonstationary and its volatility"),
    ("F1", 10, 13, "clusters. Linear models such as ARIMA assume neither of these."),
    ("F2", 12, 26, "2. Data and Method"),
    ("F1", 10, 18, "We use daily LME copper settlement prices from 2010 to 2021, a public"),
    ("F1", 10, 13, "series of 2,847 observations. The forecaster is a two-layer LSTM with a"),
    ("F1", 10, 13, "single attention head over a 30-day context window, trained with the Adam"),
    ("F1", 10, 13, "optimizer. The baseline is an ARIMA(2,1,2) fitted on the same window."),
    ("F2", 12, 26, "3. Results"),
    ("F1", 10, 18, "Table 1 reports out-of-sample accuracy. The LSTM with attention reaches"),
    ("F1", 10, 13, "an RMSE of 0.12 against 0.21 for ARIMA, and the gap widens during the"),
    ("F1", 10, 13, "2015 and 2020 volatility regimes. Directional accuracy rises from 51 to"),
    ("F1", 10, 13, "58 percent. Ablating the attention layer removes about half the gain."),
    ("F2", 12, 26, "4. Limitations and Future Work"),
    ("F1", 10, 18, "The study covers a single metal and ignores transaction costs. Future"),
    ("F1", 10, 13, "work should incorporate macroeconomic covariates and test the model on"),
    ("F1", 10, 13, "other base metals such as aluminium and nickel."),
    ("F2", 12, 26, "References"),
    ("F1", 9, 18, "[1] Box, G. and Jenkins, G. Time Series Analysis: Forecasting and"),
    ("F1", 9, 12, "    Control. Holden-Day, 1970."),
    ("F1", 9, 12, "[2] Hochreiter, S. and Schmidhuber, J. Long Short-Term Memory. Neural"),
    ("F1", 9, 12, "    Computation, 9(8):1735-1780, 1997."),
    ("F1", 9, 12, "[3] Vaswani, A. et al. Attention Is All You Need. Advances in Neural"),
    ("F1", 9, 12, "    Information Processing Systems, 2017."),
]

_TOP = 748
_LEFT = 72


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _content_stream() -> bytes:
    ops = ["BT"]
    y = _TOP
    for font, size, gap, text in _LINES:
        y -= gap
        ops.append(f"/{font} {size} Tf")
        ops.append(f"1 0 0 1 {_LEFT} {y} Tm")
        ops.append(f"({_escape(text)}) Tj")
    ops.append("ET")
    return "\n".join(ops).encode("latin-1")


def build_sample_pdf() -> bytes:
    """Return a valid single-page PDF of the paper above."""
    stream = _content_stream()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n".encode()
    out += b"%%EOF\n"
    return bytes(out)
