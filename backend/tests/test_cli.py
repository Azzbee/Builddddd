from __future__ import annotations

import pytest
from lattice.cli import main
from lattice.demo import demo_pdf_bytes

pytest.importorskip("lxml")


@pytest.fixture
def _demo_env(monkeypatch: pytest.MonkeyPatch):
    from lattice.config import get_settings

    monkeypatch.setenv("LATTICE_DEMO_MODE", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    assert "lattice" in capsys.readouterr().out


def test_eval_runs(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["eval", "--ci"]) == 0
    out = capsys.readouterr().out
    assert "extraction" in out and "retrieval" in out


def test_ingest_and_query_demo(_demo_env, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    pdf = tmp_path / "lstm_copper.pdf"
    pdf.write_bytes(demo_pdf_bytes("lstm_copper.pdf"))
    assert main(["ingest", str(pdf)]) == 0
    assert "succeeded" in capsys.readouterr().out

    assert main(["query", "what is unknown?"]) == 0
    assert capsys.readouterr().out.strip()  # prints an answer


def test_eval_runner_thresholds() -> None:
    from lattice.eval.runner import run_offline_eval

    report, ok = run_offline_eval()
    assert ok is True
    assert report["extraction"]["macro_f1"] >= 0.99
