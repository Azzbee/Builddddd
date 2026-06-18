"""Lattice command-line interface.

    lattice serve [--port 8000] [--reload]   # run the API
    lattice demo  [--port 8000]              # run the API in offline demo mode
    lattice ingest paper.pdf                 # ingest a local PDF (in-process)
    lattice query "which methods beat ARIMA?"
    lattice eval [--ci]                      # run the offline eval harness
    lattice eval --judge                     # LLM-graded RAG eval (needs a key)
    lattice version

``ingest`` and ``query`` run against an in-process container using current config
(set LATTICE_DEMO_MODE=true to run fully offline with deterministic models).
``eval --judge`` runs the agent over the golden Q/A set and grades the answers
with an LLM (faithfulness / relevance / context precision / correctness); it needs
a real provider key, so it is separate from the dependency-free offline harness.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from lattice import __version__


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lattice", description="Lattice knowledge graph CLI")
    sub = p.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the API server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    demo = sub.add_parser("demo", help="run the API in offline demo mode")
    demo.add_argument("--host", default="127.0.0.1")
    demo.add_argument("--port", type=int, default=8000)

    ing = sub.add_parser("ingest", help="ingest a local PDF")
    ing.add_argument("path")

    q = sub.add_parser("query", help="ask the agent a question")
    q.add_argument("question")

    ev = sub.add_parser("eval", help="run the offline eval harness")
    ev.add_argument("--ci", action="store_true")
    ev.add_argument("--judge", action="store_true", help="LLM-graded RAG eval (needs a key)")

    sub.add_parser("version", help="print the version")
    return p


async def _ingest(filename: str, data: bytes) -> int:
    from lattice.api.deps import build_container

    container = build_container()
    job = await container.ingestion.ingest_pdf(filename, data)
    await container.jobs.save(job)
    print(json.dumps(job.model_dump(mode="json"), indent=2, default=str))
    return 0 if job.status.value in ("succeeded", "duplicate") else 1


def _contexts_from_result(result: object) -> list[str]:
    """Collect retrieved chunk texts from an AgentResult's tool calls."""
    ctx: list[str] = []
    for tc in getattr(result, "tool_calls", []):
        res = getattr(tc, "result", None)
        if isinstance(res, dict):
            for ch in res.get("chunks", []) or []:
                if isinstance(ch, dict) and ch.get("text"):
                    ctx.append(str(ch["text"]))
    return ctx


async def _judge_eval(ci: bool) -> int:
    from lattice.api.deps import build_container
    from lattice.eval.golden import load_qa_pairs
    from lattice.eval.llm_judge import LLMJudge, evaluate_rag_judged

    container = build_container()
    pairs = load_qa_pairs()
    results = []
    contexts = []
    for qa in pairs:
        res = await container.make_agent().run(qa.question)
        results.append(res)
        contexts.append(_contexts_from_result(res))
    judge = LLMJudge(container.llm, container.settings.rag.agent_model)
    report = await evaluate_rag_judged(pairs, results, judge, contexts=contexts)
    print(json.dumps(report.to_json(), indent=2))
    if ci and not report.passes():
        print("JUDGED EVAL CHECK FAILED", file=sys.stderr)
        return 1
    return 0


async def _query(question: str) -> int:
    from lattice.api.deps import build_container

    container = build_container()
    result = await container.make_agent().run(question)
    print(result.answer)
    if result.citations:
        print("\nSources:")
        for c in result.citations:
            loc = f" ({c.evidence_location})" if c.evidence_location else ""
            print(f"  [{c.marker}] {c.title}{loc}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "version":
        print(f"lattice {__version__}")
        return 0

    if args.command == "eval":
        if getattr(args, "judge", False):
            return asyncio.run(_judge_eval(args.ci))
        from lattice.eval.runner import run_offline_eval

        report, ok = run_offline_eval()
        print(json.dumps(report, indent=2))
        if args.ci and not ok:
            print("EVAL CI CHECK FAILED", file=sys.stderr)
            return 1
        return 0

    if args.command in ("serve", "demo"):
        import os

        import uvicorn

        if args.command == "demo":
            os.environ["LATTICE_DEMO_MODE"] = "true"
        from lattice.config import get_settings

        get_settings.cache_clear()
        uvicorn.run(
            "lattice.api.app:app",
            host=args.host,
            port=args.port,
            reload=getattr(args, "reload", False),
        )
        return 0

    if args.command == "ingest":
        path = Path(args.path)
        return asyncio.run(_ingest(path.name, path.read_bytes()))
    if args.command == "query":
        return asyncio.run(_query(args.question))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
