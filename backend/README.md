# Lattice (backend)

The Python backend for Lattice, a living knowledge graph for scientific literature.

See the top-level [`README.md`](../README.md), [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md),
and [`docs/SIMILARITY.md`](../docs/SIMILARITY.md) for the full picture.

## Quick start

```bash
uv sync --extra dev          # install core + dev dependencies
uv run pytest                # run the offline test suite
uv run ruff format --check . # check formatting
uv run ruff check .          # lint
uv run mypy lattice          # type-check
```

The pure-logic core (similarity, extraction schema, chunker, dedup, entity
resolution, RAG router, eval harness) runs with no external services. Heavier
integrations (FastAPI, Neo4j, Postgres, GROBID, Docling, LLM providers,
sentence-transformers) are optional extras: `api`, `db`, `llm`, `parsing`,
`embeddings`, `observability`.
