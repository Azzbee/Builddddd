# Contributing to Lattice

Thanks for helping. This repo holds itself to a high bar: typed, tested, observed.

## Development setup

```bash
cd backend
uv sync --extra dev          # core + dev deps
uv run pytest                # offline test suite
uv run ruff check . && uv run mypy lattice
```

Frontend:

```bash
cd web
npm install
npm run build && npm run lint
```

## Standards (enforced in CI)

- **Types**: `mypy --strict` must pass on `backend/lattice`. Pydantic at every boundary.
- **Lint/format**: `ruff check` and `ruff format` must be clean.
- **Tests**: every change ships with tests. Pure logic is unit-tested; integrations
  use injected fakes offline and live service-container tests in CI.
- **No em dashes** in code, UI strings, or docs (house style).
- **Conventional-ish commits**: a short imperative subject plus a body explaining
  the why.

## Adding a feature

1. Put pure logic in a dedicated module with a focused unit test.
2. Inject external dependencies (LLM, stores, HTTP) behind protocols so the logic
   stays testable without services.
3. Expose it via an API router and, where it has a UI, a Next.js page.
4. Update `docs/` (at least `API.md` and `ROADMAP.md`).
5. Run `make check` (lint + typecheck + tests) before pushing.

## Running the live integration tests

```bash
# with Postgres+pgvector and Neo4j reachable:
export LATTICE_TEST_PG_DSN=postgresql://lattice:lattice@localhost:5432/lattice
export LATTICE_TEST_NEO4J_URI=bolt://localhost:7687
make test-integration
```

CI runs these automatically against service containers.
