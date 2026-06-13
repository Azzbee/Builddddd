# Lattice

**A living knowledge graph for scientific literature.**

Drop PDFs in, get back a queryable, visual, self-updating map of a research
field: paper-level cards, weighted relationships, and an agent that can answer
*"which methods have been tried for X, what data did they use, and where do they
disagree?"*

Lattice ingests scientific papers (PDF, arXiv id, or DOI), extracts their
intellectual skeleton (problem, methodology, datasets, results, limitations,
contributions) into structured **PaperCards**, and assembles them into a
weighted, **bi-temporal** knowledge graph that updates incrementally as new
papers arrive. It is explored visually and queried conversationally through an
agentic RAG layer with inline, location-grounded citations.

This is not a demo. Every component is typed, tested, observed, containerized,
resumable, and idempotent.

---

## Why it is built this way

| Concern | Decision | Rationale |
| --- | --- | --- |
| PDF structure | **GROBID** | Production standard for scholarly metadata + references |
| Tables / formulas / figures | **Docling** region router + vision fallback | GROBID is weak on tables/formulas; route per region type |
| Paper similarity | **SPECTER2** (S2 precomputed first) | Citation-informed, best quality-per-compute |
| Aspect similarity | **PaperCard field embeddings** (bge-m3) | Sidesteps SPECTER2 domain bias; feeds the weight function |
| Graph | **Custom typed schema in Neo4j** | Owning the ontology is the product |
| Retrieval | **LightRAG-style dual-level** | Low-level facts + high-level synthesis, incremental |
| Evolution | **Graphiti-style bi-temporal edges** | Supersede, never delete; O(k) per new paper |
| Enrichment | **Semantic Scholar + OpenAlex + Crossref** | Citation structure without perfect reference parsing |

Full reasoning lives in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## The core IP: the edge-weight function

```
w(i,j) = sigma( alpha*S_sem + beta*S_meth + gamma*S_cit + delta*S_data ) * T(i,j)
```

Four normalized, *auditable* components (thematic, methodological, citation,
dataset) combined into one weight, squashed by a sigmoid and modulated by a
temporal recency factor. Every weight change is written to an audit log; nothing
is silently overwritten. The full math, calibration, and tuning story is in
[`docs/SIMILARITY.md`](docs/SIMILARITY.md).

## Architecture

```
Next.js Web App  ──REST+SSE──>  FastAPI Backend
                                   │
        ┌──────────────┬──────────┴───────────┐
   Ingestion        Graph Core            Agentic RAG
   Workers (arq)     (Neo4j)               Service
        │                                      │
  GROBID · Docling · LLM extract        Vector store (pgvector)
        │
  Enrichment: Semantic Scholar · OpenAlex · Crossref · arXiv watcher
```

## Repository layout

```
lattice/
├── docker-compose.yml      # neo4j, postgres+pgvector, redis, grobid, api, worker, web
├── Makefile                # make up, make ingest FILE=..., make test, make eval
├── backend/                # Python 3.11+, FastAPI, fully typed, pytest
│   └── lattice/
│       ├── config.py       # every knob, pydantic-settings
│       ├── core/           # logging, errors, hashing, cost accounting
│       ├── ingestion/      # state machine, GROBID/Docling clients, dedup, chunker
│       ├── extraction/     # PaperCard schema + validate-and-repair extractor
│       ├── enrichment/     # S2 / OpenAlex / Crossref / arXiv clients
│       ├── embeddings/     # SPECTER2 + chunk embeddings
│       ├── graph/          # schema, idempotent writer, similarity, evolution, analytics
│       ├── rag/            # router, tools, agent loop, synthesis
│       ├── landscape/      # gap matrix, epistemic quadrants, momentum
│       ├── digest/         # weekly delta report
│       └── eval/           # extraction / retrieval / edge-quality harnesses + golden set
├── web/                    # Next.js 15 (App Router), TypeScript, Tailwind, shadcn/ui
└── docs/                   # ARCHITECTURE, SIMILARITY, RUNBOOK
```

## Quick start

```bash
cp .env.example .env        # fill in API keys
make up                     # bring up the full stack
make seed                   # pull 20 arXiv commodity-forecasting papers
make ingest FILE=paper.pdf  # ingest a local PDF
```

Develop and test the backend without any services:

```bash
cd backend
uv sync --extra dev
uv run pytest               # offline test suite
uv run ruff check . && uv run mypy lattice
```

## Beyond the core: shipped extras

All eight PRD "extras" are built, not just stubbed:

1. **Contradiction & convergence detection** - claim-level NLI surfaces where the
   corpus disagrees, as first-class `CONTRADICTS`/`SUPPORTS`/`EXTENDS` edges.
2. **Landscape Intelligence** - gap matrix (Empty vs Blind-spot via OpenAlex global
   counts), epistemic quadrants, and momentum scorecards.
3. **Lineage view** - the temporal DAG of a method family.
4. **Related-work generator** - grounded, hedged draft grouped by community, with
   BibTeX export.
5. **Obsidian export** - one wiki-linked Markdown note per paper, mirroring the graph.
6. **Reading-queue ranking** - unread papers by expected information gain.
7. **Multi-corpus workspaces** - isolated corpora via an `X-Workspace-Id` header.
8. **MCP server** - the graph as read-only tools for Claude Desktop/Code.

Plus Prometheus metrics (`/metrics`), a concrete vision fallback, and the weekly
digest loop. See [`docs/API.md`](docs/API.md) for the full endpoint reference.

## Status

Built milestone by milestone (M0 skeleton through M9 landscape intelligence) plus
the extras above. Each milestone is independently shippable with its own tests and
eval criteria; see [`docs/ROADMAP.md`](docs/ROADMAP.md) for the state. The backend
test suite (200+ tests) runs fully offline; `mypy --strict` and `ruff` are clean;
the web app builds clean.

## License

MIT.
