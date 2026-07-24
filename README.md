# Lattice

![CI](https://github.com/azzbee/builddddd/actions/workflows/ci.yml/badge.svg)

**A living knowledge graph for scientific literature.**

Drop PDFs in, get back a queryable, visual, self-updating map of a research
field: paper-level cards, weighted relationships, and an agent that can answer
*"which methods have been tried for X, what data did they use, and where do they
disagree?"*

Lattice ingests scientific papers from a PDF or arXiv id, extracts their
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
| Tables and complex layout | **Docling** reconciliation + vision fallback | GROBID is weak on layout; compare section text and arbitrate disagreements |
| Paper similarity | **SPECTER2** (S2 precomputed first) | Citation-informed, best quality-per-compute |
| Aspect similarity | **PaperCard field embeddings** (bge-m3) | Sidesteps SPECTER2 domain bias; feeds the weight function |
| Graph | **Custom typed schema in Neo4j** | Owning the ontology is the product |
| Retrieval | **LightRAG-style dual-level** | Low-level facts + high-level synthesis, incremental |
| Evolution | **Graphiti-style bi-temporal edges** | Supersede + invalidate (`invalid_at`), never delete; O(k) per new paper |
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
Browser  ──same-origin /api──>  Next.js Server  ──REST+SSE──>  FastAPI Backend
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
├── web/                    # Next.js 15 (App Router), TypeScript, Tailwind
└── docs/                   # ARCHITECTURE, SIMILARITY, RUNBOOK
```

## Quick start

See it working in 30 seconds, no API keys or databases required:

```bash
make demo      # API in offline demo mode with a populated graph
make web-dev   # open http://localhost:3000
```

Or run the full stack:

```bash
cp .env.example .env        # fill in API keys
make up                     # bring up the full stack (persistent: Postgres + Neo4j)
make seed                   # pull 20 arXiv commodity-forecasting papers
make ingest FILE=paper.pdf  # ingest a local PDF
```

The persistent API stages each PDF and returns a queued job before arq starts
work. The Next.js server adds the production bearer token to browser requests;
the token never enters client-side code.

There is also a CLI (installed as `lattice`):

```bash
lattice demo                       # run the API in offline demo mode
lattice ingest paper.pdf           # ingest a local PDF in-process
lattice query "which methods beat ARIMA, and where do they disagree?"
lattice eval --ci                  # run the offline eval harness
```

Develop and test the backend without any services:

```bash
cd backend
uv sync --extra dev
uv run pytest               # offline test suite
uv run ruff format --check .
uv run ruff check . && uv run mypy lattice
```

## Beyond the core: shipped extras

All eight PRD "extras" are built, not just stubbed:

1. **Contradiction & convergence detection** - claim-level relation classification, run incrementally
   at ingest so disagreements surface as papers arrive, plus a full-corpus LLM
   pass on demand. Surfaces where the
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

Three more capabilities round out the experience:

9. **LLM-judge RAG eval** - a RAGAS-style harness (`lattice eval --judge`) that grades
   answers on faithfulness (atomic-claim entailment), answer relevance, context
   precision, and correctness via the same provider-agnostic LLM client. The
   dependency-free offline harness still gates CI; the judge adds depth when a key
   is present. Abstentions are scored without a model call.
10. **In-app PDF reader + citation deep-linking** - source PDFs are stored at ingest
    time; the paper page embeds the PDF and chat citations carry the page number, so
    `[3] · p.8` jumps straight to the evidence.
11. **Lasso-select-to-summarize** - drag a box over the graph to select a cluster and
    get an instant, grounded brief: shared methods/datasets, year span, recurring
    open problems, and any contradictions *within* the selection.
12. **Time-travel explorer** - a publication-year slider replays how the field formed:
    the graph is reconstructed as of any year (communities + centrality recomputed on
    that subgraph), with a "what's new since" delta of papers and links that arrived
    after. The living graph, rewound.
13. **Gap -> research-proposal generator** - turns the corpus's highest-pressure gaps
    into grounded proposals: the thesis, the building blocks already in your library
    (each cited), the method to borrow and where from, why now (momentum + demand),
    baselines to beat, and honest risks - with the Empty-vs-Blind-spot novelty call.
    The "what should I work on next" payoff.

## Status

Built milestone by milestone (M0 skeleton through M9 landscape intelligence) plus
the extras above. Each milestone is independently shippable with its own tests and
eval criteria; see [`docs/ROADMAP.md`](docs/ROADMAP.md) for the state.

- Backend: 416 tests pass offline; four live integration modules skip without
  service DSNs; `mypy --strict`, Ruff lint, and Ruff format checks are clean.
- Preprint -> published **supersession** is wired end-to-end (SUPERSEDED_BY edge,
  bi-temporal edge invalidation, out of analytics and the candidate pool), and
  extraction has a **cross-provider fallback** for provider outages.
- Embeddings: real `sentence-transformers` (bge-m3 + SPECTER) wire in automatically in
  prod (`LATTICE_ENVIRONMENT=prod`) or via `LATTICE_EMBEDDING__BACKEND=local`; the
  hashing fallback keeps demo/dev/CI offline and fast. Load failures degrade, never crash.
- The production datastore code is verified **live in CI**: a dedicated job spins up
  real Postgres+pgvector and Neo4j service containers and runs the integration
  suite (chunk ANN/hybrid search, PDF blob storage, idempotent graph writes,
  bi-temporal edges, claim relations, and the Neo4j read paths) on every push.
- Web app has 17 application routes and passes Prettier, ESLint, Vitest,
  TypeScript, the production build, and `npm audit` with zero reported
  vulnerabilities. Static SQL is parsed and validated with sqlglot.
- Offline demo mode (`make demo`) loads a populated graph with zero external
  services.

## License

MIT.
