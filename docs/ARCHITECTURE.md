# Architecture

Lattice turns a pile of PDFs into a queryable, visual, self-updating knowledge
graph. This document explains the components, the data flow, and the design
decisions behind them.

## System overview

```
Next.js Web App  ──REST + SSE──>  FastAPI Backend
                                     │
        ┌──────────────┬────────────┴───────────┐
   Ingestion        Graph Core              Agentic RAG
   Workers (arq)     (Neo4j + GDS)           Service
        │                                        │
  GROBID · Docling · LLM extract          Vector store (pgvector)
        │
  Enrichment: Semantic Scholar · OpenAlex · Crossref · arXiv watcher
```

## Data flow (ingestion)

A PDF (or arXiv id / DOI) moves through a resumable state machine
(`ingestion/pipeline.py`, wired by `ingestion/service.py`). `job.stage` is always
the last *completed* stage, so a crash resumes exactly where it stopped. Linking
is the final, transactional stage, so a crash never leaves a partial paper in the
graph.

1. **PARSING** - `classify_pdf` rejects corrupted/scanned/paywalled inputs into
   typed, actionable error states. GROBID extracts structure, metadata, and
   references (`grobid_client.parse_tei`, a pure function). Docling handles
   tables/figures and reconciles regions by token overlap, attaching a
   `parse_confidence`; low-confidence pages escalate to a vision LLM. Dedup
   (content hash / DOI / arXiv / fuzzy title+author) makes re-ingestion a no-op.
2. **EXTRACTING** - the LLM fills a `PaperCard` via structured output with a
   validate-and-repair loop. Confidence blends the model's self-report,
   extraction completeness, and parse confidence; low confidence escalates to a
   stronger model once, then flags `needs_review`. Results lacking an
   `evidence_location` are dropped (anti-hallucination).
3. **ENRICHING** - Semantic Scholar / OpenAlex / Crossref add ids, citation
   counts, references, concepts, and (free) precomputed SPECTER2 vectors.
   Best-effort: failures degrade to text-only similarity.
4. **EMBEDDING** - SPECTER2 paper vector (precomputed-first), per-aspect vectors
   (problem / methodology / results), and section-anchored chunk vectors into the
   vector store.
5. **LINKING** - candidate generation (ANN), composite similarity, entity
   resolution, idempotent graph writes, edge audit, citation edges. The paper is
   now live in the graph.

## Component decisions

- **GROBID + Docling region router.** GROBID is the backbone for structure and
  references; Docling specializes in its weaknesses (tables, layout); a vision LLM
  arbitrates low-confidence regions. See PRD section 2.1.
- **Three-layer embeddings.** SPECTER2 for citation-informed paper similarity
  (precomputed via S2 where possible); per-aspect embeddings of LLM-distilled
  PaperCard fields to sidestep SPECTER2's domain bias and feed `S_meth`; chunk
  embeddings for hybrid retrieval. A deterministic `HashingEmbedder` is the
  offline fallback so the system always runs.
- **Custom typed graph in Neo4j.** Owning the ontology (Paper, Author, Method,
  Dataset, Concept, Claim, OpenProblem, Formula) is the product. We borrow
  LightRAG's dual-level retrieval and Graphiti's bi-temporal edges as *patterns*,
  not dependencies.
- **Provider-agnostic LLM access** via a thin `LLMClient` (LiteLLM in prod),
  so extraction / RAG / vision / synthesis are portable and cost-tracked.

## Backend module map

| Package | Responsibility |
| --- | --- |
| `core` | logging, typed errors, hashing/idempotency, LLM client, cost caps |
| `ingestion` | state machine, GROBID/Docling/vision, dedup, chunker, service |
| `extraction` | PaperCard schema, versioned prompts, validate-and-repair extractor |
| `enrichment` | cached/back-off S2/OpenAlex/Crossref/arXiv clients |
| `embeddings` | SPECTER2, chunk + aspect embedders, hashing fallback |
| `graph` | schema/constraints, similarity (core IP), evolution, entity resolution, writer, analytics, store, contradictions (NLI), lineage |
| `rag` | router, typed tools, ReAct agent (SSE), cited synthesis, related-work generator |
| `landscape` | gap matrix, epistemic quadrants, momentum, global/demand signals, reading queue |
| `export` | Obsidian/Markdown notes (BibTeX lives in `rag.related_work`) |
| `digest` | weekly delta report |
| `eval` | extraction / retrieval / edge-quality harnesses + golden set |
| `db` | vector store (in-memory + pgvector), card/job stores, SQL schema |
| `api` | FastAPI routers + DI container (workspace-scoped) |
| `mcp_server` | read-only graph tools over MCP for Claude Desktop/Code |
| `core.metrics` | dependency-free Prometheus registry (`/metrics`) |

## Storage

- **Neo4j 5 + GDS**: the typed graph; PageRank/Louvain/betweenness write back onto
  nodes for the explorer.
- **Postgres 16 + pgvector**: papers, chunks + embeddings, ingest job state, edge
  audit, watch subscriptions/queue, digests. Supabase-compatible (`db/schema.sql`).
- **Redis**: arq task queue and enrichment cache.

### In-memory vs persistent backends

Every store sits behind a protocol with two implementations: an in-memory one
(default; powers demo mode, dev, and the offline test suite) and a
Postgres/Neo4j-backed one. Set `LATTICE_PERSISTENT=true` (docker-compose does this
for the `api` and `worker`) to wire `PgVectorStore`, `PgCardStore`, `PgJobStore`,
the `Neo4jGraphStore`, and a `Neo4jGraphReader`. Because the API and worker are
separate processes, the persistent read paths query the live graph via the reader
(`graph/reader.py`) rather than any in-process mirror, so reads are cross-process
correct. The shared pool and driver are created once at startup
(`deps.init_persistence`). All persistent code is verified live in CI against real
service containers.

## Agentic RAG

A query is classified (factual / relational / global / comparative) to steer
retrieval. A ReAct-style agent (provider-agnostic JSON protocol) calls typed tools
(`search_chunks`, `get_paper_card`, `graph_neighbors`, `find_papers`,
`compare_papers`, `run_cypher` [read-only, validated], `find_contradictions`),
capped at 8 tool calls. Every answer streams over SSE with inline citations
validated against retrieved provenance (the agent cannot cite what it did not
see). Citations carry the section *and PDF page*, so the chat UI deep-links
`[3] · p.8` straight into the embedded reader. Low confidence yields an honest
"I don't know". Token and cost are tracked per query; a tool trace powers the
"how I got this" panel.

## Evaluation

Two harnesses share one golden set. The **offline** harness (`lattice eval`,
`scripts/run_eval.py`) is dependency-free and gates CI: extraction P/R/F1 with
regression checks, retrieval citation-correctness/faithfulness proxies, and
edge-quality rank correlation against the PRD thresholds. The **LLM-judge**
harness (`lattice eval --judge`, `lattice/eval/llm_judge.py`) adds a RAGAS-style
suite graded by a model through the same `LLMClient`: faithfulness via
atomic-claim entailment, answer relevance, context precision, and answer
correctness. It runs the agent over the golden Q/A set, captures the retrieved
context from the tool trace, and scores each item; abstentions are graded without
a model call. Because the judge is an injected `LLMClient`, the whole harness is
unit-tested offline with a scripted model.

## Time-travel (graph replay)

`graph_snapshot(as_of_year=Y)` reconstructs the explorer graph as of a publication
year: only papers published up to `Y` and edges whose both endpoints qualify are
included, and (in the in-memory/demo path) communities and weighted-degree
centrality are *recomputed on that subgraph* so the replay is faithful, not just a
filter. The Neo4j reader applies the same year bound in Cypher (keeping its
precomputed Louvain/PageRank). `graph_timeline` returns the year bounds and
cumulative growth for the slider; `graph_delta(since, until)` is a set-difference
of two reconstructed snapshots, so "what entered the field in (since, until]" is
correct on both backends with no extra queries. Paper nodes carry a stable
`created_at` (`coalesce` on first write) alongside the bi-temporal edge
`valid_from`/`invalid_at`, so an ingest-time axis is available too.

## Source PDFs

The raw PDF is persisted at the (final, transactional) linking stage via a
`BlobStore` (in-memory offline; Postgres `bytea` in production), keyed by
`paper_id` with a precomputed page count. Section-anchored chunks carry their
page, which flows chunk -> vector record -> chunk hit -> provenance -> citation,
giving the reader and chat a precise deep-link target.

## Non-functional guarantees

- **Idempotency**: content-hash + DOI dedup; every graph write is a MERGE.
- **Resumability**: persisted job stage; linking is the final transaction.
- **Type safety**: mypy strict (backend), Pydantic at every boundary.
- **Prompt versioning**: prompts are hashed files; cards record their version.
- **Cost ceilings**: per-job and daily caps; jobs pause (not fail) at the cap.
- **No vendor lock**: LiteLLM for models, portable pgvector schema, a `GraphStore`
  abstraction so Neo4j could be swapped for an embedded engine.
