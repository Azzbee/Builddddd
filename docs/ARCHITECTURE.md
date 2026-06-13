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

## Agentic RAG

A query is classified (factual / relational / global / comparative) to steer
retrieval. A ReAct-style agent (provider-agnostic JSON protocol) calls typed tools
(`search_chunks`, `get_paper_card`, `graph_neighbors`, `find_papers`,
`compare_papers`, `run_cypher` [read-only, validated], `find_contradictions`),
capped at 8 tool calls. Every answer streams over SSE with inline citations
validated against retrieved provenance (the agent cannot cite what it did not
see). Low confidence yields an honest "I don't know". Token and cost are tracked
per query; a tool trace powers the "how I got this" panel.

## Non-functional guarantees

- **Idempotency**: content-hash + DOI dedup; every graph write is a MERGE.
- **Resumability**: persisted job stage; linking is the final transaction.
- **Type safety**: mypy strict (backend), Pydantic at every boundary.
- **Prompt versioning**: prompts are hashed files; cards record their version.
- **Cost ceilings**: per-job and daily caps; jobs pause (not fail) at the cap.
- **No vendor lock**: LiteLLM for models, portable pgvector schema, a `GraphStore`
  abstraction so Neo4j could be swapped for an embedded engine.
