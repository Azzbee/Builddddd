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

A PDF (uploaded directly or fetched from an arXiv id) moves through a resumable state machine
(`ingestion/pipeline.py`, wired by `ingestion/service.py`). `job.stage` is always
the last *completed* stage, so a crash resumes exactly where it stopped. Linking
is the final stage. Its cross-store writes are idempotent rather than one database
transaction, so rerunning a partially completed link converges without duplicate
graph records.

In persistent mode, the API first writes the job, source PDF, and serialized
stage context to `ingest_jobs` and `ingest_artifacts`. Only the workspace and job
IDs enter Redis. An arq worker rebuilds the workspace container, hydrates linking
state from PostgreSQL, loads the artifact, and resumes after the last completed
stage. A retry uses the same artifact and never uploads a large payload to Redis.
The persisted `retryable` flag comes from the typed pipeline error, so malformed
or scanned PDFs cannot consume worker attempts while timeouts and paused jobs can
resume. Retryable worker failures use arq's deferred retry with exponential delay
until `LATTICE_INGEST_MAX_ATTEMPTS` is reached.

1. **PARSING** - `classify_pdf` rejects corrupted/scanned/paywalled inputs into
   typed, actionable error states. GROBID extracts structure, metadata, and
   references (`grobid_client.parse_tei`, a pure function). Docling processes
   the PDF once, contributes Markdown sections and tables, and reconciles each
  matching section by token overlap. The selected text receives a
  `parse_confidence`; disputed sections can escalate to a vision LLM using a
  rendered image of the source page. DOI metadata extracted from the PDF is used
  for identity, enrichment, and deduplication, but DOI URLs are not acquisition
  endpoints. Dedup
   (content hash / DOI / arXiv / fuzzy title+author) makes re-ingestion a no-op.
2. **EXTRACTING** - the LLM fills a `PaperCard` via structured output with a
   validate-and-repair loop. Confidence blends the model's self-report,
   extraction completeness, and parse confidence; low confidence escalates to a
   stronger model once, then flags `needs_review`. Results lacking an
   `evidence_location` are dropped (anti-hallucination).

   The extraction boundary is designed for weak or cheap models (small
   local models via `ollama/...`, budget hosted tiers), each measure driven by a
   failure observed live from a local 7B model:
   - The prompt embeds a JSON skeleton *generated from the pydantic model*
     (`extraction/skeleton.py`), so weak models see the exact shape instead of a
     prose description, and the prompt can never drift from the validator. The
     skeleton is folded into `extraction_version`'s hash, so a schema change is
     visible as a version change for re-extraction backfills.
   - `LLMResponse.json()` salvages fenced and prose-wrapped JSON before giving up.
   - LLM-facing models inherit `LLMTolerantModel`: null-for-empty means absent,
     bare scalars wrap into lists, unknown keys are ignored, degenerate list
     entries (all-null datasets/results) are dropped, and `paper_type` is
     case-insensitive. Required-field omissions still fail loudly into the repair
     loop, and the internal `PaperCard` stays `extra="forbid"`. Net effect: fewer
     repair round-trips and escalations, which is also what keeps per-paper cost
     down on hosted models.
3. **ENRICHING** - Semantic Scholar / OpenAlex / Crossref add ids, citation
   counts, references, concepts, and (free) precomputed SPECTER2 vectors.
   Best-effort: failures degrade to text-only similarity.
4. **EMBEDDING** - SPECTER2 paper vector (precomputed-first), per-aspect vectors
   (problem / methodology / results), and section-anchored chunk vectors into the
   vector store.
5. **LINKING** - candidate generation (ANN), composite similarity, entity
   resolution, idempotent graph writes, edge audit, citation edges, version
   supersession, and incremental claim relations. The paper is now live in the
   graph, with any contradictions it introduces already surfaced.

Three living-graph behaviors happen inside linking:

- **Incremental linking** keeps a per-paper feature pool (SPECTER + aspect
  vectors, method and dataset sets) and the entity-resolver registries in memory
  for O(k)-per-paper linking; the calibrator refit is bounded (a deterministic
  subsample of vectors), so per-ingest cost stays flat as the corpus grows.
  Because in-memory state is lost on restart, the persistent path stores the
  SPECTER vector (`papers.specter`) and the aspect vectors (`papers.aspects`) and
  rehydrates the pool + resolver registries once, before the first ingest of a
  process (`IngestionService.hydrate`). Rehydrated papers link at full similarity
  fidelity: SPECTER (sem), methodology-section vectors (meth), method tags,
  datasets, citation reference sets (`papers.reference_ids`, for bibliographic
  coupling), and each paper's own external ids (DOI / arXiv / S2 / OpenAlex, for
  direct-citation detection) all persist and restore. Nothing degrades on restart.
- **Version supersession.** A title-level dedup match with a preprint<->published
  identifier asymmetry (arXiv-only vs DOI-bearing) is a new *version*, not a
  duplicate: the published version ingests, a `SUPERSEDED_BY` edge records the
  succession, every `RELATED_TO` edge touching the preprint is invalidated (both
  directions, `invalid_at` set - never deleted), and the preprint leaves the
  candidate pool, rehydration, analytics, and default views via the
  `_active_cards()` seam so the same work is never counted twice. It remains
  retrievable by id. An outdated preprint arriving after its published version is
  rejected with a precise reason.
- **Incremental contradiction detection.** The new paper's claims are judged
  against existing same-concept claims (offline heuristic NLI, O(new x
  same-concept)), so SUPPORTS/CONTRADICTS/EXTENDS edges accumulate as papers
  arrive. `POST /contradictions/analyze` remains for full-corpus passes with the
  LLM judge. Toggle: `LATTICE_INCREMENTAL_CONTRADICTIONS`.

## Component decisions

- **GROBID + Docling region router.** GROBID is the backbone for structure and
  references; Docling specializes in its weaknesses (tables, layout); a vision LLM
  arbitrates low-confidence regions. See PRD section 2.1.
- **Embedding backend.** `make_text_embedder` resolves to real
  `sentence-transformers` (bge-m3 for chunks, allenai-specter for papers) when
  `embedding.backend` is `local`, or `auto` in production; demo/dev/test use the
  deterministic `HashingEmbedder` so CI stays offline. A failed model load degrades
  to hashing with a warning rather than crashing. Set `LATTICE_EMBEDDING__BACKEND`.
- **In-memory communities.** The offline explorer path detects communities by
  weighted greedy modularity (`graph/community.py`), not connected components, so a
  connected graph still resolves into real sub-communities. The persistent path uses
  Neo4j GDS Louvain. Cross-community claim clustering is polarity-aware (a claim and
  its negation never merge), and the heuristic classifier requires genuine subject overlap
  (0.5) before a polarity clash counts as a contradiction - both prevent the
  false-positive contradictions that otherwise suppress known-knowns.
- **Best-effort signals are time-boxed.** The OpenAlex global-gap signal runs under a
  hard `GLOBAL_SIGNAL_BUDGET_S` wall-clock budget, so a slow/unreachable API degrades
  to "no global signal" instead of hanging the request.
- **Three-layer embeddings.** SPECTER2 for citation-informed paper similarity
  (precomputed via S2 where possible); per-aspect embeddings of LLM-distilled
  PaperCard fields to sidestep SPECTER2's domain bias and feed `S_meth`; chunk
  embeddings for hybrid retrieval. A deterministic `HashingEmbedder` is the
  offline fallback so the system always runs.
- **Custom typed graph in Neo4j.** The persisted ontology contains Paper, Author,
  Method, Dataset, Concept, and Claim nodes. We borrow
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
| `graph` | schema/constraints, similarity, evolution, entity resolution, writer, analytics, store, claim relations, lineage |
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
- **Postgres 16 + pgvector**: papers, chunks and embeddings, ingest job state,
  staged ingest artifacts, source PDFs, edge audit, watch queues, and digests.
  Supabase-compatible (`db/schema.sql`).
- **Redis**: arq task queue and enrichment cache.

### In-memory vs persistent backends

Every store sits behind a protocol with two implementations: an in-memory one
(default; powers demo mode, dev, and the offline test suite) and a
Postgres/Neo4j-backed one. Set `LATTICE_PERSISTENT=true` (docker-compose does this
for the `api` and `worker`) to wire `PgVectorStore`, `PgCardStore`, `PgJobStore`,
`PgIngestArtifactStore`, `Neo4jGraphStore`, and `Neo4jGraphReader`. The API uses
`ArqIngestionDispatcher` only when persistent Redis is available; development and
demo containers use the inline dispatcher. Because the API and worker are
separate processes, persistent reads query the live graph via `graph/reader.py`
instead of an in-process mirror. The shared pool, graph driver, and Redis client
are created once at startup in `deps.init_persistence`. Live integration tests
exercise the datastore implementations against PostgreSQL and Neo4j in CI.

## Web authentication boundary

Browser code calls the same-origin Next.js `/api/[...path]` route. That route
forwards method, query, body, content type, and workspace headers to FastAPI. It
removes any browser-supplied authorization header and injects
`LATTICE_AUTH_TOKEN` from the Next.js server environment. Direct FastAPI clients
still send the bearer token themselves. This keeps the production secret out of
JavaScript bundles, browser storage, URLs, and public environment variables.

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

## Gap -> research proposal

`landscape/proposal.py` turns a gap-matrix cell (row facet x column facet) into a
grounded, deterministic proposal. `build_proposal` is pure over injected
`FacetPaper`s: it gathers the row's track record and the column's track record
(the building blocks already in the corpus), names the method to borrow and its
strongest application, frames novelty via the Empty-vs-Blind-spot global signal,
times it with per-facet momentum and the corpus's own demand signal, lists
baselines to beat (the methods already used on the target) and honest risks
(aggregated flanking limitations + novelty caveats), and scores opportunity
confidence from building-block strength x demand x momentum. Every referenced
paper is a real corpus paper, so the proposal is grounded by construction; an LLM
can polish prose later. `IngestionService.research_opportunities` builds the gap
matrix, takes the top gaps, and drafts one proposal per cell (reusing the matrix's
global counts, so no extra network calls), ranked by confidence.

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
- **Resumability**: persisted stage context and source PDF; queue messages contain
  identifiers only; linking is final and idempotent across stores.
- **Type safety**: mypy strict (backend), Pydantic at every boundary.
- **Prompt versioning**: prompts are hashed files; cards record their version.
- **Cost ceilings**: per-job and daily caps; jobs pause (not fail) at the cap.
- **No vendor lock**: LiteLLM for models, portable pgvector schema, a `GraphStore`
  abstraction so Neo4j could be swapped for an embedded engine.
