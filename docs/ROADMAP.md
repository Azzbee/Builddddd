# Milestones

Each milestone is independently shippable with its own tests and eval criteria.

| Milestone | Scope | State |
| --- | --- | --- |
| M0 | Skeleton: config, core utils, tooling, CI, healthchecks | Done |
| M1 | Ingestion pipeline: state machine, GROBID TEI, Docling reconcile, dedup, chunker, PDF chaos handling | Done |
| M2 | Extraction: PaperCard schema, versioned prompts, validate-and-repair, confidence + escalation, vision fallback | Done |
| M3 | Embeddings + enrichment: SPECTER2, chunk/aspect embedders, S2/OpenAlex/Crossref/arXiv with cache + backoff | Done |
| M4 | Graph: schema/constraints, idempotent writer, entity resolution, composite similarity, evolution, GDS analytics | Done |
| M5 | Graph explorer UI: Sigma.js, paper cards, filters, edge anatomy | Done (web) |
| M6 | Agentic RAG: router, typed tools, ReAct agent (SSE), citations, cost, RAG eval | Done |
| M7 | Evolution + watcher + digest: incremental updates, arXiv approval queue, weekly delta | Done |
| M8 | Hardening: auth, cost caps, runbook, schema, deploy compose | Done |
| M9 | Landscape Intelligence: gap matrix, epistemic quadrants, momentum | Done |

## Definitions of done (met)

- **M1**: PDFs ingest end to end with per-stage status; broken files fail
  gracefully into typed states. (`test_ingestion`, `test_grobid_parse`,
  `test_service_integration`)
- **M2**: extraction eval runs and reports field-level F1. (`test_extraction`,
  `eval/extraction_eval`, `scripts/run_eval.py`)
- **M3**: every paper has embeddings + enrichment or an explicit not-found state.
  (`test_embeddings`, `test_enrichment`)
- **M4**: ingesting the corpus yields a sane graph; edge-quality eval runs.
  (`test_graph`, `eval/edge_quality`)
- **M6**: RAG eval reports faithfulness and citation correctness against
  thresholds. (`test_rag`, `eval/retrieval_eval`)
- **M9**: matrix/quadrants/momentum computed and exposed. (`test_landscape`)

## Extras (PRD section 10) - implemented

| Extra | State | Where |
| --- | --- | --- |
| 1. Contradiction & convergence detection (NLI) | Done | `graph/contradictions.py`, `/contradictions` |
| 2. Landscape Intelligence (matrix, quadrants, momentum) | Done | `landscape/`, `/landscape/*` |
| 3. Lineage view (temporal DAG of a method family) | Done | `graph/lineage.py`, `/lineage` |
| 4. Related-work generator + BibTeX export | Done | `rag/related_work.py`, `/related-work`, `/export/bibtex` |
| 5. Obsidian/Markdown export with wiki-links | Done | `export/obsidian.py`, `/export/obsidian` |
| 6. Reading-queue ranking (expected information gain) | Done | `landscape/reading_queue.py`, `/reading-queue` |
| 7. Multi-corpus workspaces | Done | `api/deps.py` (X-Workspace-Id), `/workspaces` |
| 8. MCP server (graph as tools) | Done | `mcp_server.py` |

Plus, beyond the PRD extras: OpenAlex global signals powering the Empty-vs-Blind
-spot distinction (`landscape/signals.py`); a concrete vision-fallback model
(`ingestion/vision_fallback.py`); Prometheus metrics + request timing
(`core/metrics.py`, `/metrics`); and the weekly digest generation loop
(`/digest/generate`).

| Extra | State | Where |
| --- | --- | --- |
| 9. LLM-judge RAG eval (RAGAS-style, faithfulness/relevance/precision/correctness) | Done | `eval/llm_judge.py`, `lattice eval --judge` |
| 10. Source-PDF storage + in-app reader + citation page deep-linking | Done | `db/blobs.py`, `/papers/{id}/pdf`, `web/app/papers/[id]` |
| 11. Lasso-select-to-summarize in the explorer | Done | `IngestionService.summarize_papers`, `/papers/summarize`, `web/components/GraphExplorer.tsx` |
| 12. Time-travel graph replay (as-of-year snapshot + delta) | Done | `IngestionService.graph_snapshot/timeline/delta`, `/graph?as_of_year`, `/graph/timeline`, `/graph/delta` |
| 13. Gap -> research-proposal generator | Done | `landscape/proposal.py`, `IngestionService.research_proposal/research_opportunities`, `/landscape/proposal`, `/landscape/opportunities`, `web/app/opportunities` |

## Production verification

The production datastore code is no longer "written but unrun": a CI job spins up
real Postgres+pgvector and Neo4j service containers and runs the live integration
suite on every push (chunk ANN + hybrid search with monotonic fused scores,
idempotent graph writes, an idempotent edge-audit trail, bi-temporal edges, claim
relations, the Neo4j read paths, SPECTER-vector persistence + feature rehydration
for cross-restart linking, and the Postgres card / job / watch / digest / PDF-blob
stores). Backend coverage is ~92% with an 85% CI floor. Persistence is wired by
`LATTICE_PERSISTENT` (on by default in docker-compose), and a `lattice` CLI
provides serve/demo/ingest/query/eval.

## Future work

- GROBID/LLM end-to-end run in CI (needs a model key); both are thin/fixture-tested.
- Optional Phase-2 domain-adapted embedding fine-tune (ship only if eval wins).
- Unknown-unknowns proxies surfaced in the UI (question-coverage probing).
- **Preprint -> published paper supersession.** `RELATED_TO` edges are already
  bi-temporal (invalidated via `invalid_at`, never deleted) and cross-restart
  incremental linking is wired. What is *not* yet wired is the paper-level
  supersession heuristic (`detect_supersession` / `writer.set_superseded` +
  `invalidate_related_edge`, both unit-tested): today the dedup stage rejects a DOI
  -bearing published version of an existing arXiv preprint as a duplicate before
  linking, so the SUPERSEDED_BY edge never fires. Wiring it means teaching dedup to
  route a preprint<->published match into supersession instead of rejection.
