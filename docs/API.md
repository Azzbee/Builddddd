# API reference

FastAPI app (`lattice.api.app:app`). Interactive docs are at `/docs` in development
and are disabled in production. Production startup requires `LATTICE_AUTH_TOKEN`;
send it as `Authorization: Bearer <token>`.
The Next.js app sends browser requests through its same-origin `/api` route and
adds the token on the server, so the secret is never exposed to browser code.

Application data is scoped by the optional `X-Workspace-Id` header. IDs must be
1 to 64 characters using letters, numbers, dot, `_`, or hyphen. The API
caps the number of cached workspace containers with `LATTICE_MAX_WORKSPACES`.

## Health & ops
| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health`, `/healthz` | Liveness + version |
| GET | `/readyz` | Readiness; probes PostgreSQL, Neo4j, and Redis in persistent mode, returns 503 on failure |
| GET | `/metrics` | Authenticated Prometheus exposition (requests, latency histogram) |
| GET | `/workspaces` | Authenticated list of corpora touched by this process |

## Ingestion
| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/ingest/file` | Upload a PDF (multipart `file`); returns 202 with a queued job in persistent mode |
| POST | `/ingest/arxiv` | `{ "arxiv_id": "..." }`; fetches and queues ingestion |
| GET | `/ingest/jobs` | All jobs (newest first) |
| GET | `/ingest/jobs/{id}` | One job (stage, status, error) |
| POST | `/ingest/jobs/{id}/retry` | Requeue a retryable failed or paused job from its last completed stage |

Demo and in-memory development run ingestion inline and return 200 with the
completed job. Persistent mode stores the PDF and stage context in PostgreSQL
before returning 202, then sends only workspace and job IDs through Redis. Poll
`GET /ingest/jobs/{id}` until the status is terminal. Retry is capped by
`LATTICE_INGEST_MAX_ATTEMPTS`. Job responses include `retryable`; retry returns
409 for terminal failures, active or completed jobs, and jobs at the attempt cap.
If Redis rejects submission, the API returns 503 and persists the job as a
retryable `queue_unavailable` failure instead of leaving it stuck as queued.

## Papers & graph
| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/papers` | Card summaries (incl. `superseded_by` when a newer version exists) |
| GET | `/papers/{id}` | Full PaperCard (incl. `superseded_by`; superseded versions stay retrievable) |
| GET | `/papers/{id}/neighbors` | Weighted neighbors |
| POST | `/papers/{id}/review` | Human-in-the-loop card corrections |
| POST | `/papers/summarize` | `{ "paper_ids": [...] }` -> grounded brief for a selection (graph lasso) |
| GET | `/papers/{id}/pdf` | Stream the stored source PDF (inline `application/pdf`) |
| GET | `/papers/{id}/pdf/meta` | `{ available, pages, size }` for the reader UI |
| GET | `/graph?min_weight&year_from&year_to&as_of_year` | Nodes + edges for the explorer (`as_of_year` time-travels) |
| GET | `/graph/stats` | Papers / edges / communities |
| GET | `/graph/timeline` | Publication-year bounds + cumulative growth (slider bounds) |
| GET | `/graph/delta?since_year&until_year` | Papers + edges that entered the field in (since, until] |

## Query (agentic RAG)
| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/query` | `{ "question": "..." }` -> final answer + citations |
| POST | `/query/stream` | Same, streamed as SSE (status / tool_call / tool_result / final) |

## Landscape intelligence
| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/landscape/matrix?row&col&use_global` | Gap matrix (cell states + top gaps) |
| GET | `/landscape/opportunities?row_facet&col_facet&limit&use_global` | Top gaps, each as a grounded research proposal |
| GET | `/landscape/proposal?row&col&row_facet&col_facet&use_global` | Research proposal for one specific cell |
| GET | `/landscape/momentum` | Concept momentum scorecards |
| GET | `/landscape/quadrants` | Known knowns / known unknowns / unknown knowns |

## Insight features
| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/contradictions/analyze?use_llm` | Detect claim relations (NLI), persist edges |
| GET | `/contradictions?relation` | Cached relations (e.g. `CONTRADICTS`) |
| GET | `/lineage?method` | Temporal DAG of a method family |
| GET | `/reading-queue?read=id1,id2` | Papers ranked by expected information gain |
| GET | `/related-work` | Grounded related-work draft + BibTeX |
| GET | `/export/bibtex` | `.bib` of the corpus |
| GET | `/export/obsidian` | `.zip` of wiki-linked Markdown notes |

## Watch & digest
| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/watch/queue` | Pending arXiv approval queue |
| POST | `/watch/approve` | `{ "arxiv_id", "approve" }` |
| POST | `/digest/generate` | Build + store the delta digest |
| GET | `/digest/latest`, `/digest/history` | Stored digests |

## MCP

`python -m lattice.mcp_server` exposes a read-only tool subset (`search_chunks`,
`find_papers`, `graph_neighbors`, `get_paper_card`) over stdio for Claude
Desktop/Code. Requires the `mcp` extra.
