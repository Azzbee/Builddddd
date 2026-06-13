# API reference

FastAPI app (`lattice.api.app:app`). Interactive docs at `/docs`. All routes are
single-user bearer-auth when `LATTICE_AUTH_TOKEN` is set, and workspace-scoped via
an optional `X-Workspace-Id` header (defaults to the configured workspace).

## Health & ops
| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health`, `/healthz` | Liveness + version |
| GET | `/readyz` | Readiness |
| GET | `/metrics` | Prometheus exposition (requests, latency histogram) |
| GET | `/workspaces` | Corpora touched this process |

## Ingestion
| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/ingest/file` | Upload a PDF (multipart `file`); returns the job |
| POST | `/ingest/arxiv` | `{ "arxiv_id": "..." }`; fetches + ingests |
| GET | `/ingest/jobs` | All jobs (newest first) |
| GET | `/ingest/jobs/{id}` | One job (stage, status, error) |

## Papers & graph
| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/papers` | Card summaries |
| GET | `/papers/{id}` | Full PaperCard |
| GET | `/papers/{id}/neighbors` | Weighted neighbors |
| POST | `/papers/{id}/review` | Human-in-the-loop card corrections |
| GET | `/graph?min_weight&year_from&year_to` | Nodes + edges for the explorer |
| GET | `/graph/stats` | Papers / edges / communities |

## Query (agentic RAG)
| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/query` | `{ "question": "..." }` -> final answer + citations |
| POST | `/query/stream` | Same, streamed as SSE (status / tool_call / tool_result / final) |

## Landscape intelligence
| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/landscape/matrix?row&col&use_global` | Gap matrix (cell states + top gaps) |
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
