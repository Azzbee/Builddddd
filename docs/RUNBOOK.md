# Runbook

Operational guide for running, deploying, and maintaining Lattice.

## Local development

```bash
cp .env.example .env            # fill in ANTHROPIC_API_KEY etc.
cd backend
uv sync --extra dev             # core + dev deps
uv run pytest                   # offline test suite (no services needed)
uv run ruff format --check .
uv run ruff check . && uv run mypy lattice
```

The pure-logic core runs with no external services. To run the API locally
against in-memory backends (no Neo4j/Postgres required for a demo):

```bash
make api-dev                    # http://localhost:8000/docs
```

## Offline demo

The fastest way to see Lattice working, with no external services or API keys:

```bash
make demo        # API in demo mode at :8000 (synthetic populated corpus)
make web-dev     # web app at :3000, pointed at the demo API
```

Note: to verify a production web build while the dev server is running, use
`npm run build:check` (builds into `.next-build`). A plain `next build` shares
`.next` with `next dev` and corrupts the running server's chunks ("Cannot find
module './NNN.js'"); if that ever happens, stop the dev server, `rm -rf web/.next`,
and restart it.

Demo mode (`LATTICE_DEMO_MODE=true`) wires deterministic models and loads a small,
deliberately interconnected corpus at startup, so the graph, contradictions,
quadrants, momentum, lineage, and reading queue are all populated immediately.

## Local model (real papers, zero API cost)

Run the full real pipeline (GROBID parse -> extraction -> embeddings -> linking ->
graph) against a local Ollama model instead of a hosted API. Verified end to end
with `qwen2.5:7b`: real arXiv PDFs ingest, link, and chat for $0.00.

```bash
# prerequisites: Ollama running with a model pulled, e.g. `ollama pull qwen2.5:7b`
docker run -d --name grobid -p 8070:8070 lfoppiano/grobid:0.8.1
cd backend && uv pip install -e ".[llm,parsing]"

LATTICE_EXTRACTION__PRIMARY_MODEL=ollama/qwen2.5:7b \
LATTICE_EXTRACTION__ESCALATION_MODEL=ollama/qwen2.5:7b \
LATTICE_EXTRACTION__FALLBACK_MODEL=ollama/qwen2.5:7b \
LATTICE_RAG__AGENT_MODEL=ollama/qwen2.5:7b \
LATTICE_RAG__ROUTER_MODEL=ollama/qwen2.5:7b \
LATTICE_GROBID__URL=http://localhost:8070 \
LATTICE_EXTRACTION__MAX_INPUT_CHARS=8000 \
uv run uvicorn lattice.api.app:app --port 8000
```

Notes:
- `MAX_INPUT_CHARS=8000` matters: small models lose the output schema under very
  long contexts (the default 120k is sized for hosted frontier models). 8k covers
  abstract + intro + methodology, which is where PaperCard content lives.
- Extraction is tolerant of weak-model output quirks (fenced/prose-wrapped JSON,
  null-for-empty fields, degenerate list entries; see ARCHITECTURE.md), so a 7B
  model typically extracts in one call with zero repair rounds.
- Expect rougher extractions than hosted models (fewer datasets/results captured);
  use this mode to validate infrastructure, not extraction quality.

## Full stack

```bash
make up                         # neo4j, postgres, redis, grobid, api, worker, web
make db-schema                  # apply Postgres schema (also auto-applied on init)
make seed                       # ingest ~20 arXiv forecasting papers
make ingest FILE=paper.pdf      # ingest a local PDF
make down
```

Endpoints: API `:8000`, Web `:3000`, Neo4j browser `:7474`, GROBID `:8070`.
Compose reads `.env` by default. Set `LATTICE_ENV_FILE` to use another file,
for example `LATTICE_ENV_FILE=.env.staging docker compose up -d`.

Persistent ingestion is asynchronous. The API stores the job and PDF before it
returns 202, then an arq worker resumes the staged job. The web app polls the job
endpoint. Direct clients can do the same:

```bash
curl -H "Authorization: Bearer $LATTICE_AUTH_TOKEN" \
  -F "file=@paper.pdf" http://localhost:8000/ingest/file
curl -H "Authorization: Bearer $LATTICE_AUTH_TOKEN" \
  http://localhost:8000/ingest/jobs/JOB_ID
curl -X POST -H "Authorization: Bearer $LATTICE_AUTH_TOKEN" \
  http://localhost:8000/ingest/jobs/JOB_ID/retry
```

## Configuration

All knobs are in `backend/lattice/config.py`, overridable via `LATTICE_*` env
vars (nested with `__`, e.g. `LATTICE_SIMILARITY__ALPHA=0.45`). See `.env.example`.

Key knobs:
- `LATTICE_SIMILARITY__{ALPHA,BETA,GAMMA,DELTA,TAU}` - edge weights and threshold.
- `LATTICE_COST__{PER_JOB_USD_CAP,DAILY_USD_CAP}` - spend caps (jobs pause at cap).
- `LATTICE_EXTRACTION__{PRIMARY_MODEL,ESCALATION_MODEL,FALLBACK_MODEL}` - extraction
  models; the fallback (a different provider family) is tried once when the
  primary provider fails at the transport level (outage/timeout).
- `LATTICE_INCREMENTAL_CONTRADICTIONS` - judge new papers' claims against the
  corpus at ingest (default true; offline heuristic judge).
- `LATTICE_AUTH_TOKEN` - single-user bearer token. Production startup fails if
  this is missing. The Next.js server proxy injects it for browser requests.
- `LATTICE_CORS_ORIGINS` - JSON list of allowed browser origins. Wildcard CORS
  is rejected in production.
- `LATTICE_MAX_WORKSPACES` - maximum workspace containers cached by one process.
- `LATTICE_INGEST_MAX_ATTEMPTS` - maximum dispatch attempts for one job.
- `LATTICE_DOCLING__VISION_ENABLED` and `LATTICE_DOCLING__VISION_MODEL` - control
  image arbitration for disputed GROBID and Docling sections.
- `LATTICE_API_BASE` - FastAPI origin used by the Next.js server proxy. Keep it
  server-side; do not put the bearer token in a public browser variable.

## Operations

### Health
- `GET /health` is process liveness and does not touch dependencies.
- `GET /readyz` probes PostgreSQL, Neo4j, and Redis when persistence is enabled.
  It returns 503 if any probe fails or exceeds `LATTICE_READINESS_TIMEOUT_S`.
- Worker logs `worker.started`; ingestion logs `ingest.stage_complete` per stage.

### Reprocessing / backfill
Prompts are versioned and hashed; each card records `extraction_version`. To
re-extract after a prompt change, bump `LATTICE_EXTRACTION__PROMPT_VERSION` and
re-run ingestion for affected papers (dedup makes parsing a no-op; extraction
re-runs). Edge weights are versioned via `computed_by_version`.

### Tuning edge weights
1. Edit `LATTICE_SIMILARITY__*`.
2. `make eval` (edge-quality correlation + precision@k).
3. Keep the configuration that wins; the change is recorded in `edge_audit`.

### Cost control
Per-job and daily caps live in config. When a cap is hit, the job moves to
`PAUSED` (not `FAILED`) and can be resumed once the daily window rolls over or the
cap is raised. Per-query cost is logged and returned in the agent result.

## Failure modes (chaos-tested)

| Symptom | Cause | State / action |
| --- | --- | --- |
| `corrupted_pdf` | not a PDF / truncated | job FAILED; re-upload a valid file |
| `scanned_pdf` | no text layer | job FAILED; needs OCR/vision path |
| `paywalled_stub` | downloaded a landing page | job FAILED; obtain the real PDF |
| `duplicate` | already in corpus | job DUPLICATE (no-op, idempotent) |
| `duplicate` ("outdated preprint") | published version already ingested | job DUPLICATE; the corpus keeps the published version |
| published version of an ingested preprint | version supersession | ingests + SUPERSEDED_BY; preprint's edges invalidated, out of analytics |
| `parser_timeout` | GROBID slow | retryable; worker retries with backoff |
| `queue_unavailable` | Redis rejected dispatch | job FAILED and retryable; restore Redis, then retry |
| `internal_error` | unexpected stage crash | retryable until the attempt cap; inspect server logs |
| `cost_cap_exceeded` | spend cap hit | job PAUSED; resume after reset |
| `rate_limited` | S2/OpenAlex throttling | cached + backoff; degrades to text-only similarity |

Failed and paused jobs retain their source PDF and completed stage context in
PostgreSQL. When the job response has `retryable: true`, call
`POST /ingest/jobs/{id}/retry`; the worker resumes after the last completed
stage. Terminal input errors and jobs at the configured attempt cap return 409
and are not requeued. Retryable worker failures are also retried automatically
with exponential delay until the same cap.

## Deployment (Hetzner VPS + Cloudflare Tunnel)

1. Provision a VPS, install Docker + Compose.
2. Clone the repo, set `LATTICE_ENVIRONMENT=prod`, a strong
   `LATTICE_AUTH_TOKEN`, explicit `LATTICE_CORS_ORIGINS`, real API keys, and a
   strong `LATTICE_NEO4J__PASSWORD` in `.env`.
3. `make up`.
4. Expose `:3000` via a Cloudflare Tunnel. Expose `:8000` only if non-browser
   clients need direct API access, and keep it behind TLS and bearer auth. Never
   expose Neo4j, PostgreSQL, or Redis to the internet.

### Backups (nightly)
```bash
BACKUP_DIR=/mnt/storage RETAIN_DAYS=14 scripts/backup.sh
```
The script pauses the API and worker, writes a PostgreSQL custom archive, stops
Neo4j for the Community Edition offline dump, includes both `neo4j` and `system`,
validates every archive, and restarts only services that were running. Expect a
brief write outage. It publishes completed files atomically with mode `0600` in a
mode `0700` directory and prevents overlapping runs.

Ship all three files to a Hetzner Storage Box. Test a restore regularly with
`pg_restore` and `neo4j-admin database load`; a backup is not proven until a
restore drill succeeds.

## Observability
- Structured JSON logs (structlog) to stdout; ship to your log stack.
- The dependency-free Prometheus endpoint is available at authenticated
  `GET /metrics`; configure the scraper with `LATTICE_AUTH_TOKEN`.
- The worker generates a digest every Monday at 08:00 and stores it for the UI.
  Lattice does not send email; export or deliver the stored Markdown separately.
