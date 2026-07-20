# Lattice production hardening

## Objective

Close the gaps between the documented architecture and the running system. The result must preserve the offline demo, make production ingestion genuinely asynchronous and resumable, wire the parser fallbacks into the live path, tighten request boundaries, and keep the backend and frontend contracts typed and tested.

## Constraints

- Preserve the existing `web/tsconfig.json` change.
- Keep demo and offline tests free of Redis, PostgreSQL, Neo4j, Docling, model downloads, and network calls.
- Use PostgreSQL as the durable source of job and ingest-artifact state in persistent mode.
- Do not put PDF bytes into Redis job arguments.
- Keep every migration idempotent because `schema.sql` runs at application startup.
- Commit each logical change separately. Run focused tests before each commit and the full verification gate before pushing.

## Change 1: durable resumable ingestion

1. Add an ingest-artifact persistence boundary for the raw PDF and stage outputs.
2. Add PostgreSQL and in-memory implementations. Store the parsed document, extracted card, chunks, enrichment payload, and computed vectors in typed JSON fields.
3. Pass the job store and artifact store into `IngestionService` and `IngestionPipeline`.
4. Make `ingest_pdf` create or resume the deterministic job rather than replacing it.
5. Rehydrate `PipelineContext` from persisted artifacts before selecting the next stage.
6. Make paused and failed jobs explicitly resumable, enforce `max_attempts`, and add a retry service/API path.
7. Persist each completed stage before moving to the next stage. Preserve idempotent linking for crash recovery.
8. Persist `papers.content_hash` and use it when rebuilding the corpus dedup index.
9. Add unit and PostgreSQL integration tests for stage persistence, restart recovery, retry limits, and content-hash dedup after restart.

Commit: `Fix durable ingestion resume and persistent dedup`

## Change 2: real arq dispatch

1. Add an ingestion dispatcher protocol with inline and arq implementations.
2. In persistent mode, stage the PDF and queued job in PostgreSQL, then enqueue only workspace and job identifiers.
3. Return `202 Accepted` immediately from file and arXiv ingestion endpoints.
4. Make the worker load the workspace-scoped container and resume the staged job.
5. Add retry behavior with bounded arq attempts and deterministic job IDs.
6. Hydrate worker feature state before arXiv watcher scoring.
7. Close persistent resources during worker shutdown.
8. Add API and worker tests proving dispatch, status polling, retry, duplicate dispatch idempotency, and inline demo behavior.

Commit: `Route production ingestion through arq`

## Change 3: live Docling and vision routing

1. Introduce a hybrid parser that wraps GROBID, optional Docling extraction, reconciliation, and optional vision arbitration.
2. Run blocking Docling conversion off the event loop and use temporary files with guaranteed cleanup.
3. Attach Docling tables to `ParsedDocument` and reconcile section text by normalized headings and ordered fallback matching.
4. Render or obtain the relevant PDF page only when reconciliation requires vision. If the rendering dependency is unavailable, keep the higher-information parse and mark low confidence.
5. Add explicit configuration for the vision model and parser fallback behavior.
6. Wire the hybrid parser in `build_container`; preserve `DemoParser` for demo mode.
7. Add end-to-end parser tests covering disabled/unavailable Docling, accepted reconciliation, low-confidence fallback, vision arbitration, tables, and cleanup.

Commit: `Wire Docling and vision into production parsing`

## Change 4: production request boundaries

1. Validate workspace IDs by length and character set. Bound the in-process registry and reject new IDs once the configured cap is reached.
2. Require auth for `/workspaces` and avoid exposing workspace names through health endpoints.
3. Key unauthenticated rate limits by client address, never by an unverified authorization value. Use a constant authenticated key only after token comparison succeeds.
4. Make CORS origins configurable and reject wildcard CORS in production.
5. Require an auth token when `environment=prod`.
6. Replace the browser-facing Next.js rewrite with a server-side route proxy that injects `LATTICE_AUTH_TOKEN`, forwards the workspace header, preserves streaming responses and PDFs, and never exposes the token to browser JavaScript.
7. Add FastAPI and Next.js tests for auth, workspace validation, rate-limit rotation, proxy streaming, uploads, and binary responses.

Commit: `Harden auth workspace and proxy boundaries`

## Change 5: operational truth and documentation

1. Make readiness probe configured persistent dependencies instead of reporting ready unconditionally.
2. Update the README, architecture, API, runbook, security policy, environment example, and Docker wiring to match the final behavior.
3. Update stale test and route counts from generated results.
4. Add or update CI steps for frontend proxy tests, backend coverage, and live persistence tests.
5. Run the offline eval, full backend gate, frontend lint/type-check/tests/build, and any available live integration suite.
6. Run the `slop-scan` checklist against the complete diff and fix real findings.
7. Run an adversarial pass focused on crash boundaries, duplicate dispatch, workspace isolation, malformed parser output, and partial datastore failure.

Commit: `Align operations docs and verification with production behavior`

## Acceptance criteria

- A queued ingest returns before parsing begins in persistent mode.
- Killing a worker after any completed stage and retrying continues from the next stage with the required artifacts available.
- Repeated upload or dispatch cannot create duplicate jobs, papers, graph edges, PDFs, or edge-audit rows.
- Docling output reaches extraction when enabled. Low-agreement regions either go through vision or remain explicitly low confidence.
- The browser can use a token-protected backend without receiving the bearer token.
- Random authorization and workspace headers cannot bypass rate limits or allocate unbounded containers.
- Ruff, Ruff format, strict mypy, pytest with coverage, offline eval, frontend lint, TypeScript, frontend tests, and production build pass.
- Git history contains one commit per change above and the branch is pushed.
