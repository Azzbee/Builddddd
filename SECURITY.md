# Security policy

## Reporting

Email security reports to adamzeraiki@gmail.com. Please do not open public issues
for vulnerabilities. You will get an acknowledgement within a few days.

## Security posture

- **Auth**: single-user bearer token (`LATTICE_AUTH_TOKEN`). Production startup
  fails if it is missing. All routes except liveness and readiness require the
  configured token. The workspace listing and metrics endpoint are authenticated.
- **API schema**: Swagger, ReDoc, and the OpenAPI document are disabled in
  production. They remain available in development.
- **Browser secret handling**: browser requests use the same-origin Next.js
  `/api` route. The server removes browser-provided authorization and injects the
  bearer token from its private environment. The token is never sent in a
  JavaScript bundle, URL, local storage, or public environment variable.
- **CORS**: origins come from `LATTICE_CORS_ORIGINS`; production rejects a
  wildcard. The default permits only `http://localhost:3000`.
- **Workspace isolation**: workspace IDs are validated before store lookup and
  one process caches at most `LATTICE_MAX_WORKSPACES` containers.
- **Rate limiting**: per-client sliding-window limiter on the API
  (`LATTICE_RATE_LIMIT_PER_MIN`). Only the configured bearer token can identify
  an authenticated client, so invented authorization headers cannot rotate the
  rate-limit key.
- **Cypher safety**: the agent's `run_cypher` tool is validated against a read-only
  allowlist (writes, multiple statements, and missing RETURN are rejected) and a
  LIMIT is enforced. Edge-type identifiers in traversals are whitelisted to prevent
  label injection.
- **SQL safety**: all queries are parameterized (no string interpolation), checked
  statically in `tests/test_sql.py`.
- **Secrets**: provided via environment variables only; `.env` is gitignored.
  Datastores (Neo4j/Postgres/Redis) must not be exposed to the public internet;
  expose only the API and web behind a tunnel/reverse proxy.
- **Cost controls**: per-job and daily LLM spend caps pause work rather than
  overspend.
- **Privacy**: uploaded PDFs are stored privately and never re-published; exports
  contain only structured notes and short quotes.
- **Dependency checks**: CI runs `npm audit` at moderate severity and builds from
  the lockfile with `npm ci`. The current web dependency tree has zero reported
  vulnerabilities.

## Hardening checklist for production

- Set `LATTICE_ENVIRONMENT=prod`, a strong `LATTICE_AUTH_TOKEN`, and a strong
  `LATTICE_NEO4J__PASSWORD`.
- Put the API behind TLS (Cloudflare Tunnel or a reverse proxy).
- Set an explicit `LATTICE_CORS_ORIGINS` list for the deployed web origin.
- Keep `LATTICE_MAX_WORKSPACES` bounded for the available memory.
- Run nightly backups (`scripts/backup.sh`).
