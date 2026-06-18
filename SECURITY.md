# Security policy

## Reporting

Email security reports to adamzeraiki@gmail.com. Please do not open public issues
for vulnerabilities. You will get an acknowledgement within a few days.

## Security posture

- **Auth**: single-user bearer token (`LATTICE_AUTH_TOKEN`). All non-health routes
  require it when set. Multi-user auth (Supabase) is schema-ready.
- **Rate limiting**: per-client sliding-window limiter on the API
  (`LATTICE_RATE_LIMIT_PER_MIN`).
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

## Hardening checklist for production

- Set a strong `LATTICE_AUTH_TOKEN` and `LATTICE_NEO4J__PASSWORD`.
- Put the API behind TLS (Cloudflare Tunnel or a reverse proxy).
- Restrict CORS origins (currently permissive for development).
- Run nightly backups (`scripts/backup.sh`).
