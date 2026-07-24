#!/usr/bin/env bash
# Application-consistent backup for the Docker Compose deployment.
# Usage: BACKUP_DIR=/mnt/storage scripts/backup.sh
set -euo pipefail

umask 077

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-$ROOT_DIR/backups}"
RETAIN_DAYS="${RETAIN_DAYS:-14}"
STAMP="$(date +%F-%H%M%S)"

if [[ ! "$RETAIN_DAYS" =~ ^[0-9]+$ ]]; then
  echo "RETAIN_DAYS must be a non-negative integer" >&2
  exit 2
fi

COMPOSE=("${COMPOSE_BIN:-docker}" compose --project-directory "$ROOT_DIR" -f "$ROOT_DIR/docker-compose.yml")
if [[ -n "${LATTICE_ENV_FILE:-}" ]]; then
  COMPOSE+=(--env-file "$LATTICE_ENV_FILE")
fi

mkdir -p -- "$BACKUP_DIR"
chmod 700 "$BACKUP_DIR"

LOCK_DIR="$BACKUP_DIR/.backup.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another backup is already running: $LOCK_DIR" >&2
  exit 1
fi
WORK_DIR="$(mktemp -d "$BACKUP_DIR/.backup-$STAMP.XXXXXX")"

running_services=()
while IFS= read -r service; do
  [[ -n "$service" ]] && running_services+=("$service")
done < <("${COMPOSE[@]}" ps --status running --services)

was_running() {
  local wanted="$1"
  local service
  for service in "${running_services[@]}"; do
    [[ "$service" == "$wanted" ]] && return 0
  done
  return 1
}

stopped_writers=()
neo4j_stopped=false

restore_services() {
  local restore_failed=0
  if [[ "$neo4j_stopped" == true ]]; then
    "${COMPOSE[@]}" start neo4j || restore_failed=1
    neo4j_stopped=false
  fi
  if ((${#stopped_writers[@]})); then
    "${COMPOSE[@]}" start "${stopped_writers[@]}" || restore_failed=1
    stopped_writers=()
  fi
  return "$restore_failed"
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if ! restore_services; then
    echo "Backup finished, but one or more services failed to restart" >&2
    status=1
  fi
  rm -rf -- "$WORK_DIR"
  rmdir "$LOCK_DIR" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT INT TERM

for service in api worker; do
  if was_running "$service"; then
    stopped_writers+=("$service")
  fi
done
if ((${#stopped_writers[@]})); then
  echo "[$(date -Is)] Pausing application writers..."
  "${COMPOSE[@]}" stop "${stopped_writers[@]}"
fi

echo "[$(date -Is)] Backing up PostgreSQL..."
"${COMPOSE[@]}" exec -T postgres pg_dump -U lattice --format=custom lattice \
  >"$WORK_DIR/postgres.dump"
"${COMPOSE[@]}" exec -T postgres pg_restore --list - \
  <"$WORK_DIR/postgres.dump" >/dev/null

if was_running neo4j; then
  echo "[$(date -Is)] Stopping Neo4j for a Community Edition offline dump..."
  "${COMPOSE[@]}" stop neo4j
  neo4j_stopped=true
fi

for database in neo4j system; do
  echo "[$(date -Is)] Backing up Neo4j database: $database..."
  "${COMPOSE[@]}" run --rm --no-deps -T neo4j \
    neo4j-admin database dump "$database" --to-stdout \
    >"$WORK_DIR/$database.dump"
  "${COMPOSE[@]}" run --rm --no-deps -T neo4j \
    neo4j-admin database load "$database" --from-stdin --info \
    <"$WORK_DIR/$database.dump" >/dev/null
done

mv "$WORK_DIR/postgres.dump" "$BACKUP_DIR/postgres-$STAMP.dump"
mv "$WORK_DIR/neo4j.dump" "$BACKUP_DIR/neo4j-$STAMP.dump"
mv "$WORK_DIR/system.dump" "$BACKUP_DIR/neo4j-system-$STAMP.dump"

echo "[$(date -Is)] Pruning backups older than $RETAIN_DAYS days..."
find "$BACKUP_DIR" -type f \
  \( -name 'postgres-*.dump' -o -name 'neo4j-*.dump' -o -name 'neo4j-system-*.dump' \) \
  -mtime "+$RETAIN_DAYS" -delete

echo "[$(date -Is)] Backup complete: $BACKUP_DIR"
