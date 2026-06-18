#!/usr/bin/env bash
# Nightly backup of Neo4j + Postgres to a target directory.
# Usage: BACKUP_DIR=/mnt/storage scripts/backup.sh
# Cron:  0 3 * * *  BACKUP_DIR=/mnt/storage /path/to/scripts/backup.sh >> /var/log/lattice-backup.log 2>&1
set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-./backups}"
STAMP="$(date +%F-%H%M%S)"
COMPOSE="${COMPOSE:-docker compose}"
mkdir -p "$BACKUP_DIR"

echo "[$(date -Is)] Backing up Postgres..."
$COMPOSE exec -T postgres pg_dump -U lattice lattice | gzip > "$BACKUP_DIR/pg-$STAMP.sql.gz"

echo "[$(date -Is)] Backing up Neo4j..."
$COMPOSE exec -T neo4j neo4j-admin database dump neo4j --to-stdout > "$BACKUP_DIR/neo4j-$STAMP.dump"

echo "[$(date -Is)] Pruning backups older than ${RETAIN_DAYS:-14} days..."
find "$BACKUP_DIR" -name 'pg-*.sql.gz' -mtime "+${RETAIN_DAYS:-14}" -delete
find "$BACKUP_DIR" -name 'neo4j-*.dump' -mtime "+${RETAIN_DAYS:-14}" -delete

echo "[$(date -Is)] Backup complete -> $BACKUP_DIR"
