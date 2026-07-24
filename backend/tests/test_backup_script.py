from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _backup_environment(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    fake_docker = tmp_path / "docker"
    command_log = tmp_path / "commands.log"
    backup_dir = tmp_path / "backups"
    fake_docker.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "$COMMAND_LOG"
case "$*" in
  *" ps --status running --services") printf 'api\\nworker\\nneo4j\\npostgres\\n' ;;
  *"pg_dump"*) printf 'postgres archive' ;;
  *"pg_restore --list -"*) grep -q 'postgres archive' ;;
  *"database dump neo4j --to-stdout"*)
    [[ "${FAIL_NEO4J_DUMP:-}" == "true" ]] && exit 9
    printf 'neo4j archive'
    ;;
  *"database dump system --to-stdout"*) printf 'system archive' ;;
  *"database load neo4j --from-stdin --info"*) grep -q 'neo4j archive' ;;
  *"database load system --from-stdin --info"*) grep -q 'system archive' ;;
esac
""",
        encoding="utf-8",
    )
    fake_docker.chmod(0o700)
    script = Path(__file__).parents[2] / "scripts" / "backup.sh"
    env = {
        **os.environ,
        "BACKUP_DIR": str(backup_dir),
        "COMMAND_LOG": str(command_log),
        "COMPOSE_BIN": str(fake_docker),
        "RETAIN_DAYS": "14",
    }
    return script, command_log, backup_dir, env


def test_backup_script_creates_validated_archives_and_restores_services(tmp_path: Path) -> None:
    script, command_log, backup_dir, env = _backup_environment(tmp_path)

    result = subprocess.run(
        ["bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert [path.read_bytes() for path in backup_dir.glob("postgres-*.dump")] == [
        b"postgres archive"
    ]
    assert [path.read_bytes() for path in backup_dir.glob("neo4j-[0-9]*.dump")] == [
        b"neo4j archive"
    ]
    assert [path.read_bytes() for path in backup_dir.glob("neo4j-system-*.dump")] == [
        b"system archive"
    ]
    log = command_log.read_text(encoding="utf-8")
    assert "stop api worker" in log
    assert "stop neo4j" in log
    assert "start neo4j" in log
    assert "start api worker" in log
    assert "database load neo4j --from-stdin --info" in log
    assert not (backup_dir / ".backup.lock").exists()


def test_backup_script_restores_services_and_publishes_nothing_after_failure(
    tmp_path: Path,
) -> None:
    script, command_log, backup_dir, env = _backup_environment(tmp_path)
    env["FAIL_NEO4J_DUMP"] = "true"

    result = subprocess.run(
        ["bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 9
    assert list(backup_dir.glob("*.dump")) == []
    log = command_log.read_text(encoding="utf-8")
    assert "start neo4j" in log
    assert "start api worker" in log
    assert not (backup_dir / ".backup.lock").exists()


def test_backup_script_rejects_invalid_retention_without_running_docker(tmp_path: Path) -> None:
    script = Path(__file__).parents[2] / "scripts" / "backup.sh"
    result = subprocess.run(
        ["bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "BACKUP_DIR": str(tmp_path), "RETAIN_DAYS": "never"},
    )

    assert result.returncode == 2
    assert "non-negative integer" in result.stderr
