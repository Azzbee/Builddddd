"""Static validation of the Postgres SQL: schema DDL + PgVectorStore queries.

Parses every statement with sqlglot's Postgres dialect so a malformed query is
caught offline, before it ever reaches a database. This complements the live
integration tests (tests/integration/) that run against a real pgvector instance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

sqlglot = pytest.importorskip("sqlglot")

from lattice.db.blobs import BLOB_SQL  # noqa: E402
from lattice.db.pg_stores import PG_STORE_SQL  # noqa: E402
from lattice.db.vector import PGVECTOR_SQL  # noqa: E402

SCHEMA = Path(__file__).parent.parent / "lattice" / "db" / "schema.sql"
ALL_SQL = {**PGVECTOR_SQL, **PG_STORE_SQL, **BLOB_SQL}


def test_schema_sql_parses() -> None:
    statements = [s for s in sqlglot.parse(SCHEMA.read_text(), read="postgres") if s is not None]
    # The schema has tables, indexes, and extensions.
    assert len(statements) >= 15
    rendered = "\n".join(s.sql(dialect="postgres") for s in statements)
    assert "papers" in rendered.lower()
    assert "chunks" in rendered.lower()


@pytest.mark.parametrize("name", sorted(ALL_SQL))
def test_store_queries_parse(name: str) -> None:
    # Each store query must parse as valid Postgres (vector <=> operator included).
    parsed = sqlglot.parse_one(ALL_SQL[name], read="postgres")
    assert parsed is not None


def test_store_queries_are_parameterized() -> None:
    # Every query uses bind parameters ($1...), never string interpolation.
    for sql in ALL_SQL.values():
        assert "$1" in sql
        assert "%s" not in sql  # no printf-style formatting that could enable injection
