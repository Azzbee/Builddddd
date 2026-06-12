"""Datastore implementations: vectors (pgvector), cards, jobs.

In-memory implementations make the whole stack runnable and testable offline; the
Postgres-backed implementations share the same protocols for production.
"""
