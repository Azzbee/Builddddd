"""Watch-queue and digest stores (in-memory + protocols).

Like the card/job stores, these sit behind protocols with an in-memory default and
a Postgres-backed implementation (db/pg_stores.py). Persisting them is what lets the
worker (which queues arXiv matches and builds digests) and the API (which serves
them) share state across processes in production.
"""

from __future__ import annotations

from typing import Any, Protocol


class WatchStore(Protocol):
    async def enqueue(self, items: list[dict[str, Any]]) -> int: ...
    async def pending(self) -> list[dict[str, Any]]: ...
    async def set_status(self, arxiv_id: str, status: str) -> bool: ...


class DigestStore(Protocol):
    async def add(self, report: dict[str, Any]) -> None: ...
    async def latest(self) -> dict[str, Any] | None: ...
    async def history(self) -> list[dict[str, Any]]: ...


class InMemoryWatchStore:
    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}

    async def enqueue(self, items: list[dict[str, Any]]) -> int:
        added = 0
        for item in items:
            aid = str(item.get("arxiv_id"))
            if aid and aid not in self._items:
                self._items[aid] = {**item, "status": item.get("status", "pending")}
                added += 1
        return added

    async def pending(self) -> list[dict[str, Any]]:
        return [i for i in self._items.values() if i.get("status") == "pending"]

    async def set_status(self, arxiv_id: str, status: str) -> bool:
        item = self._items.get(arxiv_id)
        if item is None:
            return False
        item["status"] = status
        return True


class InMemoryDigestStore:
    def __init__(self) -> None:
        self._digests: list[dict[str, Any]] = []

    async def add(self, report: dict[str, Any]) -> None:
        self._digests.append(report)

    async def latest(self) -> dict[str, Any] | None:
        return self._digests[-1] if self._digests else None

    async def history(self) -> list[dict[str, Any]]:
        return list(reversed(self._digests))
