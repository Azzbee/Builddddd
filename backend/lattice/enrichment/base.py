"""Shared HTTP plumbing for enrichment clients: caching + backoff.

External APIs (S2, OpenAlex, Crossref, arXiv) are rate-limited and occasionally
miss. The :class:`CachedHTTPClient` adds a pluggable cache (so repeated lookups
during development cost nothing), exponential backoff that honors ``Retry-After``,
and maps responses to typed errors:

* 404           -> :class:`NotFoundError` (terminal; the record does not exist)
* 429 / 5xx     -> retried, then :class:`RateLimitedError` (retryable upstream)

The system degrades gracefully: if enrichment fails, the caller records the gap in
``weight_components`` and falls back to text-only similarity.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Protocol

import httpx

from lattice.core.errors import EnrichmentError, NotFoundError, RateLimitedError
from lattice.core.logging import get_logger

log = get_logger("enrichment")


class Cache(Protocol):
    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, ttl_s: int) -> None: ...


class InMemoryCache:
    """Process-local TTL cache. Swapped for Redis in production via the protocol."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, Any]] = {}

    async def get(self, key: str) -> Any | None:
        item = self._store.get(key)
        if item is None:
            return None
        expires, value = item
        if expires < time.time():
            self._store.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: Any, ttl_s: int) -> None:
        self._store[key] = (time.time() + ttl_s, value)


class CachedHTTPClient:
    """A small async GET client with caching and retry/backoff."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        cache: Cache | None = None,
        max_retries: int = 4,
        backoff_base_s: float = 1.0,
        cache_ttl_s: int = 60 * 60 * 24,
        default_headers: dict[str, str] | None = None,
    ):
        self._client = client
        self._cache = cache or InMemoryCache()
        self._max_retries = max_retries
        self._backoff_base = backoff_base_s
        self._ttl = cache_ttl_s
        self._headers = default_headers or {}

    async def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        cache_key: str | None = None,
    ) -> Any:
        return await self._fetch(url, params, cache_key=cache_key, as_text=False)

    async def get_text(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        cache_key: str | None = None,
    ) -> str:
        result = await self._fetch(url, params, cache_key=cache_key, as_text=True)
        return str(result)

    async def _fetch(
        self,
        url: str,
        params: dict[str, Any] | None,
        *,
        cache_key: str | None,
        as_text: bool,
    ) -> Any:
        key = (cache_key or _cache_key(url, params)) + (":text" if as_text else "")
        cached = await self._cache.get(key)
        if cached is not None:
            return cached

        client = self._client or httpx.AsyncClient(timeout=30.0)
        owns = self._client is None
        try:
            resp = await self._get_with_retry(client, url, params)
            data: Any = resp.text if as_text else resp.json()
        finally:
            if owns:
                await client.aclose()

        await self._cache.set(key, data, self._ttl)
        return data

    async def _get_with_retry(
        self, client: httpx.AsyncClient, url: str, params: dict[str, Any] | None
    ) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = await client.get(url, params=params, headers=self._headers)
            except httpx.HTTPError as exc:  # network-level
                last_exc = exc
                await self._sleep(attempt, None)
                continue

            if resp.status_code == 404:
                raise NotFoundError(f"{url} -> 404")
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = _retry_after(resp)
                log.warning(
                    "enrichment.retry",
                    url=url,
                    status=resp.status_code,
                    attempt=attempt,
                    retry_after=retry_after,
                )
                await self._sleep(attempt, retry_after)
                last_exc = RateLimitedError(f"{url} -> {resp.status_code}")
                continue
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise EnrichmentError(f"{url} -> {resp.status_code}") from exc
            return resp

        if isinstance(last_exc, RateLimitedError):
            raise last_exc
        raise RateLimitedError(f"{url} failed after {self._max_retries} attempts: {last_exc}")

    async def _sleep(self, attempt: int, retry_after: float | None) -> None:
        delay = retry_after if retry_after is not None else self._backoff_base * (2**attempt)
        await asyncio.sleep(min(delay, 30.0))


def _retry_after(resp: httpx.Response) -> float | None:
    val = resp.headers.get("Retry-After")
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _cache_key(url: str, params: dict[str, Any] | None) -> str:
    if not params:
        return url
    items = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"{url}?{items}"
