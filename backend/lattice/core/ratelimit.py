"""In-process sliding-window rate limiter.

Dependency-free and thread-safe, keyed by an arbitrary identity (client IP or auth
token). Used by the API middleware to cap request rate per client. For a
multi-process deployment behind a load balancer, swap the backing store for Redis
behind the same ``allow`` interface; the algorithm is identical.
"""

from __future__ import annotations

import threading
import time
from collections import deque


class SlidingWindowLimiter:
    def __init__(self, max_requests: int, window_s: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds). retry_after is 0 when allowed."""
        if self.max_requests <= 0:
            return True, 0.0
        now = time.monotonic() if now is None else now
        cutoff = now - self.window_s
        with self._lock:
            dq = self._hits.setdefault(key, deque())
            while dq and dq[0] <= cutoff:
                dq.popleft()
            if len(dq) >= self.max_requests:
                retry_after = dq[0] + self.window_s - now
                return False, max(0.0, retry_after)
            dq.append(now)
            return True, 0.0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
