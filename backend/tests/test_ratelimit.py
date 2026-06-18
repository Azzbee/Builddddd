from __future__ import annotations

from lattice.core.ratelimit import SlidingWindowLimiter


def test_allows_under_limit() -> None:
    lim = SlidingWindowLimiter(max_requests=3, window_s=60)
    for _ in range(3):
        allowed, retry = lim.allow("k", now=100.0)
        assert allowed and retry == 0.0


def test_blocks_over_limit_with_retry_after() -> None:
    lim = SlidingWindowLimiter(max_requests=2, window_s=60)
    lim.allow("k", now=100.0)
    lim.allow("k", now=101.0)
    allowed, retry = lim.allow("k", now=102.0)
    assert not allowed
    assert 0 < retry <= 60


def test_window_slides() -> None:
    lim = SlidingWindowLimiter(max_requests=1, window_s=10)
    assert lim.allow("k", now=0.0)[0]
    assert not lim.allow("k", now=5.0)[0]
    # After the window passes, the old hit expires.
    assert lim.allow("k", now=11.0)[0]


def test_keys_are_independent() -> None:
    lim = SlidingWindowLimiter(max_requests=1, window_s=60)
    assert lim.allow("a", now=0.0)[0]
    assert lim.allow("b", now=0.0)[0]  # different key, own budget
    assert not lim.allow("a", now=1.0)[0]


def test_zero_disables() -> None:
    lim = SlidingWindowLimiter(max_requests=0)
    for _ in range(100):
        assert lim.allow("k")[0]
