"""Tests for the per-model sliding-window rate limiter."""

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from atl.agent.llm_cache import CachedCompletion
from atl.agent.rate_limit import ModelLimit, RateLimiter, estimate_input_tokens

M = "gemini/gemma-4-31b-it"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s


def limiter(clock: FakeClock, **limit: float) -> RateLimiter:
    return RateLimiter({M: ModelLimit(**limit)}, clock=clock, sleep=clock.sleep)


def test_under_the_cap_never_waits() -> None:
    clock = FakeClock()
    rl = limiter(clock, input_tpm=1000)
    rl.acquire(M, 400)
    rl.acquire(M, 500)
    assert clock.sleeps == []


def test_waits_until_oldest_tokens_leave_the_window() -> None:
    clock = FakeClock()
    rl = limiter(clock, input_tpm=1000)
    rl.acquire(M, 600)  # t = 0
    clock.now = 10.0
    rl.acquire(M, 600)  # 1,200 > 1,000: must wait until t = 60, when the first one expires
    assert clock.sleeps == [pytest.approx(50.0)]
    assert clock.now == pytest.approx(60.0)


def test_requests_per_minute() -> None:
    clock = FakeClock()
    rl = limiter(clock, rpm=2)
    rl.acquire(M, 1)
    clock.now = 1.0
    rl.acquire(M, 1)
    clock.now = 2.0
    rl.acquire(M, 1)  # third in the window: waits until t = 60
    assert clock.now == pytest.approx(60.0)


def test_reconcile_frees_overestimated_tokens() -> None:
    clock = FakeClock()
    rl = limiter(clock, input_tpm=1000)
    handle = rl.acquire(M, 900)
    rl.reconcile(handle, 300)  # the provider reported far fewer tokens than estimated
    clock.now = 1.0
    rl.acquire(M, 600)  # 300 + 600 fits: no wait
    assert clock.sleeps == []


def test_unlimited_model_passes_straight_through() -> None:
    clock = FakeClock()
    rl = limiter(clock, input_tpm=10)
    assert rl.acquire("ollama_chat/qwen3:8b", 10_000) is None
    assert clock.sleeps == []


def test_single_request_larger_than_cap_is_not_deadlocked() -> None:
    clock = FakeClock()
    rl = limiter(clock, input_tpm=1000)
    rl.acquire(M, 5000)  # empty window: allowed
    clock.now = 1.0
    rl.acquire(M, 10)  # but the next one waits for it to expire
    assert clock.now == pytest.approx(60.0)


def test_parallel_callers_never_exceed_the_cap() -> None:
    rl = RateLimiter({M: ModelLimit(input_tpm=1000)}, window_s=0.3)
    grants: list[tuple[float, int]] = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(3):
            rl.acquire(M, 400)
            with lock:
                grants.append((time.monotonic(), 400))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(grants) == 12
    for t0, _ in grants:  # tokens granted inside any window that starts at a grant
        in_window = sum(tok for t, tok in grants if t0 <= t < t0 + 0.3 - 0.02)
        assert in_window <= 1000


def test_estimate_overcounts_at_3_5_chars_per_token() -> None:
    kwargs = {"messages": [{"role": "user", "content": "x" * 350}]}
    chars = len('[{"role": "user", "content": "' + "x" * 350 + '"}]')
    assert estimate_input_tokens(kwargs) == int(chars / 3.5) + 1


def test_cached_completion_uses_and_reconciles_the_limiter(tmp_path: Path) -> None:
    calls: list[tuple[str, Any]] = []

    class SpyLimiter(RateLimiter):
        def acquire(self, model: str, est_tokens: int) -> list[float] | None:
            calls.append(("acquire", (model, est_tokens)))
            return [0.0, float(est_tokens)]

        def reconcile(self, handle: list[float] | None, actual_tokens: int | None) -> None:
            calls.append(("reconcile", actual_tokens))

    cached = CachedCompletion(
        lambda **kw: {"usage": {"prompt_tokens": 42}},
        cache_dir=tmp_path / "c",
        cost_log=tmp_path / "costs.jsonl",
        budget_usd=1.0,
        cost_fn=lambda r: 0.0,
        to_dict=lambda r: r,
        from_dict=lambda d: d,
        rate_limiter=SpyLimiter({}),
    )
    req = {"model": M, "messages": [{"role": "user", "content": "hi"}]}
    cached(**req)
    cached(**req)  # cache hit: the limiter is not consulted again
    assert [c[0] for c in calls] == ["acquire", "reconcile"]
    assert calls[0][1][0] == M and calls[1][1] == 42
