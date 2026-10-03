"""Tests for the on-disk LLM cache, cost log, budget cap and rate limit."""

import json
from pathlib import Path
from typing import Any

import pytest

from atl.agent.llm_cache import BudgetExceeded, CachedCompletion, cache_key, cache_namespace


class FakeLLM:
    """Counts calls; each response costs `cost` dollars."""

    def __init__(self, cost: float = 0.25) -> None:
        self.calls = 0
        self.cost = cost

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        return {"text": f"reply {self.calls}", "usage": {"prompt_tokens": 10}, "cost": self.cost}


def make(tmp_path: Path, llm: FakeLLM, budget: float = 10.0, **kw: Any) -> CachedCompletion:
    return CachedCompletion(
        llm,
        cache_dir=tmp_path / "cache",
        cost_log=tmp_path / "costs.jsonl",
        budget_usd=budget,
        cost_fn=lambda r: r["cost"],
        to_dict=lambda r: r,
        from_dict=lambda d: d,
        **kw,
    )


REQ = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0}


def req(**override: Any) -> dict[str, Any]:
    return {**REQ, **override}


def test_second_identical_call_is_cached(tmp_path: Path) -> None:
    llm = FakeLLM()
    cached = make(tmp_path, llm)
    first = cached(**REQ)
    second = cached(**REQ)
    assert first == second
    assert llm.calls == 1
    assert (cached.misses, cached.hits) == (1, 1)


def test_cache_survives_restart(tmp_path: Path) -> None:
    llm = FakeLLM()
    make(tmp_path, llm)(**REQ)
    make(tmp_path, llm)(**REQ)  # new instance, same directory
    assert llm.calls == 1


def test_namespace_separates_trials(tmp_path: Path) -> None:
    llm = FakeLLM()
    cached = make(tmp_path, llm)
    for trial in ("task1/0", "task1/1"):
        token = cache_namespace.set(trial)
        try:
            cached(**REQ)
        finally:
            cache_namespace.reset(token)
    assert llm.calls == 2


def test_transport_kwargs_do_not_change_key() -> None:
    assert cache_key(req(num_retries=3, timeout=30), "") == cache_key(REQ, "")
    assert cache_key(req(temperature=1.0), "") != cache_key(REQ, "")


def test_cost_log_records_paid_and_cached_calls(tmp_path: Path) -> None:
    cached = make(tmp_path, FakeLLM(cost=0.25))
    cached(**REQ)
    cached(**REQ)
    lines = [json.loads(x) for x in (tmp_path / "costs.jsonl").read_text().splitlines()]
    assert [(e["cached"], e["cost_usd"]) for e in lines] == [(False, 0.25), (True, 0.0)]
    assert cached.spent_usd == 0.25


def test_budget_stops_paid_calls(tmp_path: Path) -> None:
    llm = FakeLLM(cost=0.6)
    cached = make(tmp_path, llm, budget=1.0)
    cached(**req(messages=[{"role": "user", "content": "a"}]))  # spent 0.6
    cached(**req(messages=[{"role": "user", "content": "b"}]))  # spent 1.2 (starts under cap)
    with pytest.raises(BudgetExceeded):
        cached(**req(messages=[{"role": "user", "content": "c"}]))
    assert llm.calls == 2


def test_budget_persists_across_sessions(tmp_path: Path) -> None:
    make(tmp_path, FakeLLM(cost=1.5), budget=1.0)(**REQ)
    restarted = make(tmp_path, FakeLLM(), budget=1.0)
    assert restarted.spent_usd == 1.5
    with pytest.raises(BudgetExceeded):
        restarted(**req(temperature=0.5))


def test_cache_hit_allowed_over_budget(tmp_path: Path) -> None:
    make(tmp_path, FakeLLM(cost=1.5), budget=1.0)(**REQ)
    assert make(tmp_path, FakeLLM(), budget=1.0)(**REQ)["text"] == "reply 1"


def test_unpriceable_response_logs_zero(tmp_path: Path) -> None:
    cached = CachedCompletion(
        FakeLLM(),
        cache_dir=tmp_path / "c",
        cost_log=tmp_path / "costs.jsonl",
        budget_usd=1.0,
        cost_fn=lambda r: 1 / 0,
        to_dict=lambda r: r,
        from_dict=lambda d: d,
    )
    cached(**REQ)
    assert cached.spent_usd == 0.0


def test_rate_limit_spaces_paid_calls(tmp_path: Path) -> None:
    now = [100.0]
    sleeps: list[float] = []

    def sleep(s: float) -> None:
        sleeps.append(s)
        now[0] += s

    cached = make(tmp_path, FakeLLM(), requests_per_minute=6, clock=lambda: now[0], sleep=sleep)
    cached(**req(temperature=0.1))
    now[0] += 4.0
    cached(**req(temperature=0.2))  # 10 s interval, 4 s elapsed -> waits 6 s
    cached(**req(temperature=0.1))  # cache hit: no wait
    assert sleeps == [pytest.approx(6.0)]


def test_paid_calls_run_in_parallel(tmp_path: Path) -> None:
    import threading
    import time

    class SlowLLM(FakeLLM):
        def __call__(self, **kwargs: Any) -> dict[str, Any]:
            time.sleep(0.3)
            return super().__call__(**kwargs)

    cached = make(tmp_path, SlowLLM(cost=0.0))
    threads = [threading.Thread(target=cached, kwargs=req(temperature=i / 10)) for i in range(4)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert cached.misses == 4
    assert time.perf_counter() - start < 0.9  # serialised would take >= 1.2 s
    assert len((tmp_path / "costs.jsonl").read_text().splitlines()) == 4
