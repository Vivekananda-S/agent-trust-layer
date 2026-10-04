"""Tests for self-hosted serving helpers: proxy routing/auth, GPU ledger, config expansion."""

from pathlib import Path

import pytest

from atl.agent.run_tau2 import AgentRunConfig, expand_env, is_transient, session_time_limit
from atl.serving.ledger import (
    BudgetExhausted,
    GpuBudget,
    record_session,
    session_seconds,
    spent_usd,
)
from atl.serving.proxy import UPSTREAMS, authorized, route


def test_route_by_model_name() -> None:
    assert route("qwen3-8b", UPSTREAMS) == "http://127.0.0.1:8001"
    assert route("gemma-4-31b", UPSTREAMS) == "http://127.0.0.1:8002"
    assert route("gpt-x", UPSTREAMS) is None


def test_bearer_auth() -> None:
    assert authorized("Bearer s3cret", "s3cret")
    assert not authorized("Bearer wrong", "s3cret")
    assert not authorized("s3cret", "s3cret")  # missing "Bearer "
    assert not authorized(None, "s3cret")
    assert not authorized("Bearer ", "")  # an empty key never authorises


def budget(tmp_path: Path) -> GpuBudget:
    return GpuBudget(
        gpu="H100", hourly_usd=3.95, cap_usd=27.0, overhead_s=900, ledger=tmp_path / "ledger.jsonl"
    )


def test_ledger_math(tmp_path: Path) -> None:
    b = budget(tmp_path)
    # 27 / 3.95 h = 24,607.59 s, minus 900 s overhead
    assert session_seconds(b) == pytest.approx(23707.59, abs=0.01)
    # a 1-hour session is charged (3600 + 900) s at $3.95/h = $4.9375
    assert record_session(b, "run", 3600.0) == pytest.approx(4.9375)
    assert spent_usd(b) == pytest.approx(4.9375, abs=1e-4)
    # remaining 22.0625 / 3.95 h = 20,107.59 s, minus 900
    assert session_seconds(b) == pytest.approx(19207.59, abs=0.1)


def test_ledger_refuses_when_spent(tmp_path: Path) -> None:
    b = budget(tmp_path)
    record_session(b, "big", 6 * 3600.0)  # (21600 + 900) s -> $24.69
    record_session(b, "more", 1800.0)  # (1800 + 900) s -> $2.96; total $27.65 > cap
    with pytest.raises(BudgetExhausted):
        session_seconds(b)


def test_expand_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATL_SERVING_URL", "https://x.modal.run")
    monkeypatch.setenv("ATL_SERVING_KEY", "k")
    args = {
        "api_base": "${ATL_SERVING_URL}/v1",
        "api_key": "${ATL_SERVING_KEY}",
        "timeout": 120,
        "extra_body": {"note": "${ATL_SERVING_KEY}"},
    }
    assert expand_env(args) == {
        "api_base": "https://x.modal.run/v1",
        "api_key": "k",
        "timeout": 120,
        "extra_body": {"note": "k"},
    }
    monkeypatch.delenv("ATL_SERVING_KEY")
    with pytest.raises(ValueError, match="unset environment variable"):
        expand_env({"api_key": "${ATL_SERVING_KEY}"})


def _cfg(**kw: object) -> AgentRunConfig:
    base = {"run_name": "r", "domain": "retail", "agent_model": "a", "user_model": "u"}
    return AgentRunConfig.model_validate({**base, "budget_usd": 1.0, **kw})


def test_session_time_limit_takes_the_smaller(tmp_path: Path) -> None:
    assert session_time_limit(_cfg()) is None
    assert session_time_limit(_cfg(max_session_s=1200)) == 1200
    b = budget(tmp_path).model_dump()
    assert session_time_limit(_cfg(gpu_budget=b)) == pytest.approx(23707.59, abs=0.01)
    assert session_time_limit(_cfg(gpu_budget=b, max_session_s=1200)) == 1200


def test_vllm_context_error_is_transient() -> None:
    class BadRequestError(Exception): ...

    assert is_transient(BadRequestError("This model's maximum context length is 32768 tokens"))
    assert not is_transient(BadRequestError("invalid tool schema"))


def test_proxy_app_parses_requests() -> None:
    from fastapi.testclient import TestClient

    from atl.serving.proxy import build_app

    client = TestClient(build_app(UPSTREAMS, "s3cret"))
    assert client.get("/health").json() == {"models": ["gemma-4-31b", "qwen3-8b"]}
    body = {"model": "qwen3-8b", "messages": [{"role": "user", "content": "hi"}]}
    assert client.post("/v1/chat/completions", json=body).status_code == 401
    bad = {**body, "model": "nope"}
    r = client.post("/v1/chat/completions", json=bad, headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 404  # parsed the body and routed: not a 422 validation error
