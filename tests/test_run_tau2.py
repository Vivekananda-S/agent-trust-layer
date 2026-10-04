"""Tests for the tau2 runner: config, task selection, pricing, and an offline smoke run."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from atl.agent.run_tau2 import (
    DEFAULT_TAU2_CHECKOUT,
    AgentRunConfig,
    ContextOverflow,
    Price,
    _guard_context,
    _price,
    _select_tasks,
    choose_conditions,
    pending_runs,
    run,
    run_stats,
)

CONFIGS = sorted((Path(__file__).parents[1] / "configs" / "agent").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_repo_configs_load(path: Path) -> None:
    cfg = AgentRunConfig.from_yaml(path)
    assert cfg.agent_model in cfg.prices and cfg.user_model in cfg.prices


def _cfg(**kw: object) -> AgentRunConfig:
    base = {"run_name": "r", "domain": "retail", "agent_model": "a", "user_model": "u"}
    return AgentRunConfig.model_validate({**base, "budget_usd": 1.0, **kw})


TASKS = [SimpleNamespace(id=str(i)) for i in range(20)]


def test_task_sample_is_seeded_and_not_first_n() -> None:
    a = _select_tasks(TASKS, _cfg(num_tasks=5, task_sample_seed=0))
    b = _select_tasks(TASKS, _cfg(num_tasks=5, task_sample_seed=0))
    assert [t.id for t in a] == [t.id for t in b]
    assert [t.id for t in a] != [str(i) for i in range(5)]


def test_task_ids_override_and_unknown_rejected() -> None:
    assert [t.id for t in _select_tasks(TASKS, _cfg(task_ids=["3", "1"]))] == ["3", "1"]
    with pytest.raises(ValueError, match="unknown task ids"):
        _select_tasks(TASKS, _cfg(task_ids=["99"]))


def test_price_from_config_table() -> None:
    response = SimpleNamespace(
        model="gemini-3.8-flash", usage=SimpleNamespace(prompt_tokens=2_000, completion_tokens=100)
    )
    prices = {"gemini/gemini-3.8-flash": Price(input=0.75, output=3.75)}
    # 2000 * 0.75e-6 + 100 * 3.75e-6 = 0.0015 + 0.000375
    assert _price(response, prices, fallback=None) == pytest.approx(0.001875)


needs_tau2 = pytest.mark.skipif(
    importlib.util.find_spec("tau2") is None
    or not (DEFAULT_TAU2_CHECKOUT / "data" / "tau2" / "domains").is_dir(),
    reason="needs the agent extra and `atl-agent fetch-data`",
)


@needs_tau2
def test_smoke_run_end_to_end(tmp_path: Path) -> None:
    cfg = AgentRunConfig.from_yaml(CONFIGS[0]).as_smoke(tmp_path)
    traces_path = run(cfg, smoke=True)
    traces = [json.loads(x) for x in traces_path.read_text().splitlines()]
    assert len(traces) == 1
    assert traces[0]["env_outcome"]["task_success"] is True
    assert [s["type"] for s in traces[0]["steps"]] == [
        "assistant",
        "user",
        "tool_call",
        "tool_result",
        "assistant",
    ]
    # Re-collect from scratch: every LLM call (2 user + 2 agent) must come from the cache.
    (traces_path.parent / "raw.jsonl").unlink()
    run(cfg, smoke=True)
    log = [json.loads(x) for x in cfg.cost_log.read_text().splitlines()]
    assert [e["cached"] for e in log].count(False) == 4
    assert [e["cached"] for e in log].count(True) == 4


def test_ollama_model_without_num_ctx_rejected() -> None:
    with pytest.raises(ValueError, match="num_ctx"):
        _cfg(agent_model="ollama_chat/qwen3:8b", agent_llm_args={"api_base": "http://x"})
    assert _cfg(agent_model="ollama_chat/qwen3:8b", agent_llm_args={"num_ctx": 32768})


def test_context_guard() -> None:
    calls: list[dict[str, object]] = []
    guarded = _guard_context(lambda **kw: calls.append(kw) or "ok")
    small = [{"role": "user", "content": "x" * 3_000}]
    huge = [{"role": "user", "content": "x" * 110_000}]  # ~31.4k tokens at 3.5 chars/token
    assert guarded(model="ollama_chat/m", messages=small, num_ctx=32768) == "ok"
    with pytest.raises(ContextOverflow, match="num_ctx=32768"):
        guarded(model="ollama_chat/m", messages=huge, num_ctx=32768)
    assert guarded(model="gemini/m", messages=huge) == "ok"  # no num_ctx: hosted model, no guard
    assert len(calls) == 2


def test_run_stats(tmp_path: Path) -> None:
    sims = [
        {
            "reward_info": {"reward": 1.0},
            "duration": 120.0,
            "agent_cost": 0.05,
            "user_cost": 0.01,
            "termination_reason": "user_stop",
            "start_time": "2026-10-03T10:00:00",
            "end_time": "2026-10-03T10:02:00",
        },
        {
            "reward_info": {"reward": 0.0},
            "duration": 240.0,
            "agent_cost": 0.0,
            "user_cost": 0.02,
            "termination_reason": "max_steps",
            "start_time": "2026-10-03T10:00:00",
            "end_time": "2026-10-03T10:04:00",
        },
    ]
    (tmp_path / "raw.jsonl").write_text("".join(json.dumps({"simulation": s}) + "\n" for s in sims))
    (tmp_path / "review_flags.jsonl").write_text('{"trace_id": "x", "flags": ["f"]}\n')
    assert run_stats(tmp_path) == {
        "runs": 2,
        "env_success_rate": 0.5,
        "terminations": {"user_stop": 1, "max_steps": 1},
        "mean_minutes_per_run": 3.0,  # (120 + 240) / 2 s
        "runs_per_hour_wall_clock": 30.0,  # 2 overlapping runs, 10:00 -> 10:04
        "mean_cost_usd": 0.04,  # (0.06 + 0.02) / 2
        "customer_loop_runs": 0,
        "flagged_for_review": 1,
    }
    assert run_stats(tmp_path / "missing") == {"runs": 0}


@needs_tau2
def test_smoke_parallel_runs_with_mixed_variants(tmp_path: Path) -> None:
    cfg = AgentRunConfig.from_yaml(CONFIGS[0]).as_smoke(tmp_path)
    cfg = cfg.model_copy(
        update={
            "num_trials": 4,
            "max_concurrency": 3,
            "prompt_variants": {"tau2_default": 1.0, "sloppy": 1.0, "no_policy": 1.0},
            "user_variants": {"tau2_default": 1.0, "pushy": 1.0},
        }
    )
    run(cfg, smoke=True)
    records = [
        json.loads(x) for x in (tmp_path / cfg.run_name / "raw.jsonl").read_text().splitlines()
    ]
    assert sorted(r["run_info"]["trial"] for r in records) == [0, 1, 2, 3]
    for r in records:
        expected = choose_conditions(cfg, "create_task_1", r["run_info"]["trial"])
        assert (r["run_info"]["prompt_variant"], r["run_info"]["user_variant"]) == expected[:2]
        assert r["faults"] == [] and r["run_info"]["fault"] is None
    assert len({r["run_info"]["prompt_variant"] for r in records}) > 1  # the mix is really used


@needs_tau2
def test_failed_run_is_logged_not_silently_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import atl.agent.run_tau2 as runner

    real_fake = runner._fake_completion

    def empty_agent(**kwargs: object) -> object:  # tau2 raises on an empty agent reply
        response = real_fake(**kwargs)
        if kwargs["model"] == "smoke/agent":
            response.choices[0].message.content = None
            response.choices[0].message.tool_calls = None
        return response

    monkeypatch.setattr(runner, "_fake_completion", empty_agent)
    cfg = AgentRunConfig.from_yaml(CONFIGS[0]).as_smoke(tmp_path)
    run(cfg, smoke=True)
    run_dir = tmp_path / cfg.run_name
    assert not (run_dir / "raw.jsonl").exists()
    losses = [json.loads(x) for x in (run_dir / "failed_runs.jsonl").read_text().splitlines()]
    assert [(x["task_id"], x["trial"]) for x in losses] == [("create_task_1", 0)]
    assert "must have either content or tool_calls" in losses[0]["error"]


@needs_tau2
def test_context_overflow_ends_only_that_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import atl.agent.run_tau2 as runner
    from atl.agent.llm_cache import cache_namespace

    real_fake = runner._fake_completion

    def overflow_on_trial_0(**kwargs: object) -> object:
        if cache_namespace.get().endswith("/0"):
            raise ContextOverflow("too long")
        return real_fake(**kwargs)

    monkeypatch.setattr(runner, "_fake_completion", overflow_on_trial_0)
    cfg = (
        AgentRunConfig.from_yaml(CONFIGS[0]).as_smoke(tmp_path).model_copy(update={"num_trials": 2})
    )
    run(cfg, smoke=True)
    run_dir = tmp_path / cfg.run_name
    kept = [
        json.loads(x)["run_info"]["trial"] for x in (run_dir / "raw.jsonl").read_text().splitlines()
    ]
    lost = [json.loads(x) for x in (run_dir / "failed_runs.jsonl").read_text().splitlines()]
    assert kept == [1]
    assert [(x["trial"], x["error"]) for x in lost] == [(0, "ContextOverflow: too long")]


def test_pending_runs_are_trial_major_and_skip_done() -> None:
    tasks = [SimpleNamespace(id=i) for i in ("a", "b", "c")]
    got = [(t.id, tr) for t, tr in pending_runs(tasks, 2, done={("b", 0)})]
    assert got == [("a", 0), ("c", 0), ("a", 1), ("b", 1), ("c", 1)]


def test_customer_loop_detection() -> None:
    from atl.agent.run_tau2 import _customer_loops

    user = lambda text: {"role": "user", "content": text}  # noqa: E731
    looping = {"messages": [user("Mia Garcia.")] * 5 + [{"role": "assistant", "content": "?"}]}
    normal = {"messages": [user("hi"), user("thanks"), user("hi")]}
    assert _customer_loops(looping) is True
    assert _customer_loops(normal) is False
    assert _customer_loops({"messages": []}) is False


def test_is_transient() -> None:
    from atl.agent.run_tau2 import is_transient

    class RateLimitError(Exception): ...

    class InternalServerError(Exception): ...

    class AuthenticationError(Exception): ...

    assert is_transient(RateLimitError("429"))
    assert is_transient(InternalServerError("500"))
    assert is_transient(ContextOverflow("long"))

    class ContextWindowExceededError(Exception): ...

    assert is_transient(ContextWindowExceededError("maximum context length is 32768 tokens"))
    assert is_transient(ValueError("AssistantMessage must have either content or tool_calls."))
    assert not is_transient(AuthenticationError("bad key"))
    assert not is_transient(ValueError("something else"))


@needs_tau2
def test_transient_failures_do_not_stop_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import atl.agent.run_tau2 as runner
    from atl.agent.llm_cache import cache_namespace

    class RateLimitError(Exception): ...

    real_fake = runner._fake_completion
    monkeypatch.setattr(runner, "TRANSIENT_PAUSE_S", 0.0)

    def flaky(**kwargs: object) -> object:  # trials 0-3 hit the provider's rate limit
        if int(cache_namespace.get().rsplit("/", 1)[1]) < 4:
            raise RateLimitError("429")
        return real_fake(**kwargs)

    monkeypatch.setattr(runner, "_fake_completion", flaky)
    cfg = (
        AgentRunConfig.from_yaml(CONFIGS[0]).as_smoke(tmp_path).model_copy(update={"num_trials": 6})
    )
    run(cfg, smoke=True)
    run_dir = tmp_path / cfg.run_name
    kept = sorted(
        json.loads(x)["run_info"]["trial"] for x in (run_dir / "raw.jsonl").read_text().splitlines()
    )
    lost = [json.loads(x) for x in (run_dir / "failed_runs.jsonl").read_text().splitlines()]
    assert kept == [4, 5]  # 4 consecutive transient failures did not stop the session
    assert len(lost) == 4 and all(x["transient"] for x in lost)
