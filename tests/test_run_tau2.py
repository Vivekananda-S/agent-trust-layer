"""Tests for the tau2 runner: config, task selection, pricing, and an offline smoke run."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from atl.agent.run_tau2 import (
    DEFAULT_TAU2_CHECKOUT,
    AgentRunConfig,
    Price,
    _price,
    _select_tasks,
    run,
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
