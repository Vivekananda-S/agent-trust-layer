"""Tests for tool-fault injection: injector logic, and the hook on a real tau2 environment."""

import importlib.util

import pytest

from atl.agent.faults import (
    FAULT_MESSAGES,
    INJECTIONS,
    FaultConfig,
    FaultInjector,
    active_injector,
    install_fault_hook,
)
from atl.agent.run_tau2 import DEFAULT_TAU2_CHECKOUT


def cfg(**kw: object) -> FaultConfig:
    base = {"run_rate": 1.0, "call_rate": 1.0, "max_per_run": 2, "types": ["timeout"]}
    return FaultConfig.model_validate({**base, **kw})


def test_decisions_are_seeded() -> None:
    c = cfg(call_rate=0.5, max_per_run=100, types=["timeout", "error", "empty", "injection"])
    a, b = FaultInjector(c, "s"), FaultInjector(c, "s")
    seq_a = [a.decide() for _ in range(50)]
    assert seq_a == [b.decide() for _ in range(50)]
    assert set(seq_a) - {None}  # some faults happen
    assert None in seq_a  # and some calls run normally


def test_max_per_run_caps_recorded_faults() -> None:
    inj = FaultInjector(cfg(max_per_run=2), "s")
    for i in range(5):
        if (fault := inj.decide()) is not None:
            inj.record(fault, "get_x", f"c{i}")
    assert len(inj.events) == 2
    assert inj.decide() is None


def test_summary() -> None:
    inj = FaultInjector(cfg(), "s")
    assert inj.summary() is None
    inj.record("timeout", "a", "1")
    inj.record("empty", "b", "2")
    inj.record("timeout", "c", "3")
    assert inj.summary() == "empty+timeout"


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        cfg(types=[])
    with pytest.raises(ValueError):
        cfg(types=["meteor"])
    with pytest.raises(ValueError):
        cfg(call_rate=0.0)


needs_tau2 = pytest.mark.skipif(
    importlib.util.find_spec("tau2") is None
    or not (DEFAULT_TAU2_CHECKOUT / "data" / "tau2" / "domains").is_dir(),
    reason="needs the agent extra and `atl-agent fetch-data`",
)


@pytest.fixture
def mock_env():  # type: ignore[no-untyped-def]
    from atl.agent.run_tau2 import _point_tau2_at_data

    _point_tau2_at_data()
    from tau2.data_model.message import ToolCall, ToolMessage
    from tau2.domains.mock.environment import get_environment
    from tau2.environment.environment import Environment

    install_fault_hook(Environment, ToolMessage)
    install_fault_hook(Environment, ToolMessage)  # idempotent: must not double-wrap
    return get_environment(), ToolCall


def _with(injector: FaultInjector | None, fn):  # type: ignore[no-untyped-def]
    token = active_injector.set(injector)
    try:
        return fn()
    finally:
        active_injector.reset(token)


@needs_tau2
def test_no_injector_passes_through(mock_env) -> None:  # type: ignore[no-untyped-def]
    env, ToolCall = mock_env
    msg = env.get_response(ToolCall(id="c1", name="get_users", arguments={}))
    assert msg.error is False and "user_1" in msg.content


@needs_tau2
@pytest.mark.parametrize("fault", ["timeout", "error"])
def test_read_call_fails(mock_env, fault: str) -> None:  # type: ignore[no-untyped-def]
    env, ToolCall = mock_env
    inj = FaultInjector(cfg(types=[fault]), "s")
    msg = _with(inj, lambda: env.get_response(ToolCall(id="c1", name="get_users", arguments={})))
    assert msg.error is True and msg.content == FAULT_MESSAGES[fault] and msg.id == "c1"
    assert inj.events == [{"type": fault, "tool": "get_users", "call_id": "c1"}]


@needs_tau2
def test_empty_and_injection(mock_env) -> None:  # type: ignore[no-untyped-def]
    env, ToolCall = mock_env
    call = ToolCall(id="c1", name="get_users", arguments={})
    empty = _with(FaultInjector(cfg(types=["empty"]), "s"), lambda: env.get_response(call))
    assert empty.error is False and empty.content == ""
    injected = _with(FaultInjector(cfg(types=["injection"]), "s"), lambda: env.get_response(call))
    assert injected.error is False and "user_1" in injected.content  # real result kept
    assert any(text in injected.content for text in INJECTIONS)


@needs_tau2
def test_write_calls_never_faulted(mock_env) -> None:  # type: ignore[no-untyped-def]
    env, ToolCall = mock_env
    inj = FaultInjector(cfg(types=["timeout"]), "s")
    call = ToolCall(id="w1", name="create_task", arguments={"user_id": "user_1", "title": "T"})
    msg = _with(inj, lambda: env.get_response(call))
    assert msg.error is False and "task_" in msg.content  # the write really ran
    assert inj.events == []
