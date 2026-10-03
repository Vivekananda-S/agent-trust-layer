"""Tests for the tau2 SimulationRun -> Trace adapter (pure JSON, no tau2 import)."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from atl.traces.adapters.tau2 import (
    RunInfo,
    convert_raw_file,
    review_flags,
    sim_to_trace,
    unusable_reason,
)
from atl.traces.schema import (
    AssistantStep,
    ThoughtStep,
    ToolCallStep,
    ToolResultStep,
    UserStep,
)

FIXTURE = Path(__file__).parent / "fixtures" / "tau2_sim_small.json"
INFO = RunInfo(
    domain="retail",
    agent_model="gemini/gemini-3.8-flash",
    user_model="gemini/gemini-3.5-flash-lite",
    prompt_variant="tau2_default",
    user_variant="tau2_default",
    trial=0,
    seed=300,
)


@pytest.fixture
def sim() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


def test_ids_and_metadata(sim: dict[str, Any]) -> None:
    trace = sim_to_trace(sim, INFO)
    assert trace.trace_id == "tau2-sim-0001"
    assert trace.task_id == "tau2/retail/42"
    assert trace.policy_doc_id == "tau2_retail_policy@5bfa7e3"
    assert trace.meta.domain == "retail"
    assert trace.meta.seed == 300


def test_task_is_first_user_message_not_hidden_scenario(sim: dict[str, Any]) -> None:
    assert sim_to_trace(sim, INFO).task == "I want to return order #W123."


def test_step_mapping(sim: dict[str, Any]) -> None:
    steps = sim_to_trace(sim, INFO).steps
    assert [type(s) for s in steps] == [
        AssistantStep,  # greeting
        UserStep,
        ThoughtStep,  # reasoning_content from raw_data
        ToolCallStep,
        ToolCallStep,
        ToolResultStep,  # c2 answered first: matched by id
        ToolResultStep,
        AssistantStep,
        UserStep,  # "Thanks!" with the control token removed
    ]
    result_c2, result_c1 = steps[5], steps[6]
    assert isinstance(result_c2, ToolResultStep) and result_c2.tool == "get_user_details"
    assert isinstance(result_c1, ToolResultStep) and result_c1.tool == "get_order_details"
    assert result_c1.error == "Error: order not found"
    assert result_c2.error is None


def test_control_tokens_stripped(sim: dict[str, Any]) -> None:
    sim["messages"].append({"role": "user", "content": "###TRANSFER###"})
    steps = sim_to_trace(sim, INFO).steps
    user_texts = [s.content for s in steps if isinstance(s, UserStep)]
    assert user_texts == ["I want to return order #W123.", "Thanks!"]
    assert not any("###" in s.model_dump_json() for s in steps)


def test_usage_sums_agent_messages(sim: dict[str, Any]) -> None:
    usage = sim_to_trace(sim, INFO).usage
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens, usage.latency_ms) == (2200, 55, 8500)


def test_usage_none_when_not_reported(sim: dict[str, Any]) -> None:
    for m in sim["messages"]:
        m.pop("usage", None)
    assert sim_to_trace(sim, INFO).usage is None


@pytest.mark.parametrize(
    ("reward_info", "expected"),
    [({"reward": 1.0}, True), ({"reward": 0.0}, False), (None, None)],
)
def test_reward_to_task_success(
    sim: dict[str, Any], reward_info: dict[str, float] | None, expected: bool | None
) -> None:
    sim["reward_info"] = reward_info
    outcome = sim_to_trace(sim, INFO).env_outcome
    assert outcome is not None and outcome.task_success is expected


def test_tool_message_without_id_matched_fifo(sim: dict[str, Any]) -> None:
    for m in sim["messages"]:
        for c in m.get("tool_calls") or []:
            c["id"] = ""
        if m["role"] == "tool":
            m["id"] = ""
    steps = sim_to_trace(sim, INFO).steps
    results = [s for s in steps if isinstance(s, ToolResultStep)]
    assert [r.tool for r in results] == ["get_order_details", "get_user_details"]
    assert all(r.call_id is None for r in results)


def test_thought_signature_stripped_from_ids(sim: dict[str, Any]) -> None:
    for m in sim["messages"]:
        for c in m.get("tool_calls") or []:
            c["id"] += "__thought__EosFCogFAWkU+sig=="
        if m["role"] == "tool":
            m["id"] += "__thought__EosFCogFAWkU+sig=="
    steps = sim_to_trace(sim, INFO).steps
    ids = [s.call_id for s in steps if isinstance(s, ToolCallStep | ToolResultStep)]
    assert ids == ["c1", "c2", "c2", "c1"]


def test_user_tool_calls_rejected(sim: dict[str, Any]) -> None:
    sim["messages"][2]["tool_calls"] = [{"id": "u1", "name": "toggle_wifi", "arguments": {}}]
    with pytest.raises(ValueError, match="user tool calls"):
        sim_to_trace(sim, INFO)


@pytest.mark.parametrize(
    ("termination", "reason"),
    [
        ("user_error", "termination:user_error"),
        ("infrastructure_error", "termination:infrastructure_error"),
        ("unexpected_error", "termination:unexpected_error"),
        ("agent_error", None),  # protocol violation by the agent is real agent behaviour
        ("max_steps", None),
        ("user_stop", None),
    ],
)
def test_unusable_reason(sim: dict[str, Any], termination: str, reason: str | None) -> None:
    sim["termination_reason"] = termination
    assert unusable_reason(sim) == reason


def test_convert_raw_file_counts_exclusions(sim: dict[str, Any], tmp_path: Path) -> None:
    bad = copy.deepcopy(sim)
    bad["id"], bad["termination_reason"] = "sim-0002", "infrastructure_error"
    path = tmp_path / "raw.jsonl"
    with path.open("w") as f:
        for s in (sim, bad):
            f.write(json.dumps({"run_info": INFO.model_dump(), "simulation": s}) + "\n")
    traces, excluded = convert_raw_file(path)
    assert [t.trace_id for t in traces] == ["tau2-sim-0001"]
    assert excluded == {"termination:infrastructure_error": 1}


def test_review_flag_on_failed_run_with_stop_and_text(sim: dict[str, Any]) -> None:
    # Fixture: reward 0.0 and last user message "Thanks! ###STOP###".
    assert review_flags(sim) == ["user_stop_with_content"]


def test_no_review_flag_when_run_succeeded(sim: dict[str, Any]) -> None:
    sim["reward_info"] = {"reward": 1.0}
    assert review_flags(sim) == []


def test_no_review_flag_for_bare_stop(sim: dict[str, Any]) -> None:
    sim["messages"][-1]["content"] = "###STOP###"
    assert review_flags(sim) == []
