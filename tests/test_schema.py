"""Tests for the normalised trace schema."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from atl.traces.schema import (
    SCHEMA_VERSION,
    AssistantStep,
    ThoughtStep,
    ToolCallStep,
    ToolResultStep,
    Trace,
    UserStep,
    load_traces,
    save_traces,
)

FIXTURE = Path(__file__).parent / "fixtures" / "trace_minimal.json"


@pytest.fixture
def raw() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


def _with_steps(raw: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    return {**raw, "steps": steps}


def _call(tool: str, call_id: str | None = None) -> dict[str, Any]:
    return {"type": "tool_call", "tool": tool, "args": {}, "call_id": call_id}


def _result(tool: str, call_id: str | None = None) -> dict[str, Any]:
    return {"type": "tool_result", "tool": tool, "content": "ok", "call_id": call_id}


# --- parsing and round trips -------------------------------------------------------------


def test_fixture_parses(raw: dict[str, Any]) -> None:
    trace = Trace.model_validate(raw)
    assert trace.schema_version == SCHEMA_VERSION
    assert trace.task_id == "retail_task_042"
    assert trace.env_outcome is not None and trace.env_outcome.task_success is True
    assert trace.meta.domain == "retail"


def test_json_round_trip_is_identical(raw: dict[str, Any]) -> None:
    trace = Trace.model_validate(raw)
    assert Trace.model_validate_json(trace.model_dump_json()) == trace


def test_discriminator_picks_step_classes(raw: dict[str, Any]) -> None:
    steps = Trace.model_validate(raw).steps
    assert [type(s) for s in steps] == [
        UserStep,
        ThoughtStep,
        ToolCallStep,
        ToolResultStep,
        AssistantStep,
    ]


def test_optional_outcome_and_usage(raw: dict[str, Any]) -> None:
    raw["env_outcome"] = {"task_success": None}
    del raw["usage"]
    trace = Trace.model_validate(raw)
    assert trace.env_outcome is not None and trace.env_outcome.task_success is None
    assert trace.usage is None


def test_tool_error_is_kept(raw: dict[str, Any]) -> None:
    raw["steps"][3]["error"] = "timeout after 30s"
    trace = Trace.model_validate(raw)
    assert isinstance(trace.steps[3], ToolResultStep)
    assert trace.steps[3].error == "timeout after 30s"


# --- rejections ----------------------------------------------------------------------------


def test_unknown_step_type_rejected(raw: dict[str, Any]) -> None:
    raw["steps"].append({"type": "system", "content": "x"})
    with pytest.raises(ValidationError):
        Trace.model_validate(raw)


def test_extra_top_level_field_rejected(raw: dict[str, Any]) -> None:
    raw["label"] = "fail"
    with pytest.raises(ValidationError):
        Trace.model_validate(raw)


def test_extra_step_field_rejected(raw: dict[str, Any]) -> None:
    raw["steps"][0]["timestamp"] = 1
    with pytest.raises(ValidationError):
        Trace.model_validate(raw)


def test_empty_steps_rejected(raw: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Trace.model_validate(_with_steps(raw, []))


@pytest.mark.parametrize("field", ["trace_id", "task_id", "meta"])
def test_missing_required_field_rejected(raw: dict[str, Any], field: str) -> None:
    del raw[field]
    with pytest.raises(ValidationError):
        Trace.model_validate(raw)


def test_empty_task_id_rejected(raw: dict[str, Any]) -> None:
    raw["task_id"] = ""
    with pytest.raises(ValidationError):
        Trace.model_validate(raw)


def test_negative_usage_rejected(raw: dict[str, Any]) -> None:
    raw["usage"]["latency_ms"] = -1
    with pytest.raises(ValidationError):
        Trace.model_validate(raw)


def test_wrong_schema_version_rejected(raw: dict[str, Any]) -> None:
    raw["schema_version"] = "0.9"
    with pytest.raises(ValidationError):
        Trace.model_validate(raw)


def test_trace_is_frozen(raw: dict[str, Any]) -> None:
    trace = Trace.model_validate(raw)
    with pytest.raises(ValidationError):
        trace.task_id = "other"  # type: ignore[misc]


# --- tool call / result pairing -----------------------------------------------------------


def test_orphan_result_rejected(raw: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="no earlier open tool_call"):
        Trace.model_validate(_with_steps(raw, [_result("get_order")]))


def test_result_before_call_rejected(raw: dict[str, Any]) -> None:
    steps = [_result("get_order", "c1"), _call("get_order", "c1")]
    with pytest.raises(ValidationError, match="no earlier open tool_call"):
        Trace.model_validate(_with_steps(raw, steps))


def test_second_result_for_same_call_rejected(raw: dict[str, Any]) -> None:
    steps = [_call("get_order", "c1"), _result("get_order", "c1"), _result("get_order", "c1")]
    with pytest.raises(ValidationError, match="no earlier open tool_call"):
        Trace.model_validate(_with_steps(raw, steps))


def test_duplicate_call_id_rejected(raw: dict[str, Any]) -> None:
    steps = [_call("get_order", "c1"), _call("get_order", "c1")]
    with pytest.raises(ValidationError, match="duplicate call_id"):
        Trace.model_validate(_with_steps(raw, steps))


def test_tool_name_mismatch_rejected(raw: dict[str, Any]) -> None:
    steps = [_call("get_order", "c1"), _result("refund", "c1")]
    with pytest.raises(ValidationError, match="does not match"):
        Trace.model_validate(_with_steps(raw, steps))


def test_parallel_calls_matched_by_call_id(raw: dict[str, Any]) -> None:
    steps = [
        _call("get_order", "c1"),
        _call("get_order", "c2"),
        _result("get_order", "c2"),
        _result("get_order", "c1"),
    ]
    assert len(Trace.model_validate(_with_steps(raw, steps)).steps) == 4


def test_calls_without_ids_matched_fifo_by_tool(raw: dict[str, Any]) -> None:
    steps = [
        _call("get_order"),
        _call("get_user"),
        _result("get_user"),
        _result("get_order"),
    ]
    assert len(Trace.model_validate(_with_steps(raw, steps)).steps) == 4


def test_unanswered_call_allowed(raw: dict[str, Any]) -> None:
    steps = [{"type": "user", "content": "hi"}, _call("get_order", "c1")]
    assert len(Trace.model_validate(_with_steps(raw, steps)).steps) == 2


# --- JSONL io --------------------------------------------------------------------------------


def test_jsonl_round_trip(raw: dict[str, Any], tmp_path: Path) -> None:
    traces = [Trace.model_validate({**raw, "trace_id": f"t_{i}"}) for i in range(3)]
    path = tmp_path / "sub" / "traces.jsonl"
    save_traces(traces, path)
    assert load_traces(path) == traces


def test_load_reports_bad_line_number(raw: dict[str, Any], tmp_path: Path) -> None:
    good = Trace.model_validate(raw).model_dump_json()
    path = tmp_path / "traces.jsonl"
    path.write_text(good + "\n" + '{"trace_id": "broken"}\n')
    with pytest.raises(ValueError, match=r"traces\.jsonl:2"):
        load_traces(path)
