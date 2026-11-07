"""Human-readable views of a trace and its policy, for the labelling app.

This is for people, not the judge: the judge's input format is judge/serialise.py, which is
versioned separately. Nothing here hides or adds information the trace does not contain.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from atl.traces.schema import Step, ToolCallStep, ToolResultStep, Trace

DEFAULT_TAU2_DATA = Path("external/tau2-bench/data")  # pinned checkout, `atl-agent fetch-data`


@dataclass(frozen=True)
class StepView:
    """One step as the app shows it."""

    index: int
    kind: str  # user | thought | assistant | tool_call | tool_result
    title: str
    body: str
    is_error: bool = False


def step_view(index: int, step: Step) -> StepView:
    """Render one step: tool calls as name + pretty JSON args, tool errors flagged."""
    if isinstance(step, ToolCallStep):
        args = json.dumps(step.args, indent=2, ensure_ascii=False)
        return StepView(index, step.type, f"call {step.tool}", args)
    if isinstance(step, ToolResultStep):
        body = step.content if not step.error else f"ERROR: {step.error}\n{step.content}"
        return StepView(index, step.type, f"result {step.tool}", body, is_error=bool(step.error))
    return StepView(index, step.type, step.type, step.content)


def trace_views(trace: Trace) -> list[StepView]:
    """Every step of a trace, numbered from 0 (the numbering labels use)."""
    return [step_view(i, s) for i, s in enumerate(trace.steps)]


def policy_text(domain: str, data_dir: str | Path | None = None) -> str:
    """The tau2 domain policy the agent is judged against (even if its prompt omitted it)."""
    root = Path(data_dir or os.environ.get("TAU2_DATA_DIR", DEFAULT_TAU2_DATA))
    path = root / "tau2" / "domains" / domain / "policy.md"
    if not path.is_file():
        raise FileNotFoundError(f"policy not found at {path}; run `atl-agent fetch-data`")
    return path.read_text(encoding="utf-8")
