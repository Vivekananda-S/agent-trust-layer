"""Normalised trace schema: the contract between trace collection and the judge.

Every agent run is stored in this framework-independent format, one trace per JSONL line.
Labels are deliberately NOT part of a trace; they live in separate label records keyed by
`trace_id`, so the judge's input type cannot carry labels by construction.

Any change to these models must bump SCHEMA_VERSION.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

NonEmptyStr = Annotated[str, Field(min_length=1)]


class _Base(BaseModel):
    """Strict, immutable base: unknown fields are errors, records cannot be mutated."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class UserStep(_Base):
    """A message from the (simulated) user."""

    type: Literal["user"] = "user"
    content: str


class ThoughtStep(_Base):
    """Agent reasoning not shown to the user."""

    type: Literal["thought"] = "thought"
    content: str


class AssistantStep(_Base):
    """A message from the agent to the user."""

    type: Literal["assistant"] = "assistant"
    content: str


class ToolCallStep(_Base):
    """The agent invoking a tool. `call_id` links it to its result when the runtime provides one."""

    type: Literal["tool_call"] = "tool_call"
    tool: NonEmptyStr
    args: dict[str, Any] = Field(default_factory=dict)
    call_id: NonEmptyStr | None = None


class ToolResultStep(_Base):
    """What a tool returned. `error` is set when the call failed (timeouts, API errors)."""

    type: Literal["tool_result"] = "tool_result"
    tool: NonEmptyStr
    content: str
    error: str | None = None
    call_id: NonEmptyStr | None = None


Step = Annotated[
    UserStep | ThoughtStep | AssistantStep | ToolCallStep | ToolResultStep,
    Field(discriminator="type"),
]


class EnvOutcome(_Base):
    """Ground truth from the environment, when it has one. None means the env cannot tell."""

    task_success: bool | None = None


class Usage(_Base):
    """Cost and latency of the agent run."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)


class TraceMeta(_Base):
    """Conditions the trace was generated under; used for stratification and drift slices."""

    domain: NonEmptyStr  # e.g. "retail"; "fintech" is the out-of-domain set
    user_variant: NonEmptyStr  # e.g. "cooperative", "vague", "adversarial"
    fault: NonEmptyStr | None = None  # injected fault, e.g. "timeout", "injection"; None if clean
    seed: int | None = None
    created_at: datetime | None = None


class Trace(_Base):
    """One agent run, normalised. See docs/plan.md section 4."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    trace_id: NonEmptyStr
    task_id: NonEmptyStr  # splits are by task, never by trace
    agent_model: NonEmptyStr
    prompt_variant: NonEmptyStr
    policy_doc_id: NonEmptyStr
    task: NonEmptyStr
    steps: list[Step] = Field(min_length=1)
    env_outcome: EnvOutcome | None = None
    usage: Usage | None = None
    meta: TraceMeta

    @model_validator(mode="after")
    def _check_tool_pairing(self) -> Trace:
        """Every tool result must answer an earlier, still-open tool call of the same tool.

        Results with a `call_id` match the call with that id; results without one match the
        earliest open call of the same tool (FIFO). Calls may stay unanswered (e.g. the run
        ended), but call ids must be unique and each call is answered at most once.
        """
        open_calls: list[ToolCallStep] = []
        seen_ids: set[str] = set()
        for i, step in enumerate(self.steps):
            if isinstance(step, ToolCallStep):
                if step.call_id is not None:
                    if step.call_id in seen_ids:
                        raise ValueError(f"step {i}: duplicate call_id {step.call_id!r}")
                    seen_ids.add(step.call_id)
                open_calls.append(step)
            elif isinstance(step, ToolResultStep):
                match = _find_open_call(open_calls, step)
                if match is None:
                    raise ValueError(
                        f"step {i}: tool_result for {step.tool!r} (call_id={step.call_id!r}) "
                        "has no earlier open tool_call"
                    )
                if match.tool != step.tool:
                    raise ValueError(
                        f"step {i}: tool_result tool {step.tool!r} does not match "
                        f"tool_call tool {match.tool!r} for call_id {step.call_id!r}"
                    )
                open_calls.remove(match)
        return self


def _find_open_call(open_calls: list[ToolCallStep], result: ToolResultStep) -> ToolCallStep | None:
    """Return the open call a result answers, or None."""
    if result.call_id is not None:
        return next((c for c in open_calls if c.call_id == result.call_id), None)
    return next((c for c in open_calls if c.tool == result.tool), None)


def load_traces(path: str | Path) -> list[Trace]:
    """Load and validate a JSONL file of traces. Fails loudly, naming the bad line."""
    path = Path(path)
    traces: list[Trace] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                traces.append(Trace.model_validate_json(line))
            except ValidationError as e:
                raise ValueError(f"{path}:{line_no}: invalid trace\n{e}") from e
    logger.info("Loaded %d traces from %s", len(traces), path)
    return traces


def save_traces(traces: Iterable[Trace], path: str | Path) -> None:
    """Write traces as JSONL, one trace per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for trace in traces:
            f.write(trace.model_dump_json() + "\n")
            n += 1
    logger.info("Saved %d traces to %s", n, path)
