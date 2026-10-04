"""Convert tau2-bench simulation runs into normalised traces.

Works on the JSON form of tau2's `SimulationRun` (as written by `atl.agent.run_tau2`), so this
module never imports tau2 and the judge stays framework-independent.

Raw file format: one JSON object per line, `{"run_info": RunInfo, "simulation": SimulationRun}`.

Mapping decisions (see docs/results.md):
- `task` is the user's first message, not tau2's hidden user-scenario instructions. The judge
  must only see what a production agent would see; the scenario is privileged information.
- `env_outcome.task_success` is `reward == 1.0` under tau2's ENV evaluation (final DB state).
- Runs that ended because of the user simulator or infrastructure are not agent behaviour and
  are excluded (`unusable_reason`).
- The user simulator's control tokens (###STOP### etc.) are stripped: real users never send
  them, and ###TRANSFER### / ###OUT-OF-SCOPE### correlate with the outcome (label leakage).
- Tool-call ids lose LiteLLM's `__thought__<signature>` suffix (Gemini 3 reasoning state,
  provider-internal and kilobytes long). Raw files keep the full id.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import BaseModel, ConfigDict, Field

from atl.traces.schema import (
    AssistantStep,
    EnvOutcome,
    Step,
    ThoughtStep,
    ToolCallStep,
    ToolResultStep,
    Trace,
    TraceMeta,
    Usage,
    UserStep,
    save_traces,
)

logger = logging.getLogger(__name__)

TAU2_COMMIT = "5bfa7e3"  # keep in sync with the pin in pyproject.toml
EXCLUDED_TERMINATIONS = {
    "user_error",
    "infrastructure_error",
    "unexpected_error",
}
USER_CONTROL_TOKENS = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")
THOUGHT_SIGNATURE_SEP = "__thought__"


class RunInfo(BaseModel):
    """Generation conditions the runner records next to each simulation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    domain: str = Field(min_length=1)
    agent_model: str = Field(min_length=1)
    user_model: str = Field(min_length=1)
    prompt_variant: str = Field(min_length=1)
    user_variant: str = Field(min_length=1)
    fault: str | None = None
    trial: int = Field(ge=0)
    seed: int | None = None


def unusable_reason(sim: dict[str, Any]) -> str | None:
    """Return why a simulation must not become a trace, or None if it is usable."""
    reason = sim.get("termination_reason")
    if reason in EXCLUDED_TERMINATIONS:
        return f"termination:{reason}"
    if not sim.get("messages"):
        return "no_messages"
    return None


def review_flags(sim: dict[str, Any]) -> list[str]:
    """Reasons a usable run needs a human look before its env label is trusted.

    `user_stop_with_content`: the run failed and the user simulator sent text and a control
    token in one turn (e.g. "Yes, please proceed. ###STOP###"), which can end the run before
    the agent acts; the failure may then measure the simulator, not the agent. Successful runs
    are not flagged: "Thanks! ###STOP###" after a solved task is a normal goodbye.
    """
    flags: list[str] = []
    if (sim.get("reward_info") or {}).get("reward") == 1.0:
        return flags
    for msg in sim.get("messages") or []:
        content = msg.get("content") or ""
        text = _strip_control_tokens(content)
        if msg.get("role") == "user" and text and text != content.strip():
            flags.append("user_stop_with_content")
            break
    return flags


def sim_to_trace(sim: dict[str, Any], info: RunInfo) -> Trace:
    """Convert one tau2 SimulationRun (JSON dict) into a Trace."""
    steps = _convert_messages(sim["messages"])
    first_user = next((s.content for s in steps if isinstance(s, UserStep) and s.content), None)
    reward_info = sim.get("reward_info")
    task_success = None if reward_info is None else reward_info.get("reward") == 1.0

    return Trace(
        trace_id=f"tau2-{sim['id']}",
        task_id=f"tau2/{info.domain}/{sim['task_id']}",
        agent_model=info.agent_model,
        prompt_variant=info.prompt_variant,
        policy_doc_id=f"tau2_{info.domain}_policy@{TAU2_COMMIT}",
        task=first_user or "(user sent no text)",
        steps=steps,
        env_outcome=EnvOutcome(task_success=task_success),
        usage=_agent_usage(sim),
        meta=TraceMeta(
            domain=info.domain,
            user_variant=info.user_variant,
            fault=info.fault,
            seed=info.seed,
            created_at=sim.get("start_time"),
        ),
    )


def _convert_messages(messages: list[dict[str, Any]]) -> list[Step]:
    """Map tau2 messages to steps, pairing tool results with their calls by id."""
    steps: list[Step] = []
    call_tool_by_id: dict[str, str] = {}
    open_without_id: list[str] = []  # tool names of calls that carry no id, in order

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue  # the policy is referenced by policy_doc_id, not copied into steps
        if role == "user":
            if msg.get("tool_calls"):
                raise ValueError("user tool calls (e.g. tau2 telecom) are not supported")
            text = _strip_control_tokens(msg.get("content") or "")
            if text:
                steps.append(UserStep(content=text))
        elif role == "assistant":
            reasoning = _reasoning(msg)
            if reasoning:
                steps.append(ThoughtStep(content=reasoning))
            if msg.get("content"):
                steps.append(AssistantStep(content=msg["content"]))
            for call in msg.get("tool_calls") or []:
                call_id = _clean_id(call.get("id"))
                if call_id:
                    call_tool_by_id[call_id] = call["name"]
                else:
                    open_without_id.append(call["name"])
                steps.append(
                    ToolCallStep(
                        tool=call["name"], args=call.get("arguments") or {}, call_id=call_id
                    )
                )
        elif role == "tool":
            if msg.get("requestor", "assistant") != "assistant":
                raise ValueError("tool results for the user simulator are not supported")
            call_id = _clean_id(msg.get("id"))
            if call_id and call_id in call_tool_by_id:
                tool = call_tool_by_id[call_id]
            elif open_without_id:
                tool, call_id = open_without_id.pop(0), None
            else:
                raise ValueError(f"tool message {call_id!r} has no matching tool call")
            content = msg.get("content") or ""
            error = (content or "tool error") if msg.get("error") else None
            steps.append(ToolResultStep(tool=tool, content=content, error=error, call_id=call_id))
        else:
            raise ValueError(f"unknown tau2 message role {role!r}")
    return steps


def _clean_id(raw_id: str | None) -> str | None:
    """Tool-call id without a provider thought-signature suffix; None if empty."""
    return (raw_id or "").split(THOUGHT_SIGNATURE_SEP, 1)[0] or None


def _strip_control_tokens(text: str) -> str:
    """Remove user-simulator control tokens; return the remaining text, stripped."""
    for token in USER_CONTROL_TOKENS:
        text = text.replace(token, "")
    return text.strip()


def _reasoning(msg: dict[str, Any]) -> str | None:
    """Reasoning text the provider returned alongside the message, if any (e.g. Gemini)."""
    raw = msg.get("raw_data") or {}
    choices = raw.get("choices") or [{}]
    return (choices[0].get("message") or {}).get("reasoning_content") or None


def _agent_usage(sim: dict[str, Any]) -> Usage | None:
    """Sum the agent's token usage over its messages; None if the provider reported none."""
    prompt = completion = 0
    seen = False
    for msg in sim["messages"]:
        usage = msg.get("usage") if msg.get("role") == "assistant" else None
        if usage:
            seen = True
            prompt += usage.get("prompt_tokens") or 0
            completion += usage.get("completion_tokens") or 0
    if not seen:
        return None
    latency_ms = round((sim.get("duration") or 0.0) * 1000)
    return Usage(input_tokens=prompt, output_tokens=completion, latency_ms=latency_ms)


def convert_raw_file(raw_path: str | Path) -> tuple[list[Trace], Counter[str]]:
    """Convert a raw runner JSONL file. Returns traces and counts of excluded runs by reason."""
    traces: list[Trace] = []
    excluded: Counter[str] = Counter()
    with Path(raw_path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sim, info = record["simulation"], RunInfo.model_validate(record["run_info"])
            reason = unusable_reason(sim)
            if reason:
                excluded[reason] += 1
                continue
            try:
                traces.append(sim_to_trace(sim, info))
            except ValueError as e:
                raise ValueError(f"{raw_path}:{line_no}: {e}") from e
    return traces, excluded


app = typer.Typer(help="Convert raw agent output into normalised traces.", no_args_is_help=True)


@app.callback()
def _main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


@app.command()
def convert(
    raw: Annotated[Path, typer.Option("--raw", help="Raw runner JSONL.")],
    out: Annotated[Path, typer.Option("--out", help="Output trace JSONL.")],
) -> None:
    """Convert a raw tau2 runner file into a trace JSONL file."""
    traces, excluded = convert_raw_file(raw)
    save_traces(traces, out)
    logger.info("Converted %d traces; excluded %s", len(traces), dict(excluded))


@app.command("import-zip")
def import_zip_cmd(
    zip_path: Annotated[Path, typer.Option("--zip", help="A session's traces.zip.")],
    traces_dir: Annotated[Path, typer.Option(help="Local traces directory.")] = Path("data/traces"),
    imported_dir: Annotated[Path, typer.Option(help="Where the zip's logs/costs are kept.")] = Path(
        "data/imported"
    ),
) -> None:
    """Merge a session zip from Kaggle/Colab into the local dataset (dedup by task, trial)."""
    from atl.traces.merge import import_zip

    for run, (added, dup) in import_zip(zip_path, traces_dir, imported_dir).items():
        logger.info("%s: added %d runs, %d already present", run, added, dup)
