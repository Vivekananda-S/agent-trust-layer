"""Run a tau2-bench agent on sampled tasks and record raw runs plus normalised traces.

Usage:
    atl-agent run --config configs/agent/pilot_retail.yaml
    atl-agent run --config configs/agent/pilot_retail.yaml --smoke   # offline, fake LLM, $0

Every LLM call (agent and user simulator) goes through `CachedCompletion`: disk cache, cost
log, budget cap and rate limit. Runs resume: (task, trial) pairs already in the raw file are
skipped, so a killed session loses at most the runs in progress.

Diversity: each (task, trial) draws its prompt variant, user variant and whether tool faults
are injected from the config's weights, seeded by (seed, task, trial), so a run's conditions
are reproducible and recorded in its RunInfo. `max_concurrency` runs several simulations at
once (useful with a local Ollama server that serves parallel requests).

Outputs, under `<output_dir>/<run_name>/`:
    raw.jsonl     one {"run_info", "simulation", "faults"} record per run (full tau2 output)
    failed_runs.jsonl  runs that raised (e.g. empty agent reply), so losses stay measurable
    traces.jsonl  normalised traces, regenerated from raw.jsonl at the end of every invocation
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from atl.agent.faults import FaultConfig, FaultInjector, active_injector, install_fault_hook
from atl.agent.llm_cache import BudgetExceeded, CachedCompletion, cache_namespace
from atl.agent.rate_limit import ModelLimit, RateLimiter
from atl.agent.variants import (
    AGENT_NAMES,
    PromptVariant,
    UserVariant,
    register_prompt_variants,
    with_user_variant,
)
from atl.traces.adapters.tau2 import RunInfo, convert_raw_file, review_flags
from atl.traces.schema import save_traces

logger = logging.getLogger(__name__)

TAU2_REPO = "https://github.com/sierra-research/tau2-bench.git"
TAU2_COMMIT_FULL = "5bfa7e37b36656b37dc6d022156be6563c1007f3"  # same pin as pyproject.toml
DEFAULT_TAU2_CHECKOUT = Path("external/tau2-bench")
SMOKE_DOMAIN = "mock"
SMOKE_TASK_IDS = ["create_task_1"]
MAX_CONSECUTIVE_FAILURES = 3  # e.g. a broken key or daily quota exhausted: stop, resume tomorrow
# Errors that end one run but say nothing about the setup: provider hiccups (free-tier 429/5xx),
# a too-long conversation, an empty agent reply. They are logged and retried on the next resume,
# but never stop the session (free-tier Gemma produced dozens per hour and stopped runs early).
TRANSIENT_ERROR_NAMES = {
    "RateLimitError",
    "InternalServerError",
    "ServiceUnavailableError",
    "APIConnectionError",
    "Timeout",
    "ContextOverflow",
}
TRANSIENT_PAUSE_S = 30.0  # after a provider error, this worker waits before its next run
# Local (Ollama) context guard. Ollama silently truncates prompts longer than `num_ctx`, which
# would cut the policy out of the prompt and turn every run into a fake failure.
# Measured on 1,320 Qwen3 calls (retail + airline): chars/token median 4.8, minimum 3.73.
CHARS_PER_TOKEN_LOWER_BOUND = 3.5  # below the observed minimum, so the estimate over-counts
OUTPUT_TOKEN_RESERVE = 4096  # room for the reply, including thinking tokens


class ContextOverflow(RuntimeError):
    """Raised before a local-model call whose prompt may not fit in `num_ctx`.

    It ends that conversation only (logged to failed_runs.jsonl); other runs continue.
    """


class Price(BaseModel):
    """USD per million tokens."""

    model_config = ConfigDict(extra="forbid")
    input: float = Field(ge=0)
    output: float = Field(ge=0)


class AgentRunConfig(BaseModel):
    """One collection run: one domain and agent model; prompt/user/fault conditions are mixed."""

    model_config = ConfigDict(extra="forbid")

    run_name: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    num_tasks: int | None = Field(default=None, ge=1)  # None = every task in the domain
    task_ids: list[str] | None = None  # overrides num_tasks
    task_sample_seed: int = 0
    num_trials: int = Field(default=1, ge=1)
    seed: int = 300
    agent_model: str
    agent_llm_args: dict[str, Any] = Field(default_factory=dict)
    user_model: str
    user_llm_args: dict[str, Any] = Field(default_factory=dict)
    prompt_variants: dict[PromptVariant, float] = Field(
        default_factory=lambda: {"tau2_default": 1.0}
    )
    user_variants: dict[UserVariant, float] = Field(default_factory=lambda: {"tau2_default": 1.0})
    faults: FaultConfig | None = None  # None = never inject tool faults
    max_concurrency: int = Field(default=1, ge=1)
    max_steps: int = Field(default=100, ge=1)
    budget_usd: float = Field(ge=0)
    requests_per_minute: float | None = Field(default=None, gt=0)  # all calls, every model
    model_limits: dict[str, ModelLimit] = Field(default_factory=dict)  # per-model RPM / input TPM
    prices: dict[str, Price] = Field(default_factory=dict)  # used before LiteLLM's price table
    output_dir: Path = Path("data/traces")
    cache_dir: Path = Path("data/llm_cache")
    cost_log: Path = Path("data/costs.jsonl")

    @model_validator(mode="after")
    def _weights_valid(self) -> AgentRunConfig:
        for name, weights in (("prompt", self.prompt_variants), ("user", self.user_variants)):
            if not weights or any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
                raise ValueError(f"{name}_variants needs non-negative weights with a positive sum")
        return self

    @model_validator(mode="after")
    def _local_models_need_num_ctx(self) -> AgentRunConfig:
        for model, args in (
            (self.agent_model, self.agent_llm_args),
            (self.user_model, self.user_llm_args),
        ):
            if model.startswith("ollama") and not args.get("num_ctx"):
                raise ValueError(
                    f"{model}: set num_ctx in its llm_args; Ollama's small default context "
                    "silently truncates the policy"
                )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> AgentRunConfig:
        """Load and validate a run config file."""
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def as_smoke(self, out_root: Path) -> AgentRunConfig:
        """Offline variant: tau2 mock domain, scripted fake LLM, isolated cache and cost log."""
        run_dir = out_root / f"{self.run_name}_smoke"
        return self.model_copy(
            update={
                "run_name": f"{self.run_name}_smoke",
                "domain": SMOKE_DOMAIN,
                "task_ids": SMOKE_TASK_IDS,
                "num_trials": 1,
                "max_concurrency": 1,
                "agent_model": "smoke/agent",
                "user_model": "smoke/user",
                "requests_per_minute": None,
                "prices": {m: Price(input=0, output=0) for m in ("smoke/agent", "smoke/user")},
                "output_dir": out_root,
                "cache_dir": run_dir / "llm_cache",
                "cost_log": run_dir / "costs.jsonl",
            }
        )


def run(cfg: AgentRunConfig, *, smoke: bool = False) -> Path:
    """Collect runs for `cfg`; return the path of the traces file."""
    _point_tau2_at_data()
    # tau2 is imported here, not at module level, so `--help` and tests stay fast, and so
    # TAU2_DATA_DIR is set before tau2 reads it at import time.
    import litellm
    import tau2.utils.llm_utils as tau2_llm
    from tau2.data_model.message import ToolMessage
    from tau2.environment.environment import Environment
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.run import TextRunConfig, get_tasks, run_single_task

    run_dir = cfg.output_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw.jsonl"

    cached = CachedCompletion(
        _fake_completion if smoke else _guard_context(litellm.completion),
        cache_dir=cfg.cache_dir,
        cost_log=cfg.cost_log,
        budget_usd=cfg.budget_usd,
        cost_fn=lambda r: _price(r, cfg.prices, litellm.completion_cost),
        to_dict=lambda r: r.to_dict(),
        from_dict=lambda d: litellm.ModelResponse(**d),
        requests_per_minute=cfg.requests_per_minute,
        rate_limiter=RateLimiter(cfg.model_limits) if cfg.model_limits else None,
    )
    tau2_llm.completion = cached  # tau2 calls `completion` from this module for every LLM call
    register_prompt_variants()
    if cfg.faults is not None:
        install_fault_hook(Environment, ToolMessage)

    tasks = _select_tasks(get_tasks(cfg.domain, task_split_name="base"), cfg)
    done = _done_runs(raw_path)
    pending = pending_runs(tasks, cfg.num_trials, done)
    logger.info(
        "%d tasks x %d trials; %d runs already done; %d to run with concurrency %d",
        len(tasks),
        cfg.num_trials,
        len(done),
        len(pending),
        cfg.max_concurrency,
    )

    def run_one(task: Any, trial: int) -> dict[str, Any]:
        prompt, user, faulty = choose_conditions(cfg, task.id, trial)
        seed = cfg.seed + trial
        injector = (
            FaultInjector(cfg.faults, f"{cfg.seed}:{task.id}:{trial}:faults")
            if (faulty and cfg.faults is not None)
            else None
        )
        tau2_cfg = TextRunConfig(
            domain=cfg.domain,
            agent=AGENT_NAMES[prompt],
            llm_agent=cfg.agent_model,
            llm_args_agent=cfg.agent_llm_args,
            llm_user=cfg.user_model,
            llm_args_user=cfg.user_llm_args,
            max_steps=cfg.max_steps,
            seed=cfg.seed,
        )
        ns_token = cache_namespace.set(f"{cfg.run_name}/{task.id}/{trial}")
        fault_token = active_injector.set(injector)
        try:
            sim = run_single_task(
                tau2_cfg,
                with_user_variant(task, user),
                seed=seed,
                evaluation_type=EvaluationType.ENV,
            )
        finally:
            active_injector.reset(fault_token)
            cache_namespace.reset(ns_token)
        info = RunInfo(
            domain=cfg.domain,
            agent_model=cfg.agent_model,
            user_model=cfg.user_model,
            prompt_variant=prompt,
            user_variant=user,
            fault=injector.summary() if injector else None,
            trial=trial,
            seed=seed,
        )
        return {
            "run_info": info.model_dump(),
            "simulation": sim.model_dump(mode="json"),
            "faults": injector.events if injector else [],
        }

    stop = threading.Event()
    lock = threading.Lock()
    failures = 0

    def work(item: tuple[Any, int]) -> None:
        nonlocal failures
        task, trial = item
        if stop.is_set():
            return
        try:
            record = run_one(task, trial)
        except BudgetExceeded as e:
            logger.error("Stopping: %s", e)
            stop.set()
            return
        except Exception as e:
            # E.g. an empty agent reply: tau2 raises and the trajectory is lost. Recording the
            # loss keeps it measurable (and resume retries the run next time).
            logger.exception("Run failed: task %s trial %d", task.id, trial)
            prompt, user, faulty = choose_conditions(cfg, task.id, trial)
            loss = {
                "task_id": task.id,
                "trial": trial,
                "error": f"{type(e).__name__}: {e}"[:300],
                "prompt_variant": prompt,
                "user_variant": user,
                "faulty": faulty,
            }
            transient = is_transient(e)
            with lock:
                with (run_dir / "failed_runs.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({**loss, "transient": transient}) + "\n")
                if not transient:
                    failures += 1
                    if failures >= MAX_CONSECUTIVE_FAILURES:
                        logger.error("Stopping after %d consecutive failures", failures)
                        stop.set()
            if (
                transient
                and type(e).__name__ != "ContextOverflow"
                and "content or tool_calls" not in str(e)
            ):
                time.sleep(TRANSIENT_PAUSE_S)  # let the provider's per-minute quota recover
            return
        with lock:
            failures = 0
            with raw_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=cfg.max_concurrency) as pool:
        list(pool.map(work, pending))
    return _finish(run_dir, raw_path, cached)


def is_transient(error: BaseException) -> bool:
    """True for errors that end one run without implying the setup is broken."""
    if type(error).__name__ in TRANSIENT_ERROR_NAMES:
        return True
    # tau2 raises this when the agent (or simulator) sends an empty message.
    return isinstance(error, ValueError) and "must have either content or tool_calls" in str(error)


def pending_runs(tasks: list[Any], num_trials: int, done: set[tuple[str, int]]) -> list[Any]:
    """(task, trial) pairs still to run, trial-major: every task once, then every task again.

    A long collection is cut short by Colab sessions; trial-major order keeps task coverage
    balanced wherever it stops (task-major order would give early tasks all their trials first).
    """
    return [(t, tr) for tr in range(num_trials) for t in tasks if (t.id, tr) not in done]


def choose_conditions(cfg: AgentRunConfig, task_id: str, trial: int) -> tuple[str, str, bool]:
    """Seeded (prompt variant, user variant, inject faults?) for one (task, trial)."""
    rng = random.Random(f"{cfg.seed}:{task_id}:{trial}:conditions")
    prompt = rng.choices(list(cfg.prompt_variants), weights=list(cfg.prompt_variants.values()))[0]
    user = rng.choices(list(cfg.user_variants), weights=list(cfg.user_variants.values()))[0]
    faulty = cfg.faults is not None and rng.random() < cfg.faults.run_rate
    return prompt, user, faulty


def _point_tau2_at_data() -> None:
    """tau2's pip package ships no data; use the pinned checkout from `atl-agent fetch-data`."""
    data_dir = Path(os.environ.get("TAU2_DATA_DIR", DEFAULT_TAU2_CHECKOUT / "data"))
    if not (data_dir / "tau2" / "domains").is_dir():
        raise FileNotFoundError(
            f"tau2 data not found at {data_dir}; run `atl-agent fetch-data` or set TAU2_DATA_DIR"
        )
    os.environ["TAU2_DATA_DIR"] = str(data_dir.resolve())


def _select_tasks(all_tasks: list[Any], cfg: AgentRunConfig) -> list[Any]:
    """Pick tasks by id, or a seeded random sample, so pilots are not just the first N tasks."""
    if cfg.task_ids is not None:
        by_id = {t.id: t for t in all_tasks}
        missing = [i for i in cfg.task_ids if i not in by_id]
        if missing:
            raise ValueError(f"unknown task ids for {cfg.domain}: {missing}")
        return [by_id[i] for i in cfg.task_ids]
    if cfg.num_tasks is None or cfg.num_tasks >= len(all_tasks):
        return list(all_tasks)
    return random.Random(cfg.task_sample_seed).sample(list(all_tasks), cfg.num_tasks)


def _done_runs(raw_path: Path) -> set[tuple[str, int]]:
    """(task id, trial) pairs already recorded, for resuming."""
    if not raw_path.exists():
        return set()
    done = set()
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                done.add((rec["simulation"]["task_id"], rec["run_info"]["trial"]))
    return done


def _finish(run_dir: Path, raw_path: Path, cached: CachedCompletion) -> Path:
    """Regenerate traces.jsonl from all raw runs and log a summary."""
    traces_path = run_dir / "traces.jsonl"
    traces, excluded = convert_raw_file(raw_path) if raw_path.exists() else ([], {})
    save_traces(traces, traces_path)
    n_flagged = _write_review_flags(raw_path, run_dir / "review_flags.jsonl")
    known = [t.env_outcome.task_success for t in traces if t.env_outcome]
    known = [s for s in known if s is not None]
    rate = sum(known) / len(known) if known else float("nan")
    logger.info(
        "Run summary: %d traces (excluded %s, flagged for review %d), env success rate %.2f, "
        "LLM calls %d paid / %d cached, total logged spend $%.4f",
        len(traces),
        dict(excluded),
        n_flagged,
        rate,
        cached.misses,
        cached.hits,
        cached.spent_usd,
    )
    return traces_path


def _write_review_flags(raw_path: Path, out: Path) -> int:
    """Write {"trace_id", "flags"} for runs needing human review; return how many."""
    rows = []
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sim = json.loads(line)["simulation"]
                    if flags := review_flags(sim):
                        rows.append({"trace_id": f"tau2-{sim['id']}", "flags": flags})
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return len(rows)


def _guard_context(completion_fn: Any) -> Any:
    """Wrap `completion` so calls with `num_ctx` refuse prompts that may not fit."""

    def guarded(**kwargs: Any) -> Any:
        num_ctx = kwargs.get("num_ctx")
        if num_ctx:
            chars = len(json.dumps(kwargs.get("messages"), default=str))
            chars += len(json.dumps(kwargs.get("tools"), default=str))
            worst_case = chars / CHARS_PER_TOKEN_LOWER_BOUND + OUTPUT_TOKEN_RESERVE
            if worst_case > num_ctx:
                raise ContextOverflow(
                    f"{kwargs.get('model')}: prompt of {chars} chars may need ~{worst_case:.0f} "
                    f"tokens incl. reply, over num_ctx={num_ctx}; raise num_ctx"
                )
        return completion_fn(**kwargs)

    return guarded


def _price(response: Any, prices: dict[str, Price], fallback: Any) -> float:
    """Price a response from the config table, else LiteLLM's table (may raise if unknown)."""
    model = str(response.model or "").removeprefix("models/")
    match = next((p for name, p in prices.items() if name.split("/")[-1] == model), None)
    if match is None:
        return float(fallback(completion_response=response))
    usage = response.usage
    return (usage.prompt_tokens * match.input + usage.completion_tokens * match.output) / 1e6


def _fake_completion(**kwargs: Any) -> Any:
    """Scripted offline LLM for --smoke on the tau2 mock domain (task create_task_1)."""
    import litellm

    model, messages = kwargs["model"], kwargs["messages"]
    tool_calls = None
    if model == "smoke/user":
        done = any("Done" in str(m.get("content") or "") for m in messages)
        content = "###STOP###" if done else "Please create a task 'Important Meeting' for user_1."
    elif any(m.get("role") == "tool" for m in messages):
        content = "Done. The task is created."
    else:
        content = None
        args = {"user_id": "user_1", "title": "Important Meeting"}
        tool_calls = [
            {
                "id": "call_smoke_1",
                "type": "function",
                "function": {"name": "create_task", "arguments": json.dumps(args)},
            }
        ]
    message = {"role": "assistant", "content": content, "tool_calls": tool_calls}
    return litellm.ModelResponse(
        model=model,
        choices=[{"index": 0, "finish_reason": "stop", "message": message}],
        usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
    )


app = typer.Typer(help="Run tau2-bench agents and record traces.", no_args_is_help=True)


@app.callback()
def _main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


def run_stats(run_dir: Path) -> dict[str, Any]:
    """Throughput, outcome and cost summary of a run directory (raw.jsonl + review flags)."""
    sims = []
    raw_path = run_dir / "raw.jsonl"
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as f:
            sims = [json.loads(line)["simulation"] for line in f if line.strip()]
    if not sims:
        return {"runs": 0}
    rewards = [(s.get("reward_info") or {}).get("reward") for s in sims]
    scored = [r for r in rewards if r is not None]
    seconds = sum(s.get("duration") or 0.0 for s in sims)
    cost = sum((s.get("agent_cost") or 0.0) + (s.get("user_cost") or 0.0) for s in sims)
    # Wall clock from the first start to the last end, so parallel runs are not under-counted
    # (summing per-run durations would divide throughput by the concurrency).
    starts = [datetime.fromisoformat(s["start_time"]) for s in sims if s.get("start_time")]
    ends = [datetime.fromisoformat(s["end_time"]) for s in sims if s.get("end_time")]
    wall = (max(ends) - min(starts)).total_seconds() if starts and ends else 0.0
    flags_path = run_dir / "review_flags.jsonl"
    flagged = len(flags_path.read_text().splitlines()) if flags_path.exists() else 0
    return {
        "runs": len(sims),
        "env_success_rate": round(sum(r == 1.0 for r in scored) / len(scored), 3)
        if scored
        else None,
        "terminations": dict(Counter(s.get("termination_reason") for s in sims)),
        "mean_minutes_per_run": round(seconds / len(sims) / 60, 2),
        "runs_per_hour_wall_clock": round(len(sims) / wall * 3600, 1) if wall else None,
        "mean_cost_usd": round(cost / len(sims), 4),
        # Simulator health: the A/B rejected a customer that repeated itself; keep watching it.
        "customer_loop_runs": sum(_customer_loops(s) for s in sims),
        "flagged_for_review": flagged,
    }


def _customer_loops(sim: dict[str, Any], repeats: int = 5) -> bool:
    """True if the simulated customer sent the same message `repeats` or more times."""
    texts = [m.get("content") or "" for m in sim.get("messages") or [] if m.get("role") == "user"]
    return bool(texts) and Counter(texts).most_common(1)[0][1] >= repeats


@app.command("stats")
def stats_cmd(
    run_dir: Annotated[Path, typer.Option("--run-dir", help="data/traces/<run>.")],
) -> None:
    """Print throughput, success rate and cost for a run directory."""
    logger.info("%s: %s", run_dir, json.dumps(run_stats(run_dir)))


@app.command("fetch-data")
def fetch_data(
    dest: Annotated[Path, typer.Option(help="Checkout directory.")] = DEFAULT_TAU2_CHECKOUT,
) -> None:
    """Clone tau2-bench at the pinned commit (tasks, policies, DBs) for TAU2_DATA_DIR."""
    if not dest.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", TAU2_REPO, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", TAU2_COMMIT_FULL], check=True)
    head = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != TAU2_COMMIT_FULL:
        raise RuntimeError(f"{dest} is at {head}, expected {TAU2_COMMIT_FULL}")
    logger.info("tau2 data ready at %s (commit %s)", dest / "data", head[:7])


@app.command("run")
def run_cmd(
    config: Annotated[Path, typer.Option("--config", help="Agent run config YAML.")],
    smoke: Annotated[bool, typer.Option(help="Offline: mock domain, fake LLM, $0.")] = False,
    smoke_dir: Annotated[Path, typer.Option(help="Output root for --smoke.")] = Path("data/smoke"),
) -> None:
    """Run the agent on the configured tasks."""
    cfg = AgentRunConfig.from_yaml(config)
    if smoke:
        cfg = cfg.as_smoke(smoke_dir)
    else:
        from dotenv import load_dotenv

        load_dotenv()
    path = run(cfg, smoke=smoke)
    logger.info("Traces written to %s", path)
