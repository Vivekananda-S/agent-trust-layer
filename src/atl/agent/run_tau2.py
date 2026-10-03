"""Run a tau2-bench agent on sampled tasks and record raw runs plus normalised traces.

Usage:
    atl-agent run --config configs/agent/pilot_retail.yaml
    atl-agent run --config configs/agent/pilot_retail.yaml --smoke   # offline, fake LLM, $0

Every LLM call (agent and user simulator) goes through `CachedCompletion`: disk cache, cost
log, budget cap and rate limit. Runs resume: (task, trial) pairs already in the raw file are
skipped, so a killed session loses at most the run in progress.

Outputs, under `<output_dir>/<run_name>/`:
    raw.jsonl     one {"run_info", "simulation"} record per run (full tau2 output, for audit)
    traces.jsonl  normalised traces, regenerated from raw.jsonl at the end of every invocation
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from pydantic import BaseModel, ConfigDict, Field

from atl.agent.llm_cache import BudgetExceeded, CachedCompletion, cache_namespace
from atl.traces.adapters.tau2 import RunInfo, convert_raw_file
from atl.traces.schema import save_traces

logger = logging.getLogger(__name__)

TAU2_REPO = "https://github.com/sierra-research/tau2-bench.git"
TAU2_COMMIT_FULL = "5bfa7e37b36656b37dc6d022156be6563c1007f3"  # same pin as pyproject.toml
DEFAULT_TAU2_CHECKOUT = Path("external/tau2-bench")
SMOKE_DOMAIN = "mock"
SMOKE_TASK_IDS = ["create_task_1"]
MAX_CONSECUTIVE_FAILURES = 3  # e.g. a free-tier daily quota is exhausted: stop, resume tomorrow


class Price(BaseModel):
    """USD per million tokens."""

    model_config = ConfigDict(extra="forbid")
    input: float = Field(ge=0)
    output: float = Field(ge=0)


class AgentRunConfig(BaseModel):
    """One collection run: one domain, one agent model, one prompt and user variant."""

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
    prompt_variant: str = "tau2_default"
    user_variant: str = "tau2_default"
    max_steps: int = Field(default=100, ge=1)
    budget_usd: float = Field(ge=0)
    requests_per_minute: float | None = Field(default=None, gt=0)
    prices: dict[str, Price] = Field(default_factory=dict)  # used before LiteLLM's price table
    output_dir: Path = Path("data/traces")
    cache_dir: Path = Path("data/llm_cache")
    cost_log: Path = Path("data/costs.jsonl")

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
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.run import TextRunConfig, get_tasks, run_single_task

    run_dir = cfg.output_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw.jsonl"

    cached = CachedCompletion(
        _fake_completion if smoke else litellm.completion,
        cache_dir=cfg.cache_dir,
        cost_log=cfg.cost_log,
        budget_usd=cfg.budget_usd,
        cost_fn=lambda r: _price(r, cfg.prices, litellm.completion_cost),
        to_dict=lambda r: r.to_dict(),
        from_dict=lambda d: litellm.ModelResponse(**d),
        requests_per_minute=cfg.requests_per_minute,
    )
    tau2_llm.completion = cached  # tau2 calls `completion` from this module for every LLM call

    tasks = _select_tasks(get_tasks(cfg.domain, task_split_name="base"), cfg)
    done = _done_runs(raw_path)
    tau2_cfg = TextRunConfig(
        domain=cfg.domain,
        llm_agent=cfg.agent_model,
        llm_args_agent=cfg.agent_llm_args,
        llm_user=cfg.user_model,
        llm_args_user=cfg.user_llm_args,
        max_steps=cfg.max_steps,
        seed=cfg.seed,
    )
    logger.info("%d tasks x %d trials; %d runs already done", len(tasks), cfg.num_trials, len(done))

    failures = 0
    for task in tasks:
        for trial in range(cfg.num_trials):
            if (task.id, trial) in done:
                continue
            seed = cfg.seed + trial
            token = cache_namespace.set(f"{cfg.run_name}/{task.id}/{trial}")
            try:
                sim = run_single_task(tau2_cfg, task, seed=seed, evaluation_type=EvaluationType.ENV)
            except BudgetExceeded as e:
                logger.error("Stopping: %s", e)
                return _finish(run_dir, raw_path, cached)
            except Exception:
                failures += 1
                logger.exception("Run failed: task %s trial %d", task.id, trial)
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.error("Stopping after %d consecutive failures", failures)
                    return _finish(run_dir, raw_path, cached)
                continue
            finally:
                cache_namespace.reset(token)
            failures = 0
            info = RunInfo(
                domain=cfg.domain,
                agent_model=cfg.agent_model,
                user_model=cfg.user_model,
                prompt_variant=cfg.prompt_variant,
                user_variant=cfg.user_variant,
                trial=trial,
                seed=seed,
            )
            record = {"run_info": info.model_dump(), "simulation": sim.model_dump(mode="json")}
            with raw_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return _finish(run_dir, raw_path, cached)


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
    known = [t.env_outcome.task_success for t in traces if t.env_outcome]
    known = [s for s in known if s is not None]
    rate = sum(known) / len(known) if known else float("nan")
    logger.info(
        "Run summary: %d traces (excluded %s), env success rate %.2f, "
        "LLM calls %d paid / %d cached, total logged spend $%.4f",
        len(traces),
        dict(excluded),
        rate,
        cached.misses,
        cached.hits,
        cached.spent_usd,
    )
    return traces_path


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
