"""Stratified seed-set sampling for hand labelling, drawn from the train split only.

Strata are agent model x domain x injected-fault group x environment outcome. Picks go
round-robin over the strata, so any prefix of the queue (e.g. the first 100 traces labelled
before the taxonomy review) already covers every stratum. Rare strata are over-represented on
purpose: the seed set needs positives for rare classes, not the population base rate.
"""

from __future__ import annotations

import glob
import json
import logging
import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from atl.data.splits import load_manifest, load_split
from atl.traces.schema import Trace, load_traces

logger = logging.getLogger(__name__)

SEED_SPLIT = "train"  # prompt tuning happens on the seed set, so it must never touch val or test


class SeedConfig(BaseModel):
    """Settings for the seed queue and the labelling app (configs/labels/seed_v1.yaml)."""

    model_config = ConfigDict(extra="forbid")

    traces: list[str] = Field(min_length=1)  # glob patterns of trace JSONL files
    split_manifest: Path
    n: int = Field(ge=1)
    seed: int
    max_per_task: int = Field(ge=1)
    queue_path: Path
    labels_path: Path

    @classmethod
    def from_yaml(cls, path: str | Path) -> SeedConfig:
        """Load and validate a seed config file."""
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


class QueueItem(BaseModel):
    """One entry of the labelling queue."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    position: int = Field(ge=0)
    trace_id: str
    stratum: str


def fault_group(fault: str | None) -> str:
    """Collapse injected-fault combinations into none / injection / other."""
    if not fault:
        return "none"
    return "injection" if "injection" in fault.split("+") else "other"


def stratum_of(trace: Trace) -> str:
    """The stratum key of a trace, e.g. 'openai/qwen3-8b|retail|injection|fail'."""
    success = trace.env_outcome.task_success if trace.env_outcome else None
    outcome = {True: "pass", False: "fail", None: "unknown"}[success]
    return "|".join([trace.agent_model, trace.meta.domain, fault_group(trace.meta.fault), outcome])


def build_seed_queue(
    train_traces: Iterable[Trace], n: int, seed: int, max_per_task: int
) -> list[QueueItem]:
    """Pick up to `n` traces round-robin over strata, at most `max_per_task` per task."""
    rng = random.Random(seed)
    by_stratum: dict[str, list[Trace]] = defaultdict(list)
    for t in sorted(train_traces, key=lambda t: t.trace_id):
        by_stratum[stratum_of(t)].append(t)
    pools = {}
    for key in sorted(by_stratum):
        pool = by_stratum[key]
        rng.shuffle(pool)
        pools[key] = pool

    picked: list[QueueItem] = []
    per_task: dict[str, int] = defaultdict(int)
    while len(picked) < n and any(pools.values()):
        for key, pool in pools.items():
            while pool:  # take the next trace whose task still has room
                t = pool.pop()
                if per_task[t.task_id] < max_per_task:
                    per_task[t.task_id] += 1
                    picked.append(QueueItem(position=len(picked), trace_id=t.trace_id, stratum=key))
                    break
            if len(picked) == n:
                break
    if len(picked) < n:
        logger.warning("Only %d traces available for a seed queue of %d", len(picked), n)
    return picked


def load_train_traces(cfg: SeedConfig) -> list[Trace]:
    """Load every trace file in the config and keep only the train split."""
    paths = sorted({p for pattern in cfg.traces for p in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError(f"no trace files match {cfg.traces}")
    traces = [t for p in paths for t in load_traces(p)]
    return load_split(traces, load_manifest(cfg.split_manifest), SEED_SPLIT)


def save_queue(queue: list[QueueItem], path: str | Path) -> None:
    """Write the queue as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in queue:
            f.write(item.model_dump_json() + "\n")


def load_queue(path: str | Path) -> list[QueueItem]:
    """Read a queue JSONL file, in queue order."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    items = [QueueItem.model_validate(json.loads(line)) for line in lines if line.strip()]
    return sorted(items, key=lambda i: i.position)


def queued_traces(cfg: SeedConfig) -> list[tuple[QueueItem, Trace]]:
    """The queue joined with its traces. Every queued trace must still be in the train split."""
    by_id = {t.trace_id: t for t in load_train_traces(cfg)}
    queue = load_queue(cfg.queue_path)
    missing = [i.trace_id for i in queue if i.trace_id not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} queued traces are not train traces, e.g. {missing[0]}")
    return [(i, by_id[i.trace_id]) for i in queue]
