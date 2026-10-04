"""GPU cost ledger: keeps self-hosted collection on Modal inside a fixed credit.

Modal bills per GPU-second while the serving container is up. Our cost log cannot see that, so
every collection session records its wall time here, plus a fixed overhead for cold start and
scale-down. Before a session starts, the ledger turns the remaining budget into a time limit;
when spend reaches the cap, it refuses to start. Modal's own workspace spending limit is the
hard stop behind this; the ledger keeps us from ever reaching it.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class GpuBudget(BaseModel):
    """Budget for a self-hosted GPU endpoint (config block `gpu_budget`)."""

    model_config = ConfigDict(extra="forbid")

    gpu: str = Field(min_length=1)  # e.g. "H100"; recorded, and must match the deployed app
    hourly_usd: float = Field(gt=0)  # Modal list price for that GPU
    cap_usd: float = Field(gt=0)  # stop below the credit (e.g. 27 of a $30 credit)
    overhead_s: float = Field(default=900, ge=0)  # cold start + scale-down charged per session
    ledger: Path = Path("data/modal_ledger.jsonl")


class BudgetExhausted(RuntimeError):
    """Raised before a session when the ledger leaves no time to run."""


def spent_usd(budget: GpuBudget) -> float:
    """Estimated spend so far, summed over recorded sessions."""
    if not budget.ledger.exists():
        return 0.0
    with budget.ledger.open(encoding="utf-8") as f:
        return sum(json.loads(line)["est_usd"] for line in f if line.strip())


def session_seconds(budget: GpuBudget) -> float:
    """Seconds of collection the remaining budget allows, after per-session overhead."""
    remaining = budget.cap_usd - spent_usd(budget)
    seconds = remaining / budget.hourly_usd * 3600 - budget.overhead_s
    if seconds <= 0:
        raise BudgetExhausted(
            f"estimated GPU spend ${spent_usd(budget):.2f} leaves no session time under the "
            f"${budget.cap_usd:.2f} cap"
        )
    return seconds


def record_session(budget: GpuBudget, run_name: str, wall_s: float) -> float:
    """Append one session's estimated cost (wall time + overhead). Returns its cost in USD."""
    est = (wall_s + budget.overhead_s) / 3600 * budget.hourly_usd
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "run_name": run_name,
        "gpu": budget.gpu,
        "wall_s": round(wall_s, 1),
        "overhead_s": budget.overhead_s,
        "est_usd": round(est, 4),
    }
    budget.ledger.parent.mkdir(parents=True, exist_ok=True)
    with budget.ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info(
        "GPU session %s: %.0f s, est $%.2f; total est $%.2f of $%.2f cap",
        run_name,
        wall_s,
        est,
        spent_usd(budget),
        budget.cap_usd,
    )
    return est
