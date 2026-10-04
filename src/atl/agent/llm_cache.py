"""Persistent on-disk cache, cost log, budget cap and rate limit for paid LLM calls.

`CachedCompletion` wraps any `completion(**kwargs)` function (LiteLLM's in practice):

- The cache key is the sha256 of the model, messages, tools and sampling parameters, plus a
  namespace. The runner sets the namespace per (task, trial), so repeated trials of the same
  task are separate samples instead of replays of the first one.
- A cache hit returns the stored response and costs nothing; a paid call is never repeated.
- Every call is appended to a JSONL cost log. Spend is read back from that log on start-up, so
  the budget cap holds across sessions, not just within one process.
- Before each paid call, the cap is checked (`BudgetExceeded`) and calls are spaced to stay
  under a requests-per-minute limit (free tiers enforce this).
- An optional per-model `RateLimiter` paces paid calls under provider caps (requests and input
  tokens per minute), reserving an estimate before the call and reconciling it after.
- Thread-safe for parallel runs: the lock guards only bookkeeping (budget check, rate-limit
  slot, counters, log writes), never the LLM call itself. With N calls in flight the cap can be
  overshot by at most N calls' cost.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atl.agent.rate_limit import RateLimiter, estimate_input_tokens

logger = logging.getLogger(__name__)

# Transport-only kwargs that do not change the model's output.
_KEY_EXCLUDED_KWARGS = {"num_retries", "timeout", "api_key", "api_base", "metadata"}

cache_namespace: contextvars.ContextVar[str] = contextvars.ContextVar("cache_namespace", default="")


class BudgetExceeded(RuntimeError):
    """Raised before a paid call that would start once the budget is used up."""


def cache_key(kwargs: dict[str, Any], namespace: str) -> str:
    """Stable sha256 of everything that determines the model's output."""
    keyed = {k: v for k, v in kwargs.items() if k not in _KEY_EXCLUDED_KWARGS}
    payload = json.dumps(
        {"namespace": namespace, "request": keyed}, sort_keys=True, default=str, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class CachedCompletion:
    """Drop-in wrapper for a `completion(**kwargs)` function. See module docstring."""

    def __init__(
        self,
        completion_fn: Callable[..., Any],
        *,
        cache_dir: str | Path,
        cost_log: str | Path,
        budget_usd: float,
        cost_fn: Callable[[Any], float],
        to_dict: Callable[[Any], dict[str, Any]],
        from_dict: Callable[[dict[str, Any]], Any],
        requests_per_minute: float | None = None,
        rate_limiter: RateLimiter | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._completion = completion_fn
        self._cache_dir = Path(cache_dir)
        self._cost_log = Path(cost_log)
        self._budget = budget_usd
        self._cost_fn = cost_fn
        self._to_dict = to_dict
        self._from_dict = from_dict
        self._min_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._rate_limiter = rate_limiter
        self._clock = clock
        self._sleep = sleep
        self._next_slot = float("-inf")  # earliest start time for the next paid call
        self._lock = threading.Lock()
        self.spent_usd = self._read_spend()
        self.hits = 0
        self.misses = 0
        logger.info("LLM budget: spent $%.4f of $%.2f so far", self.spent_usd, self._budget)

    def __call__(self, **kwargs: Any) -> Any:
        namespace = cache_namespace.get()
        key = cache_key(kwargs, namespace)
        path = self._cache_dir / key[:2] / f"{key}.json"
        model = str(kwargs.get("model"))

        if path.exists():
            with self._lock:
                self.hits += 1
            record = json.loads(path.read_text(encoding="utf-8"))
            self._log(key, model, namespace, cost=0.0, cached=True, usage=record.get("usage"))
            return self._from_dict(record["response"])

        with self._lock:
            if self.spent_usd >= self._budget:
                raise BudgetExceeded(
                    f"spent ${self.spent_usd:.4f} of ${self._budget:.2f}; raise budget_usd to go on"
                )
        # Per-model pacing (may wait) happens outside the lock, before the global slot.
        handle = (
            self._rate_limiter.acquire(model, estimate_input_tokens(kwargs))
            if self._rate_limiter
            else None
        )
        with self._lock:
            now = self._clock()
            start = max(now, self._next_slot)  # reserve the next rate-limit slot
            self._next_slot = start + self._min_interval
        if start > now:
            self._sleep(start - now)
        response = self._completion(**kwargs)  # outside the lock: parallel runs overlap here
        cost = self._safe_cost(response)
        with self._lock:
            self.misses += 1
            self.spent_usd += cost

        record = {"model": model, "namespace": namespace, "response": self._to_dict(response)}
        record["usage"] = record["response"].get("usage")
        if self._rate_limiter:
            self._rate_limiter.reconcile(handle, (record["usage"] or {}).get("prompt_tokens"))
        _atomic_write(path, json.dumps(record, ensure_ascii=False))
        self._log(key, model, namespace, cost=cost, cached=False, usage=record["usage"])
        return response

    def _safe_cost(self, response: Any) -> float:
        try:
            return float(self._cost_fn(response))
        except Exception as e:  # unknown model price: log it, never crash a run over it
            logger.warning("Could not price response (%s); logging cost 0", e)
            return 0.0

    def _read_spend(self) -> float:
        if not self._cost_log.exists():
            return 0.0
        with self._cost_log.open(encoding="utf-8") as f:
            return sum(json.loads(line)["cost_usd"] for line in f if line.strip())

    def _log(
        self, key: str, model: str, namespace: str, *, cost: float, cached: bool, usage: Any
    ) -> None:
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "key": key,
            "model": model,
            "namespace": namespace,
            "cached": cached,
            "cost_usd": cost,
            "usage": usage,
        }
        self._cost_log.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._cost_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file and rename, so a killed process never leaves half a cache entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
