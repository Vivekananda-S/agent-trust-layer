"""Per-model sliding-window rate limiter: requests and input tokens per minute.

Free-tier APIs cap each model separately (Gemma 4 31B: 30 requests and 16K input tokens per
minute). Hitting the cap and retrying wastes requests: the retries land in the same exhausted
minute. Pacing calls to stay just under the cap gives the highest steady throughput.

Before a call, the caller reserves an *estimated* input-token count (deliberately an over-count);
after the call it reconciles the reservation with the provider's reported count. Models without
a configured limit pass straight through. Thread-safe: parallel runs share one limiter.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# Gemini tokenizer on our prompts: chars/token median 4.16, minimum 3.70 (219 calls). Dividing by
# a smaller number over-counts, so a reservation is never smaller than the real request.
CHARS_PER_TOKEN_ESTIMATE = 3.5


class ModelLimit(BaseModel):
    """Caps for one model over a sliding 60-second window. None means no cap of that kind."""

    model_config = ConfigDict(extra="forbid")

    rpm: float | None = Field(default=None, gt=0)
    input_tpm: float | None = Field(default=None, gt=0)


def estimate_input_tokens(kwargs: dict[str, Any]) -> int:
    """Over-estimate of a request's input tokens from the size of its messages and tools."""
    chars = len(json.dumps(kwargs.get("messages"), default=str))
    chars += len(json.dumps(kwargs.get("tools"), default=str)) if kwargs.get("tools") else 0
    return int(chars / CHARS_PER_TOKEN_ESTIMATE) + 1


class RateLimiter:
    """Blocks `acquire` until a request fits under its model's requests/tokens per window."""

    def __init__(
        self,
        limits: dict[str, ModelLimit],
        *,
        window_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._limits = limits
        self._window = window_s
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._entries: dict[str, deque[list[float]]] = {m: deque() for m in limits}

    def acquire(self, model: str, est_tokens: int) -> list[float] | None:
        """Wait until the request fits, then reserve it. Returns a handle for `reconcile`."""
        limit = self._limits.get(model)
        if limit is None:
            return None
        while True:
            with self._lock:
                now = self._clock()
                entries = self._entries[model]
                while entries and entries[0][0] <= now - self._window:
                    entries.popleft()
                used_tokens = sum(e[1] for e in entries)
                fits_rpm = limit.rpm is None or len(entries) + 1 <= limit.rpm
                # A single request larger than the whole cap is let through when the window is
                # empty; otherwise it could never run.
                fits_tpm = (
                    limit.input_tpm is None
                    or used_tokens + est_tokens <= limit.input_tpm
                    or not entries
                )
                if fits_rpm and fits_tpm:
                    entry = [now, float(est_tokens)]
                    entries.append(entry)
                    return entry
                wait = entries[0][0] + self._window - now
            logger.debug("Rate limit for %s: waiting %.1fs", model, wait)
            self._sleep(max(wait, 0.01))

    def reconcile(self, handle: list[float] | None, actual_tokens: int | None) -> None:
        """Replace a reservation's estimate with the provider's reported input tokens."""
        if handle is not None and actual_tokens is not None:
            with self._lock:
                handle[1] = float(actual_tokens)
