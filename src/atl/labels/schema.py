"""Label records: one judgement of one trace by one annotator (a person or an LLM labeller).

Labels live apart from traces (see atl.traces.schema), keyed by `trace_id`. Label files are
append-only JSONL; re-labelling a trace appends a new record and the latest one wins, so the
full editing history is kept.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from atl.labels.taxonomy import FailureCode

logger = logging.getLogger(__name__)

NonEmptyStr = Annotated[str, Field(min_length=1)]
Confidence = Literal["low", "medium", "high"]


class Label(BaseModel):
    """One annotator's verdict on one trace.

    `failed` is None when the trace cannot be judged (e.g. the user simulator broke); the
    rationale must then say why. `other_failure` describes a failure no class fits; it exists
    to find gaps in the draft taxonomy and should be rare once the taxonomy is frozen.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: NonEmptyStr
    failed: bool | None
    classes: list[FailureCode] = Field(default_factory=list)
    first_failure_step: int | None = Field(default=None, ge=0)  # index into Trace.steps
    other_failure: str | None = None
    confidence: Confidence
    rationale: str = ""
    annotator: NonEmptyStr  # e.g. "human", "human-relabel", "llm:gemini-2.5-flash:p1"
    taxonomy_version: NonEmptyStr
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("classes")
    @classmethod
    def _sorted_unique(cls, v: list[str]) -> list[str]:
        return sorted(set(v))

    @field_validator("other_failure", "rationale")
    @classmethod
    def _strip(cls, v: str | None) -> str | None:
        return v.strip() if v is not None else None

    @model_validator(mode="after")
    def _consistent(self) -> Label:
        has_failure = bool(self.classes) or bool(self.other_failure)
        if self.failed is True:
            if not has_failure:
                raise ValueError("a failed trace needs at least one class or other_failure")
            if self.first_failure_step is None:
                raise ValueError("a failed trace needs first_failure_step")
        else:
            if has_failure or self.first_failure_step is not None:
                raise ValueError("classes, other_failure and first_failure_step need failed=True")
            if self.failed is None and not self.rationale:
                raise ValueError("an unjudgeable trace (failed=None) needs a rationale")
        return self


def append_label(label: Label, path: str | Path) -> None:
    """Append one label to a JSONL file, creating it (and its folder) if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(label.model_dump_json() + "\n")


def load_labels(path: str | Path) -> dict[str, Label]:
    """Load a label file; the last record for each trace_id wins. A missing file is empty."""
    path = Path(path)
    if not path.is_file():
        return {}
    labels: dict[str, Label] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                label = Label.model_validate_json(line)
            except ValidationError as e:
                raise ValueError(f"{path}:{line_no}: invalid label\n{e}") from e
            labels[label.trace_id] = label
    logger.info("Loaded %d labels from %s", len(labels), path)
    return labels
