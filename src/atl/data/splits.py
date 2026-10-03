"""Task-level splits and the guarded loader for protected (gold and held-out) data.

Rules (see CLAUDE.md, "The gold test set rule"):

- Splits are by `task_id`, never by trace. Each in-domain task is assigned by hashing its id
  with a salt, so adding tasks later never moves an existing task between splits.
- The split manifest (task_id -> split) is written once, committed, and is the source of truth.
- Only `train`, `val` and `calib` may be used for training, tuning, calibration, threshold
  selection or early stopping (`load_split`).
- Protected data (`test`, `gold`, `heldout_model`, `ood`) is read only by `atl.eval.final_eval`,
  through `load_protected_split` and `load_gold`.

Trace routing: traces outside the in-domain list go to `ood`; traces from the held-out agent
model go to `heldout_model` if their task is a test task and are dropped otherwise (so the
drift test is not confounded by tasks seen in training); everything else follows its task.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated, Literal

import typer
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from atl.traces.schema import Trace, load_traces

logger = logging.getLogger(__name__)

TASK_SPLITS = ("train", "val", "calib", "test")  # hash-assigned; order defines the cut points
TRAINABLE_SPLITS = ("train", "val", "calib")
PROTECTED_SPLITS = ("test", "gold", "heldout_model", "ood")
ALLOWED_PROTECTED_CALLER = "atl.eval.final_eval"
DEFAULT_GOLD_DIR = Path("data/gold")
GOLD_MANIFEST_NAME = "MANIFEST.sha256"
GOLD_TRACES_NAME = "traces.jsonl"

TaskSplit = Literal["train", "val", "calib", "test", "ood"]


class ProtectedSplitError(RuntimeError):
    """Raised when protected data is requested outside the final evaluation."""


class LeakageError(RuntimeError):
    """Raised when traces and the split manifest disagree in a way that could leak tasks."""


class GoldIntegrityError(RuntimeError):
    """Raised when the gold directory does not match its sha256 manifest."""


class _SplitSettings(BaseModel):
    """Fields shared by the config and the manifest; they define the split."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    salt: str = Field(min_length=1)
    fractions: dict[str, float]
    heldout_model: str | None = None
    in_domains: list[str] = Field(min_length=1)  # every other domain is out-of-domain (ood)

    @model_validator(mode="after")
    def _check_fractions(self) -> _SplitSettings:
        if len(set(self.in_domains)) != len(self.in_domains) or "" in self.in_domains:
            raise ValueError("in_domains must be unique, non-empty names")
        if set(self.fractions) != set(TASK_SPLITS):
            raise ValueError(f"fractions must have exactly the keys {TASK_SPLITS}")
        if any(f <= 0 for f in self.fractions.values()):
            raise ValueError("every fraction must be positive")
        if abs(sum(self.fractions.values()) - 1.0) > 1e-9:
            raise ValueError("fractions must sum to 1")
        return self


class SplitConfig(_SplitSettings):
    """Split settings loaded from YAML (configs/splits.yaml)."""

    manifest_path: Path = Path("splits/splits_v1.json")

    @classmethod
    def from_yaml(cls, path: str | Path) -> SplitConfig:
        """Load and validate a split config file."""
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


class SplitManifest(_SplitSettings):
    """The frozen task -> split assignment, plus the settings that produced it."""

    task_splits: dict[str, TaskSplit]
    trace_counts: dict[str, int] = Field(default_factory=dict)  # informational, at build time


# --- assignment ------------------------------------------------------------------------------


def assign_split(task_id: str, salt: str, fractions: Mapping[str, float]) -> str:
    """Deterministically map a task id to train/val/calib/test via a salted sha256 hash."""
    digest = hashlib.sha256(f"{salt}:{task_id}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64  # uniform in [0, 1)
    cumulative = 0.0
    for name in TASK_SPLITS:
        cumulative += fractions[name]
        if u < cumulative:
            return name
    return TASK_SPLITS[-1]  # guard against float rounding when u is just below 1


def trace_split(trace: Trace, manifest: SplitManifest) -> str | None:
    """Return the split a trace belongs to, or None if it is deliberately dropped."""
    task_split = manifest.task_splits.get(trace.task_id)
    if task_split is None:
        raise LeakageError(
            f"task {trace.task_id!r} (trace {trace.trace_id!r}) is not in the split manifest; "
            "extend it with `atl-splits make --update`"
        )
    is_ood = trace.meta.domain not in manifest.in_domains
    if is_ood != (task_split == "ood"):
        raise LeakageError(
            f"trace {trace.trace_id!r}: domain {trace.meta.domain!r} disagrees with "
            f"task {trace.task_id!r} assigned to {task_split!r}"
        )
    if is_ood:
        return "ood"
    if trace.agent_model == manifest.heldout_model:
        return "heldout_model" if task_split == "test" else None
    return task_split


def build_manifest(
    traces: Iterable[Trace], cfg: SplitConfig, existing: SplitManifest | None = None
) -> SplitManifest:
    """Assign every task to a split. With `existing`, keep its assignments and add new tasks."""
    traces = list(traces)
    task_domains: dict[str, set[str]] = defaultdict(set)
    for trace in traces:
        task_domains[trace.task_id].add(trace.meta.domain)
    mixed = sorted(t for t, domains in task_domains.items() if len(domains) > 1)
    if mixed:
        raise LeakageError(f"tasks span more than one domain: {mixed[:10]}")

    settings = cfg.model_dump(exclude={"manifest_path"})
    task_splits: dict[str, str] = {}
    if existing is not None:
        old = existing.model_dump(exclude={"task_splits", "trace_counts"})
        if old != settings:
            raise ValueError(f"config {settings} does not match existing manifest {old}")
        task_splits = dict(existing.task_splits)

    for task_id, domains in sorted(task_domains.items()):
        if task_id in task_splits:
            continue
        (domain,) = domains
        if domain in cfg.in_domains:
            task_splits[task_id] = assign_split(task_id, cfg.salt, cfg.fractions)
        else:
            task_splits[task_id] = "ood"

    manifest = SplitManifest(**settings, task_splits=dict(sorted(task_splits.items())))
    counts = Counter(trace_split(t, manifest) or "dropped" for t in traces)
    manifest.trace_counts = dict(sorted(counts.items()))
    return manifest


def check_no_leakage(manifest: SplitManifest, traces: Iterable[Trace]) -> None:
    """Fail if the manifest was edited by hand or any trace cannot be routed consistently.

    A task appears in exactly one split by construction (the manifest is a dict keyed by
    task_id), so this checks the remaining ways tasks can leak: manifest entries that differ
    from the hash assignment, traces whose task is unknown, and domain mismatches.
    """
    problems: list[str] = []
    for task_id, split in manifest.task_splits.items():
        if split == "ood":
            continue
        expected = assign_split(task_id, manifest.salt, manifest.fractions)
        if split != expected:
            problems.append(f"task {task_id!r}: manifest says {split!r}, hash says {expected!r}")
    for trace in traces:
        try:
            trace_split(trace, manifest)
        except LeakageError as e:
            problems.append(str(e))
    if problems:
        shown = "\n".join(problems[:20])
        raise LeakageError(f"{len(problems)} split problem(s):\n{shown}")
    logger.info("No leakage: %d tasks checked", len(manifest.task_splits))


# --- manifest io -----------------------------------------------------------------------------


def save_manifest(manifest: SplitManifest, path: str | Path) -> None:
    """Write the manifest as stable, diff-friendly JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    logger.info("Saved split manifest to %s (sha256 %s)", path, file_sha256(path))


def load_manifest(path: str | Path) -> SplitManifest:
    """Load a split manifest. Log its sha256 so runs can record the split version."""
    path = Path(path)
    manifest = SplitManifest.model_validate_json(path.read_text(encoding="utf-8"))
    logger.info("Loaded split manifest %s (sha256 %s)", path, file_sha256(path))
    return manifest


def file_sha256(path: str | Path) -> str:
    """Hex sha256 of a file's bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- loaders ---------------------------------------------------------------------------------


def load_split(traces: Iterable[Trace], manifest: SplitManifest, name: str) -> list[Trace]:
    """Return the traces of a trainable split (train, val or calib). Protected names raise."""
    if name not in TRAINABLE_SPLITS:
        raise ProtectedSplitError(
            f"split {name!r} is not trainable; protected data is read only by "
            f"{ALLOWED_PROTECTED_CALLER}"
        )
    return [t for t in traces if trace_split(t, manifest) == name]


def load_protected_split(
    traces: Iterable[Trace], manifest: SplitManifest, name: str
) -> list[Trace]:
    """Return the traces of test, heldout_model or ood. Only callable from the final eval."""
    _require_final_eval_caller()
    if name not in ("test", "heldout_model", "ood"):
        raise ValueError(f"{name!r} is not a protected trace split; use load_gold for gold")
    return [t for t in traces if trace_split(t, manifest) == name]


def load_gold(gold_dir: str | Path = DEFAULT_GOLD_DIR) -> list[Trace]:
    """Verify the gold directory against its sha256 manifest, then load its traces.

    Only callable from the final eval. Gold labels get their own loader in phase 2.
    """
    _require_final_eval_caller()
    gold_dir = Path(gold_dir)
    verify_gold_manifest(gold_dir)
    return load_traces(gold_dir / GOLD_TRACES_NAME)


def verify_gold_manifest(gold_dir: str | Path) -> None:
    """Check every file in `gold_dir` against MANIFEST.sha256 (sha256sum format).

    Fails on a missing manifest, a malformed line, or any missing, extra or changed file.
    """
    gold_dir = Path(gold_dir)
    manifest_file = gold_dir / GOLD_MANIFEST_NAME
    if not manifest_file.is_file():
        raise GoldIntegrityError(f"{manifest_file} not found")

    expected: dict[str, str] = {}
    for line_no, line in enumerate(manifest_file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        digest, _, rest = line.partition(" ")
        if len(digest) != 64 or not rest or rest[0] not in " *":
            raise GoldIntegrityError(f"{manifest_file}:{line_no}: malformed line")
        expected[rest[1:]] = digest.lower()

    actual = {
        p.relative_to(gold_dir).as_posix(): file_sha256(p)
        for p in sorted(gold_dir.rglob("*"))
        if p.is_file() and p != manifest_file
    }
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    changed = sorted(k for k in expected.keys() & actual.keys() if expected[k] != actual[k])
    if missing or extra or changed:
        raise GoldIntegrityError(
            f"gold data does not match {GOLD_MANIFEST_NAME}: "
            f"missing={missing} extra={extra} changed={changed}"
        )
    logger.info("Gold manifest verified: %d files", len(actual))


def _require_final_eval_caller() -> None:
    """Raise unless the public loader that called this was called from the final eval."""
    frame = inspect.currentframe()
    try:
        caller = frame.f_back.f_back if frame and frame.f_back else None
        module = caller.f_globals.get("__name__") if caller else None
    finally:
        del frame
    if module != ALLOWED_PROTECTED_CALLER:
        raise ProtectedSplitError(
            f"protected data requested from {module!r}; only {ALLOWED_PROTECTED_CALLER} may "
            "read test, gold, heldout_model or ood data"
        )


# --- CLI -------------------------------------------------------------------------------------

app = typer.Typer(help="Build and check task-level split manifests.", no_args_is_help=True)

ConfigOpt = Annotated[Path, typer.Option("--config", help="Split config YAML.")]
TracesOpt = Annotated[list[Path], typer.Option("--traces", help="Trace JSONL file(s).")]


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command()
def make(
    traces: TracesOpt,
    config: ConfigOpt = Path("configs/splits.yaml"),
    update: Annotated[bool, typer.Option(help="Add new tasks to the existing manifest.")] = False,
    force: Annotated[bool, typer.Option(help="Overwrite an existing manifest.")] = False,
) -> None:
    """Build the split manifest from traces. Refuses to overwrite a frozen manifest."""
    cfg = SplitConfig.from_yaml(config)
    existing = None
    if cfg.manifest_path.exists():
        if update:
            existing = load_manifest(cfg.manifest_path)
        elif not force:
            logger.error("%s exists; use --update to add tasks", cfg.manifest_path)
            raise typer.Exit(code=1)
    all_traces = [t for path in traces for t in load_traces(path)]
    manifest = build_manifest(all_traces, cfg, existing)
    check_no_leakage(manifest, all_traces)
    save_manifest(manifest, cfg.manifest_path)
    _log_summary(manifest)


@app.command()
def check(traces: TracesOpt, config: ConfigOpt = Path("configs/splits.yaml")) -> None:
    """Re-run the leakage checks for the manifest named in the config."""
    cfg = SplitConfig.from_yaml(config)
    manifest = load_manifest(cfg.manifest_path)
    check_no_leakage(manifest, [t for path in traces for t in load_traces(path)])
    _log_summary(manifest)


def _log_summary(manifest: SplitManifest) -> None:
    """Log realised task fractions against targets, and trace counts per split."""
    task_counts = Counter(manifest.task_splits.values())
    n_in_domain = sum(n for s, n in task_counts.items() if s != "ood")
    for name in TASK_SPLITS:
        share = task_counts[name] / n_in_domain if n_in_domain else 0.0
        logger.info(
            "%-6s tasks=%5d (%.1f%%, target %.1f%%)",
            name,
            task_counts[name],
            100 * share,
            100 * manifest.fractions[name],
        )
    logger.info("ood    tasks=%5d", task_counts["ood"])
    logger.info("trace counts: %s", manifest.trace_counts)


if __name__ == "__main__":
    app()
