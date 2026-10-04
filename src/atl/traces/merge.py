"""Merge runner outputs collected on other machines (Kaggle, Colab) into the local dataset.

Collection runs on several platforms, each with its own storage. A session's `traces.zip`
(`traces/<run>/raw.jsonl`, `failed_runs.jsonl`, logs, its own `costs.jsonl`) is merged here:

- runs are keyed by (task id, trial); a run already present locally is never duplicated or
  overwritten, so importing the same zip twice is harmless;
- `traces.jsonl` and `review_flags.jsonl` are regenerated from the merged raw file;
- the zip's cost log and logs are kept separately under `data/imported/<zip name>/`, never mixed
  into the local `costs.jsonl` (that file drives the local budget cap).
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

from atl.traces.adapters.tau2 import convert_raw_file, review_flags
from atl.traces.schema import save_traces

logger = logging.getLogger(__name__)


def run_key(record: dict) -> tuple[str, int]:
    """Identity of one run in a raw file: (task id, trial)."""
    return record["simulation"]["task_id"], record["run_info"]["trial"]


def merge_raw(target: Path, incoming: Path) -> tuple[int, int]:
    """Append runs from `incoming` that `target` lacks. Returns (added, already present)."""
    existing: set[tuple[str, int]] = set()
    if target.exists():
        with target.open(encoding="utf-8") as f:
            existing = {run_key(json.loads(line)) for line in f if line.strip()}
    added = dup = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with incoming.open(encoding="utf-8") as src, target.open("a", encoding="utf-8") as out:
        for line in src:
            if not line.strip():
                continue
            key = run_key(json.loads(line))
            if key in existing:
                dup += 1
                continue
            existing.add(key)
            out.write(line if line.endswith("\n") else line + "\n")
            added += 1
    return added, dup


def import_zip(zip_path: Path, traces_dir: Path, imported_dir: Path) -> dict[str, tuple[int, int]]:
    """Merge every run directory in a session zip. Returns {run name: (added, already present)}."""
    summary: dict[str, tuple[int, int]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(zip_path).extractall(tmp)
        root = Path(tmp)
        for raw in sorted(root.glob("**/traces/*/raw.jsonl")):
            run_dir = traces_dir / raw.parent.name
            summary[raw.parent.name] = merge_raw(run_dir / "raw.jsonl", raw)
            failed = raw.parent / "failed_runs.jsonl"
            if failed.exists():
                with (run_dir / "failed_runs.jsonl").open("a", encoding="utf-8") as out:
                    out.write(failed.read_text(encoding="utf-8"))
            _regenerate(run_dir)
        keep = imported_dir / zip_path.stem
        keep.mkdir(parents=True, exist_ok=True)
        for extra in ("costs.jsonl", "logs"):
            src = next(root.glob(f"**/{extra}"), None)
            if src is not None and src.is_file():
                shutil.copy2(src, keep / src.name)
            elif src is not None:
                shutil.copytree(src, keep / src.name, dirs_exist_ok=True)
    return summary


def _regenerate(run_dir: Path) -> None:
    """Rebuild traces.jsonl and review_flags.jsonl from the merged raw file."""
    raw = run_dir / "raw.jsonl"
    traces, _ = convert_raw_file(raw)
    save_traces(traces, run_dir / "traces.jsonl")
    rows = []
    with raw.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sim = json.loads(line)["simulation"]
                if flags := review_flags(sim):
                    rows.append(
                        json.dumps({"trace_id": f"tau2-{sim['id']}", "flags": flags}) + "\n"
                    )
    (run_dir / "review_flags.jsonl").write_text("".join(rows), encoding="utf-8")
