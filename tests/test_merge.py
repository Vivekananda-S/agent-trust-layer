"""Tests for merging session zips from other machines into the local dataset."""

import copy
import json
import zipfile
from pathlib import Path
from typing import Any

from atl.traces.merge import import_zip, merge_raw

FIXTURE = Path(__file__).parent / "fixtures" / "tau2_sim_small.json"
INFO = {
    "domain": "retail",
    "agent_model": "ollama_chat/qwen3:8b",
    "user_model": "gemini/gemma-4-31b-it",
    "prompt_variant": "tau2_default",
    "user_variant": "tau2_default",
    "fault": None,
    "trial": 0,
    "seed": 1000,
}


def record(task_id: str, trial: int, sim_id: str) -> dict[str, Any]:
    sim = copy.deepcopy(json.loads(FIXTURE.read_text()))
    sim.update(id=sim_id, task_id=task_id)
    return {"run_info": {**INFO, "trial": trial}, "simulation": sim, "faults": []}


def write(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_merge_raw_skips_runs_already_present(tmp_path: Path) -> None:
    target, incoming = tmp_path / "local.jsonl", tmp_path / "in.jsonl"
    write(target, [record("1", 0, "a")])
    write(incoming, [record("1", 0, "a-again"), record("2", 0, "b"), record("1", 1, "c")])
    assert merge_raw(target, incoming) == (2, 1)
    assert merge_raw(target, incoming) == (0, 3)  # idempotent
    assert len(target.read_text().splitlines()) == 3


def test_import_zip(tmp_path: Path) -> None:
    session = tmp_path / "session"
    write(session / "traces/collect_x/raw.jsonl", [record("1", 0, "a"), record("2", 0, "b")])
    (session / "traces/collect_x/failed_runs.jsonl").write_text('{"task_id": "3", "trial": 0}\n')
    (session / "costs.jsonl").write_text('{"cost_usd": 0.0}\n')
    (session / "logs").mkdir()
    (session / "logs/collect_x.log").write_text("log line\n")
    zip_path = tmp_path / "traces.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for p in session.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(session))

    traces_dir = tmp_path / "data/traces"
    write(traces_dir / "collect_x/raw.jsonl", [record("1", 0, "a")])  # already have one run
    local_costs = tmp_path / "data/costs.jsonl"
    local_costs.write_text('{"cost_usd": 1.5}\n')

    summary = import_zip(zip_path, traces_dir, tmp_path / "data/imported")
    assert summary == {"collect_x": (1, 1)}
    run_dir = traces_dir / "collect_x"
    assert len((run_dir / "traces.jsonl").read_text().splitlines()) == 2  # regenerated
    assert (run_dir / "failed_runs.jsonl").read_text().strip() == '{"task_id": "3", "trial": 0}'
    assert local_costs.read_text() == '{"cost_usd": 1.5}\n'  # local budget log untouched
    kept = tmp_path / "data/imported/traces"
    assert (kept / "costs.jsonl").exists() and (kept / "logs/collect_x.log").exists()
