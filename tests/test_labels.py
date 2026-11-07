"""Tests for label records, the seed sampler and the trace views (not the Streamlit UI)."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from atl.data.splits import SplitConfig, build_manifest, save_manifest
from atl.labels.render import policy_text, trace_views
from atl.labels.sample import (
    SeedConfig,
    build_seed_queue,
    fault_group,
    load_queue,
    queued_traces,
    save_queue,
    stratum_of,
)
from atl.labels.schema import Label, append_label, load_labels
from atl.traces.schema import Trace, save_traces

FIXTURE = Path(__file__).parent / "fixtures" / "trace_minimal.json"


def make_trace(
    trace_id: str,
    task_id: str,
    fault: str | None = None,
    success: bool = True,
    agent_model: str = "model-a",
) -> Trace:
    raw: dict[str, Any] = json.loads(FIXTURE.read_text())
    raw.update(trace_id=trace_id, task_id=task_id, agent_model=agent_model)
    raw["meta"]["fault"] = fault
    raw["env_outcome"] = {"task_success": success}
    return Trace.model_validate(raw)


def label(**kw: Any) -> Label:
    base: dict[str, Any] = dict(
        trace_id="t1", failed=False, confidence="high", annotator="human", taxonomy_version="0.1"
    )
    return Label(**(base | kw))


# --- label schema ----------------------------------------------------------------------------


def test_failed_label_needs_class_and_step() -> None:
    ok = label(failed=True, classes=["F5", "F2", "F5"], first_failure_step=3)
    assert ok.classes == ["F2", "F5"]  # sorted, de-duplicated
    with pytest.raises(ValidationError, match="at least one class"):
        label(failed=True, first_failure_step=3)
    with pytest.raises(ValidationError, match="first_failure_step"):
        label(failed=True, classes=["F1"])


def test_other_failure_counts_as_a_failure() -> None:
    lab = label(failed=True, other_failure="  asked for the zip code twice ", first_failure_step=0)
    assert lab.other_failure == "asked for the zip code twice"


def test_passing_label_carries_no_failure_fields() -> None:
    with pytest.raises(ValidationError, match="need failed=True"):
        label(failed=False, classes=["F3"])
    with pytest.raises(ValidationError, match="need failed=True"):
        label(failed=False, first_failure_step=2)


def test_unjudgeable_label_needs_rationale() -> None:
    with pytest.raises(ValidationError, match="rationale"):
        label(failed=None)
    assert label(failed=None, rationale="user simulator looped").failed is None


def test_unknown_class_rejected() -> None:
    with pytest.raises(ValidationError):
        label(failed=True, classes=["F9"], first_failure_step=0)


def test_label_file_latest_wins(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "human.jsonl"
    assert load_labels(path) == {}
    append_label(label(trace_id="a"), path)
    append_label(label(trace_id="b"), path)
    append_label(label(trace_id="a", failed=True, classes=["F7"], first_failure_step=4), path)
    labels = load_labels(path)
    assert set(labels) == {"a", "b"}
    assert labels["a"].classes == ["F7"]
    assert len(path.read_text().splitlines()) == 3  # history kept


# --- sampler ---------------------------------------------------------------------------------


def test_fault_groups() -> None:
    assert fault_group(None) == "none"
    assert fault_group("timeout") == "other"
    assert fault_group("empty+injection") == "injection"
    assert stratum_of(make_trace("x", "t", "error", False)) == "model-a|retail|other|fail"


def _pool() -> list[Trace]:
    # 3 strata of different sizes over 10 tasks.
    out = [make_trace(f"clean{i}", f"task{i % 10}") for i in range(20)]
    out += [make_trace(f"inj{i}", f"task{i % 10}", "injection", False) for i in range(4)]
    out += [make_trace(f"err{i}", f"task{i % 10}", "error", False) for i in range(2)]
    return out


def test_queue_is_round_robin_over_strata() -> None:
    queue = build_seed_queue(_pool(), n=6, seed=0, max_per_task=10)
    # Every stratum appears in the first round, then again in the second.
    assert Counter(i.stratum for i in queue[:3]) == Counter(
        {
            "model-a|retail|none|pass": 1,
            "model-a|retail|injection|fail": 1,
            "model-a|retail|other|fail": 1,
        }
    )
    assert [i.position for i in queue] == list(range(6))
    assert len({i.trace_id for i in queue}) == 6


def test_queue_drains_small_strata_then_continues() -> None:
    queue = build_seed_queue(_pool(), n=12, seed=0, max_per_task=10)
    counts = Counter(i.stratum for i in queue)
    assert counts["model-a|retail|other|fail"] == 2  # all it has
    assert counts["model-a|retail|injection|fail"] == 4
    assert counts["model-a|retail|none|pass"] == 6


def test_queue_respects_max_per_task() -> None:
    traces = [make_trace(f"t{i}", "only_task") for i in range(5)]
    assert len(build_seed_queue(traces, n=5, seed=0, max_per_task=2)) == 2


def test_queue_is_deterministic() -> None:
    a = build_seed_queue(_pool(), n=10, seed=1, max_per_task=2)
    b = build_seed_queue(list(reversed(_pool())), n=10, seed=1, max_per_task=2)
    assert a == b
    assert a != build_seed_queue(_pool(), n=10, seed=2, max_per_task=2)


def _repo(tmp_path: Path) -> SeedConfig:
    """Traces on train and test tasks (plus held-out-model traces) and a manifest for them."""
    # Task ids hand-picked under salt "test-salt" (see tests/test_splits.py).
    traces = [make_trace(f"tr{i}", "retail_task_000") for i in range(3)]
    traces += [make_trace(f"te{i}", "retail_task_007") for i in range(3)]
    traces += [make_trace("held", "retail_task_007", agent_model="model-c")]
    save_traces(traces, tmp_path / "run" / "traces.jsonl")
    split_cfg = SplitConfig(
        version=1,
        salt="test-salt",
        fractions={"train": 0.65, "val": 0.10, "calib": 0.10, "test": 0.15},
        heldout_models=["model-c"],
        in_domains=["retail"],
    )
    save_manifest(build_manifest(traces, split_cfg), tmp_path / "splits.json")
    return SeedConfig(
        traces=[str(tmp_path / "*" / "traces.jsonl")],
        split_manifest=tmp_path / "splits.json",
        n=50,
        seed=0,
        max_per_task=50,
        queue_path=tmp_path / "queue.jsonl",
        labels_path=tmp_path / "labels.jsonl",
    )


def test_seed_queue_never_contains_protected_traces(tmp_path: Path) -> None:
    from atl.labels.sample import load_train_traces

    cfg = _repo(tmp_path)
    queue = build_seed_queue(load_train_traces(cfg), cfg.n, cfg.seed, cfg.max_per_task)
    assert {i.trace_id for i in queue} == {"tr0", "tr1", "tr2"}
    save_queue(queue, cfg.queue_path)
    assert load_queue(cfg.queue_path) == queue
    assert [t.trace_id for _, t in queued_traces(cfg)] == [i.trace_id for i in queue]


def test_queued_traces_rejects_a_tampered_queue(tmp_path: Path) -> None:
    cfg = _repo(tmp_path)
    cfg.queue_path.write_text(json.dumps({"position": 0, "trace_id": "te0", "stratum": "x"}) + "\n")
    with pytest.raises(ValueError, match="not train traces"):
        queued_traces(cfg)


def test_repo_seed_config_loads() -> None:
    cfg = SeedConfig.from_yaml(Path(__file__).parents[1] / "configs" / "labels" / "seed_v1.yaml")
    assert cfg.split_manifest == Path("splits/splits_v1.json")


# --- views -----------------------------------------------------------------------------------


def test_trace_views_number_steps_and_flag_errors() -> None:
    raw: dict[str, Any] = json.loads(FIXTURE.read_text())
    raw["steps"][3]["error"] = "timeout"
    views = trace_views(Trace.model_validate(raw))
    assert [v.index for v in views] == [0, 1, 2, 3, 4]
    assert views[2].title == "call get_order" and '"order_id": "4411"' in views[2].body
    assert views[3].is_error and views[3].body.startswith("ERROR: timeout")


def test_policy_text_reads_domain_policy(tmp_path: Path) -> None:
    policy = tmp_path / "tau2" / "domains" / "retail" / "policy.md"
    policy.parent.mkdir(parents=True)
    policy.write_text("# Retail policy")
    assert policy_text("retail", tmp_path) == "# Retail policy"
    with pytest.raises(FileNotFoundError):
        policy_text("airline", tmp_path)
