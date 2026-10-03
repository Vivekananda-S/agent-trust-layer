"""Tests for task-level splits and the guarded protected-data loaders."""

import json
from pathlib import Path
from typing import Any

import pytest

from atl.data import splits
from atl.data.splits import (
    GoldIntegrityError,
    LeakageError,
    ProtectedSplitError,
    SplitConfig,
    assign_split,
    build_manifest,
    check_no_leakage,
    file_sha256,
    load_gold,
    load_manifest,
    load_protected_split,
    load_split,
    save_manifest,
    trace_split,
    verify_gold_manifest,
)
from atl.traces.schema import Trace, save_traces

FIXTURE = Path(__file__).parent / "fixtures" / "trace_minimal.json"
FRACTIONS = {"train": 0.65, "val": 0.10, "calib": 0.10, "test": 0.15}
SALT = "test-salt"

# Hand-checked: u = first 8 bytes of sha256("test-salt:<id>") / 2**64, cut at .65/.75/.85.
EXPECTED = {
    "retail_task_000": "train",  # u = 0.3478
    "retail_task_003": "train",  # u = 0.6475, just under the 0.65 cut
    "retail_task_015": "val",  # u = 0.7339
    "retail_task_006": "calib",  # u = 0.7622
    "retail_task_007": "test",  # u = 0.9911
}


@pytest.fixture
def cfg() -> SplitConfig:
    return SplitConfig(
        version=1,
        salt=SALT,
        fractions=FRACTIONS,
        heldout_model="model-c",
        primary_domain="retail",
    )


def make_trace(
    trace_id: str, task_id: str, agent_model: str = "model-a", domain: str = "retail"
) -> Trace:
    raw: dict[str, Any] = json.loads(FIXTURE.read_text())
    raw.update(trace_id=trace_id, task_id=task_id, agent_model=agent_model)
    raw["meta"]["domain"] = domain
    return Trace.model_validate(raw)


@pytest.fixture
def traces() -> list[Trace]:
    out = [make_trace(f"t_{tid}", tid) for tid in EXPECTED]
    out += [
        make_trace("t_held_test", "retail_task_007", agent_model="model-c"),  # test task
        make_trace("t_held_train", "retail_task_000", agent_model="model-c"),  # train task
        make_trace("t_ood", "fintech_task_001", domain="fintech"),
    ]
    return out


# --- config ----------------------------------------------------------------------------------


def test_repo_config_loads() -> None:
    cfg = SplitConfig.from_yaml(Path(__file__).parents[1] / "configs" / "splits.yaml")
    assert cfg.fractions == FRACTIONS


@pytest.mark.parametrize(
    "fractions",
    [
        {"train": 0.7, "val": 0.1, "calib": 0.1},  # missing key
        {"train": 0.7, "val": 0.1, "calib": 0.1, "test": 0.2},  # sums to 1.1
        {"train": 0.9, "val": 0.1, "calib": 0.0, "test": 0.0},  # zero fraction
    ],
)
def test_bad_fractions_rejected(fractions: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SplitConfig(version=1, salt="s", fractions=fractions, primary_domain="retail")


# --- assignment ------------------------------------------------------------------------------


@pytest.mark.parametrize(("task_id", "expected"), EXPECTED.items())
def test_assign_split_hand_checked(task_id: str, expected: str) -> None:
    assert assign_split(task_id, SALT, FRACTIONS) == expected


def test_assignment_depends_on_salt() -> None:
    ids = [f"task_{i}" for i in range(200)]
    a = [assign_split(t, "salt-a", FRACTIONS) for t in ids]
    b = [assign_split(t, "salt-b", FRACTIONS) for t in ids]
    assert a != b


def test_realised_fractions_close_to_target() -> None:
    n = 20_000
    counts = {name: 0 for name in FRACTIONS}
    for i in range(n):
        counts[assign_split(f"task_{i}", SALT, FRACTIONS)] += 1
    for name, target in FRACTIONS.items():
        assert abs(counts[name] / n - target) < 0.01, name


def test_adding_tasks_never_moves_existing_ones(cfg: SplitConfig) -> None:
    old = [make_trace(f"t{i}", f"task_{i}") for i in range(300)]
    before = build_manifest(old, cfg)
    new = old + [make_trace(f"n{i}", f"new_task_{i}") for i in range(1000)]
    after = build_manifest(new, cfg, existing=before)
    assert all(after.task_splits[t] == s for t, s in before.task_splits.items())
    assert len(after.task_splits) == 1300


def test_update_with_changed_config_rejected(cfg: SplitConfig, traces: list[Trace]) -> None:
    manifest = build_manifest(traces, cfg)
    changed = cfg.model_copy(update={"salt": "other"})
    with pytest.raises(ValueError, match="does not match existing manifest"):
        build_manifest(traces, changed, existing=manifest)


# --- trace routing ---------------------------------------------------------------------------


def test_trace_routing(cfg: SplitConfig, traces: list[Trace]) -> None:
    manifest = build_manifest(traces, cfg)
    routed = {t.trace_id: trace_split(t, manifest) for t in traces}
    for task_id, split in EXPECTED.items():
        assert routed[f"t_{task_id}"] == split
    assert routed["t_held_test"] == "heldout_model"
    assert routed["t_held_train"] is None  # held-out model on a train task: dropped
    assert routed["t_ood"] == "ood"
    assert manifest.task_splits["fintech_task_001"] == "ood"
    assert manifest.trace_counts == {
        "calib": 1,
        "dropped": 1,
        "heldout_model": 1,
        "ood": 1,
        "test": 1,
        "train": 2,
        "val": 1,
    }


def test_trainable_splits_exclude_heldout_and_ood(cfg: SplitConfig, traces: list[Trace]) -> None:
    manifest = build_manifest(traces, cfg)
    for name in ("train", "val", "calib"):
        for t in load_split(traces, manifest, name):
            assert t.agent_model != "model-c"
            assert t.meta.domain == "retail"


def test_task_in_two_domains_rejected(cfg: SplitConfig) -> None:
    traces = [make_trace("a", "task_x"), make_trace("b", "task_x", domain="fintech")]
    with pytest.raises(LeakageError, match="more than one domain"):
        build_manifest(traces, cfg)


# --- leakage checks --------------------------------------------------------------------------


def test_clean_manifest_passes(cfg: SplitConfig, traces: list[Trace]) -> None:
    check_no_leakage(build_manifest(traces, cfg), traces)


def test_hand_edited_manifest_detected(cfg: SplitConfig, traces: list[Trace]) -> None:
    manifest = build_manifest(traces, cfg)
    manifest.task_splits["retail_task_007"] = "train"  # a test task moved into train
    with pytest.raises(LeakageError, match="hash says 'test'"):
        check_no_leakage(manifest, traces)


def test_unknown_task_detected(cfg: SplitConfig, traces: list[Trace]) -> None:
    manifest = build_manifest(traces, cfg)
    with pytest.raises(LeakageError, match="not in the split manifest"):
        check_no_leakage(manifest, [*traces, make_trace("t_new", "unseen_task")])


def test_domain_mismatch_detected(cfg: SplitConfig, traces: list[Trace]) -> None:
    manifest = build_manifest(traces, cfg)
    bad = make_trace("t_bad", "retail_task_000", domain="fintech")
    with pytest.raises(LeakageError, match="disagrees"):
        check_no_leakage(manifest, [bad])


def test_manifest_round_trip(cfg: SplitConfig, traces: list[Trace], tmp_path: Path) -> None:
    manifest = build_manifest(traces, cfg)
    path = tmp_path / "splits.json"
    save_manifest(manifest, path)
    assert load_manifest(path) == manifest
    first = file_sha256(path)
    save_manifest(manifest, path)
    assert file_sha256(path) == first  # stable bytes, so the hash identifies the split


# --- protected-data guards -------------------------------------------------------------------


@pytest.mark.parametrize("name", ["test", "gold", "heldout_model", "ood", "anything"])
def test_load_split_refuses_protected(cfg: SplitConfig, traces: list[Trace], name: str) -> None:
    manifest = build_manifest(traces, cfg)
    with pytest.raises(ProtectedSplitError):
        load_split(traces, manifest, name)


def test_protected_split_refused_outside_final_eval(cfg: SplitConfig, traces: list[Trace]) -> None:
    manifest = build_manifest(traces, cfg)
    with pytest.raises(ProtectedSplitError, match="only atl.eval.final_eval"):
        load_protected_split(traces, manifest, "test")


def test_protected_split_allowed_for_final_eval(
    cfg: SplitConfig, traces: list[Trace], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(splits, "ALLOWED_PROTECTED_CALLER", __name__)
    manifest = build_manifest(traces, cfg)
    got = load_protected_split(traces, manifest, "heldout_model")
    assert [t.trace_id for t in got] == ["t_held_test"]


def _write_gold(gold_dir: Path, traces: list[Trace]) -> None:
    save_traces(traces, gold_dir / "traces.jsonl")
    digest = file_sha256(gold_dir / "traces.jsonl")
    (gold_dir / "MANIFEST.sha256").write_text(f"{digest}  traces.jsonl\n")


def test_gold_refused_outside_final_eval(tmp_path: Path, traces: list[Trace]) -> None:
    _write_gold(tmp_path, traces)
    with pytest.raises(ProtectedSplitError):
        load_gold(tmp_path)


def test_gold_loads_when_intact(
    tmp_path: Path, traces: list[Trace], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(splits, "ALLOWED_PROTECTED_CALLER", __name__)
    _write_gold(tmp_path, traces)
    assert load_gold(tmp_path) == traces


def test_gold_tampered_file_detected(tmp_path: Path, traces: list[Trace]) -> None:
    _write_gold(tmp_path, traces)
    with (tmp_path / "traces.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(GoldIntegrityError, match=r"changed=\['traces.jsonl'\]"):
        verify_gold_manifest(tmp_path)


def test_gold_missing_file_detected(tmp_path: Path, traces: list[Trace]) -> None:
    _write_gold(tmp_path, traces)
    (tmp_path / "traces.jsonl").unlink()
    with pytest.raises(GoldIntegrityError, match=r"missing=\['traces.jsonl'\]"):
        verify_gold_manifest(tmp_path)


def test_gold_extra_file_detected(tmp_path: Path, traces: list[Trace]) -> None:
    _write_gold(tmp_path, traces)
    (tmp_path / "extra.jsonl").write_text("{}\n")
    with pytest.raises(GoldIntegrityError, match=r"extra=\['extra.jsonl'\]"):
        verify_gold_manifest(tmp_path)


def test_gold_without_manifest_detected(tmp_path: Path) -> None:
    with pytest.raises(GoldIntegrityError, match="not found"):
        verify_gold_manifest(tmp_path)
