# Results, findings and decisions

A running log of decisions, findings and failures. Newest entries at the bottom of each section.

## Decisions

### 2026-10-03 — Trace schema v1.0

- `task_id` is a required field, because splits are by task and must not depend on parsing ids.
- `meta` (domain, user variant, injected fault, seed, created_at) is required. It drives
  stratified labelling and the drift / out-of-domain slices.
- Labels are not part of `Trace`; they live in separate records keyed by `trace_id`, so the
  judge's input type cannot carry labels by construction.
- Tool call / result pairing is strict: each result must answer an earlier open call (by
  `call_id`, else FIFO by tool name). Catches collector bugs at load time.
- `env_outcome` and `meta.fault` are effectively labels. The serialiser must exclude them
  (test to be added with `judge/serialise.py`).

### 2026-10-03 — Task-level splits v1

- Each in-domain task is assigned by a salted sha256 hash of its `task_id` (65/10/10/15 for
  train/val/calib/test). Hashing, unlike a seeded shuffle, keeps every existing assignment
  fixed when new tasks are added, so later data versions cannot move test tasks into training.
- The task -> split manifest is committed under `splits/` (task ids only, no data) and is the
  source of truth; `check_no_leakage` re-derives every assignment from the hash and fails on
  hand edits.
- Held-out agent model traces are kept only on test tasks (`heldout_model` split); on other
  tasks they are dropped. Otherwise the drift test would mix "new model" with "task seen in
  training" and the two effects could not be separated.
- Non-primary-domain traces (fintech) form the `ood` split.
- Protected data (`test`, `gold`, `heldout_model`, `ood`) is readable only from
  `atl.eval.final_eval` (runtime caller check), gold files are verified against
  `MANIFEST.sha256` before loading, and a static test fails CI if any other file mentions the
  protected loaders or `data/gold`.

## Findings

### 2026-10-03 — Hash splits are noisy with few tasks

Synthetic smoke run with 400 tasks: realised task shares were train 70.5%, val 8.8%,
calib 9.0%, test 11.8% against targets 65/10/10/15. With n = 400 the binomial standard error
of a 15% share is about 1.8 points, so a 3-point shortfall is ordinary sampling noise, not a
bug. Implication for the real data (tau-bench retail has only a few hundred tasks): check the
realised counts when the manifest is first built; if the test pool is too small for a 400–600
trace gold set, change the salt once, before freezing, and record it here. Never after.
