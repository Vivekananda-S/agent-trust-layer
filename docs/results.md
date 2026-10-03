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

### 2026-10-03 — System under test: tau2-bench (τ³-bench)

- tau2-bench (MIT), pinned to commit `5bfa7e3`. The original tau-bench repo is superseded and
  its tasks are not updated. In-domain: retail (114 tasks) + airline (50); banking_knowledge
  (97) is the out-of-domain set instead of building our own fintech agent.
- Project moved to Python 3.12 (tau2 requires it; Colab runs it). tau2 is an optional extra so
  training installs never pull it, and only `src/atl/agent/` may import it (static test).
- **`env_outcome` comes from tau2's ENV evaluation only (final DB state).** Retail's default
  reward also includes LLM-judged "NL assertions" (GPT-4.1 by default). Using them would make
  the "objective" label partly an LLM opinion, cost money, and need an OpenAI key. The DB check
  is deterministic and free. Trade-off: success means "DB ended in the right state", not
  "communicated well".
- **The judge sees what a production agent sees.** `task` is the user's first message, not
  tau2's hidden user-scenario instructions (privileged information a real judge never has).
  User-simulator control tokens (`###STOP###`, `###TRANSFER###`, `###OUT-OF-SCOPE###`) are
  stripped: real users never send them and two of them correlate with the outcome.
- Runs ending in `user_error`, `infrastructure_error` or `unexpected_error` are excluded and
  counted (simulator or infrastructure, not agent behaviour). `agent_error` runs are kept: in
  tau2 that means the agent broke the communication protocol, which is a real failure.
- Every LLM call goes through our disk cache (sha256 of model + messages + tools + sampling
  params + a per-(run, task, trial) namespace). The namespace matters: without it, trial 2 of
  a task would replay trial 1 from the cache and "repeated trials" would be copies.
- Cost log records **nominal** cost at paid-tier list prices from the config. On the Gemini
  free tier real spend is $0; the nominal figure is the projection for scaling up. The budget
  cap is read back from the log on start-up, so it holds across sessions.
- Model choice for the pilot (cheapest credible): agent `gemini-3.8-flash`, user simulator
  `gemini-3.5-flash-lite` (fixed for all runs), both on the Google AI Studio free tier.
- tau2 packaging quirks: its core import needs `websockets` (listed only in its voice extra),
  and its wheel ships no data, so `atl-agent fetch-data` clones the pinned commit into
  `external/` and the runner points `TAU2_DATA_DIR` at it.

## Findings

### 2026-10-03 — Hash splits are noisy with few tasks

Synthetic smoke run with 400 tasks: realised task shares were train 70.5%, val 8.8%,
calib 9.0%, test 11.8% against targets 65/10/10/15. With n = 400 the binomial standard error
of a 15% share is about 1.8 points, so a 3-point shortfall is ordinary sampling noise, not a
bug. Implication for the real data (tau-bench retail has only a few hundred tasks): check the
realised counts when the manifest is first built; if the test pool is too small for a 400–600
trace gold set, change the salt once, before freezing, and record it here. Never after.
