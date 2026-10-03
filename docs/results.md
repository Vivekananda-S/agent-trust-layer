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

### 2026-10-03 — First live pilot (retail, gemini-3.8-flash agent, 3.5-flash-lite user)

- **Free tier is not a data source.** gemini-3.8-flash allows 20 requests/day/project on the
  free tier (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`); a trace needs ~10 agent
  calls, so ~2 traces/day. The run stopped cleanly after 3 consecutive quota errors, as designed.
- **Measured cost:** one retail trace used ~54k agent input + ~2k output tokens (my estimate
  was ~170k), nominal $0.045 agent + $0.001 user simulator. Projection per 1,000 traces at list
  prices: ~$48 on 3.8-flash, ~$21 on 3.5-flash-lite, ~$6 on 2.5-flash-lite, ~$1 on Llama 3.1 8B.
  Upper bounds: Gemini's implicit prompt caching discount is not yet credited.
- ~~Weak user simulator produces false failures.~~ **Corrected below**: concluded too fast
  from one example.
- Gemini 3 multi-turn tool calling works through tau2: LiteLLM carries the thought signature
  inside the tool-call id, which tau2 preserves. The adapter strips it from traces.

### 2026-10-03 — Paid pilot (retail, 3.8-flash agent and user simulator), stopped at 5 traces

- Stopped by the AI Studio project's own monthly spend cap ("Your project has exceeded its
  monthly spending cap") after ~$0.37 of real spend. Google's cap and our `budget_usd` are two
  independent stops; ours never fired because it was set higher.
- 5 retail traces: 4 succeeded (tasks 49, 97, 113, 53), 1 failed (108). Cost per trace
  $0.04–0.10, mean ~$0.07 including the user simulator. Thinking tokens are already inside
  `completion_tokens` (verified: 275 = 222 reasoning + 53 text), so the cost log is accurate.
  No implicit-cache discount was reported (`cached_tokens` null).
- **Correction — task 108 fails for a task-definition reason, not a simulator one.** With the
  strong user simulator the user did *not* confirm-and-stop; it asked only for the refund
  amount ("You want to know how much money you can get back") and left. The agent correctly
  asked for confirmation before acting, as the policy requires. But the task's expected
  actions include `return_delivered_order_items`, so the DB check fails either way. Same
  failure under two different user simulators; not in tau2's `task_issues` list.
- **Implication for labelling: `env_outcome` is not ground truth.** A benchmark's own
  success signal can be wrong when the task is under-specified. Phase 2 human labels must be
  made without looking at `env_outcome`, and env-vs-human disagreements become a measured
  quantity (and a review queue), not something to silently trust.
- A strong agent with clean tools succeeds ~80% here: too few failures for a judge to learn
  from. Failure diversity has to come from weaker agent models, prompt variants and injected
  faults, as planned.

### 2026-10-03 — 20-trace pilot complete (gemini-3.8-flash agent and user simulator)

| Domain | Traces | Env success | Cost/trace mean (max) | Steps median (max) | Agent input tokens median |
| --- | --- | --- | --- | --- | --- |
| retail | 10 | 9/10 | $0.060 ($0.103) | 21 (30) | 59.8k |
| airline | 10 | 9/10 | $0.077 ($0.174) | 22 (50) | 68.7k |

- All 20 traces pass strict schema validation (tool pairing included). Total spend $1.68.
- Resume worked across the spend-cap interruption: finished runs were skipped, and the calls
  made just before the cap error came back from the cache (4 cached, $0 re-spend).
- Both failures were flagged for review, for different reasons:
  - retail 108: under-specified task (see the correction above).
  - airline 32: the agent priced the upgrade at $396 for 3 passengers, said it exceeded the
    user's $100 limit, and the user left; the task expects a two-step change within budget.
    Either a genuine agent failure (wrong price reasoning: F4 / F7 candidate) or a strict
    task — a judgement for the phase 2 labelling, not something to infer from `env_outcome`.
- No tool errors occurred: with clean tools a strong agent rarely fails (90%). The F5 (ignored
  tool error) and F8 (injection) classes cannot appear without injected faults.
