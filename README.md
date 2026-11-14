# Agent Trust Layer

A small, calibrated judge model that reads an AI agent's execution trace and predicts whether the
agent failed and which failure classes (F1–F8) occurred. See [docs/plan.md](docs/plan.md).

## Setup

Python 3.12. `requirements.txt` pins the tested versions and works on Ubuntu and Windows.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt   # or: pip install -r requirements.txt
atl-agent fetch-data                 # pinned tau2-bench checkout into external/ (policies, tasks)
pytest
```

Copy `.env.example` to `.env` and fill it in. GPU training dependencies (the `train` extra) are
installed on Colab, not here.

## Moving to a new machine

None of these items is in git, so carry each one over by hand:

1. **`.env`** holds the secrets (`ATL_SERVING_URL`, `ATL_SERVING_KEY`, `HF_TOKEN`, `ATL_HF_DATASET`,
   `GEMINI_API_KEY`). Copy the file over privately; never through git. If it is lost, regenerate
   each key from its provider's dashboard. The serving key must match the Modal secret
   `atl-serving`.
2. **Push all commits** on the old machine (`git status -sb` must not say "ahead").
3. **Back up `data/`** on the old machine (see below), then restore it on the new one.
4. **Modal login:** `modal token new`. Hugging Face uses `HF_TOKEN` from `.env`.
5. **Git identity:** `git config user.name` and `git config user.email`.
6. **`external/tau2-bench`:** re-create it with `atl-agent fetch-data`.

## Data and backups

`data/` is gitignored, so nothing in it reaches GitHub. In November 2026 a laptop OS failure
destroyed the only copy of the first Modal collection, so every dataset now also lives in a
**private Hugging Face dataset, `vivekanandashonti/atl-data`** (set `ATL_HF_DATASET` and `HF_TOKEN`
in `.env`).

| What | Where | Backed up |
|---|---|---|
| Traces (`data/traces/*/traces.jsonl`), plus each run's raw output, failed runs and review flags | HF `traces/` | Yes |
| Session logs, GPU ledger (`modal_ledger.jsonl`), API cost log (`costs.jsonl`) | HF `logs/` and the repo root | Yes |
| Labelling queue (`data/labels/seed_queue_v1.jsonl`) | HF `labels/` | Yes |
| Hand labels (`data/labels/human.jsonl`) | HF `labels/` | **Yes, after every labelling session.** These cost hours of human time, so this is the most valuable file. |
| Gold set (`data/gold/`, once frozen) | HF `gold/` | Yes, together with its `MANIFEST.sha256` |
| Split manifest, configs, code, docs | GitHub | Yes, through git |
| LLM response cache (`data/llm_cache/`), smoke-test output (`data/smoke/`) | Local only | No. These are not needed to rebuild the dataset. |

**When backups happen:**

- **Modal sessions:** `scripts/modal_session.sh` uploads `data/` every 30 minutes and again on
  exit, so a crash loses at most 30 minutes of traces.
- **Labelling, or any other change to `data/`:** run the upload by hand. Leaving out the excluded
  folders, the upload is:

  ```bash
  hf upload "$ATL_HF_DATASET" data . --repo-type dataset --private \
    --exclude "llm_cache/*" --exclude "smoke/*"
  ```

- **Second copy (recommended, manual):** once a week, zip `data/` and copy it into the Google Drive
  folder `agent-trust-layer/`.

**Restoring on a new machine:**

```bash
hf download "$ATL_HF_DATASET" --repo-type dataset --local-dir data
# check the traces still match splits/splits_v1.json (one --traces per file)
atl-splits check $(for f in data/traces/*/traces.jsonl; do echo --traces "$f"; done)
```

After restoring, compare file sizes with the HF listing before you trust the copy. The audit in
`docs/results.md` (2026-11-14) is an example of that check.
