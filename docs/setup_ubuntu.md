# Setting up on Ubuntu

Step-by-step setup of this project on a fresh Ubuntu machine. It was written on 2026-11-14 for the
move from Windows 11 back to Ubuntu. A person or a Claude Code session can follow it from top to
bottom: each step says what to run, what you should see, and which steps the owner has to do by
hand.

**Claude Code: read this first.** Your memory of earlier sessions is stored on the old machine
(`~/.claude/projects/.../memory/`) and does not move with the repo. The facts you need are in
[Project state at the move](#project-state-at-the-move) below. Read `CLAUDE.md` too; its rules,
especially the gold test set rule, still apply.

## Before wiping the Windows machine (on Windows)

1. Commit and push everything: `git status -sb` must show `## main...origin/main` with no
   "ahead".
2. Back up `data/` to Hugging Face (Git Bash, from the repo root):
   ```bash
   set -a && . ./.env && set +a
   .venv/Scripts/hf upload "$ATL_HF_DATASET" data . --repo-type dataset --private \
     --exclude "llm_cache/*" --exclude "smoke/*"
   ```
   This matters most for **hand labels** (`data/labels/human.jsonl`), which exist nowhere else
   until they are uploaded.
3. Copy **`.env`** privately (a USB stick or a password manager). It holds every key, and it is
   gitignored. Never commit it, paste it into chat, or upload it.

## 1. System packages

```bash
sudo apt update && sudo apt install -y git curl
curl -LsSf https://astral.sh/uv/install.sh | sh     # installs uv, which also provides Python 3.12
exec $SHELL                                          # reload PATH so `uv` is found
```

## 2. Clone and install

```bash
git clone https://github.com/Vivekananda-S/agent-trust-layer.git
cd agent-trust-layer
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
git config user.name "Vivekananda Shonti"
git config user.email "vivekshonti3@gmail.com"
```

`requirements.txt` pins the versions tested on 2026-11-14 and resolves for Linux. It leaves out
GPU training packages, because those are installed on Colab.

## 3. Secrets (owner does this by hand)

Put the saved `.env` in the repo root. If it was lost, start from the template:
`cp .env.example .env`, then fill in:

| Key | Where it comes from |
|---|---|
| `HF_TOKEN` | huggingface.co, Settings, Access Tokens (write access) |
| `ATL_HF_DATASET` | `vivekanandashonti/atl-data` |
| `ATL_SERVING_URL` | `https://vivekshonti3--atl-serving-serve.modal.run` |
| `ATL_SERVING_KEY` | Must equal the Modal secret `atl-serving`. If it was lost, generate a new key and overwrite that secret (below). |
| `GEMINI_API_KEY` | Google AI Studio, free tier. Do not use a paid key. |
| `OPENROUTER_API_KEY`, `WANDB_API_KEY` | Optional. OpenRouter is not used (it is paid). |

To regenerate the serving key (only if it was lost):

```bash
k=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
modal secret create atl-serving "ATL_SERVING_KEY=$k" --force
echo "ATL_SERVING_KEY=$k" >> .env      # then delete any older ATL_SERVING_KEY line in .env
```

**Claude Code: don't write secrets or Modal secrets yourself.** Give the owner the commands to run.

## 4. Logins (owner does this by hand)

```bash
modal token new          # opens a browser; Modal workspace is `vivekshonti3`
```

Hugging Face reads `HF_TOKEN` from `.env`, so no separate login is needed.

The Modal secret `huggingface-secret` must contain a valid HF token, because Llama is a gated
model. It already lives on Modal and does not need to move.

## 5. tau2-bench data and restored data

```bash
atl-agent fetch-data      # clones tau2-bench at the pinned commit into external/ (policies, tasks)
set -a && . ./.env && set +a
hf download "$ATL_HF_DATASET" --repo-type dataset --local-dir data
```

## 6. Check everything

```bash
pytest -q                 # expect 194 passed (more if tests were added since)
ruff check src tests
sha256sum splits/splits_v1.json                                # must start with 414c19104ff97972
atl-splits check $(for f in data/traces/*/traces.jsonl; do echo --traces "$f"; done)
```

The last line of the check must read:

```
current trace counts: {'calib': 172, 'dropped': 8, 'heldout_model': 184, 'test': 409, 'train': 1155, 'val': 159}
```

That is 2,087 traces in 12 run folders under `data/traces/`. The line above it, "trace counts
when the manifest was built", shows smaller test and heldout_model numbers. That is expected,
because the manifest was frozen before the last collection. If the current counts are lower,
the restore is incomplete. Compare file sizes with the Hugging Face listing before doing
anything else.

Also check the labelling setup:

```bash
atl-label stats           # queue of 330; hand-label count so far
```

## 7. Ubuntu-specific notes

- **Long Modal sessions:** run `bash scripts/modal_session.sh <configs...>` inside `tmux` or with
  `nohup`, so closing the terminal does not kill it. Stop the laptop from sleeping while a session
  runs. The script finds `.venv/bin` by itself, uploads `data/` to HF every 30 minutes, and stops
  the Modal app on exit.
- **Windows-only details** (`.venv/Scripts`, `Start-Process`, `PYTHONUTF8=1`, CRLF warnings from
  git) no longer apply. `.gitattributes` keeps `*.sh` files at LF endings either way.
- **Labelling app:** run `atl-label app`. It opens in the browser at `http://localhost:8501`.

## Project state at the move

- **Phase 1 is done**, and its gate is met: 2,087 traces, and the splits are frozen in
  `splits/splits_v1.json` (salt `atl-splits-v67`, 23 test tasks). Llama 3.1 8B is the held-out
  model.
- **Phase 2 is in progress:**
  - Done so far: labelling guide v0.1 (`docs/labelling_guide.md`), the Streamlit app, and the
    330-trace seed queue from the train split.
  - Next, the owner hand-labels the first 100 traces, then the taxonomy review.
  - Claude has offered to build the LLM labeller in the meantime: the Gemini free tier with disk
    caching, plus per-class kappa.
- **Gold-path exception, approved 2026-11-07 and recorded in `CLAUDE.md`:** `atl.labels.gold` may
  read test and heldout_model traces so a person can verify gold labels. Nothing else may read
  them outside `final_eval`.
- **Budget:**
  - Modal gives a $30 monthly credit, and November's is spent: the ledger is at $28.47 in
    `data/modal_ledger.jsonl`, and Modal's dashboard figure has not been recorded yet.
  - The project uses no paid Gemini or OpenRouter.
- **Before any Modal session:**
  - run `atl-agent check --config <cfg>` (the session script does this);
  - for a model being served for the first time, also run
    `modal run src/atl/serving/modal_app.py::check_startup --model <name>`, which uses a cheap
    L40S.
  - Both checks were added after two failed starts.
- **Backups:**
  - **Data:** goes to the private HF dataset after every data change and every labelling
    session. A weekly zip of `data/` to Google Drive (`agent-trust-layer/`) is recommended as a
    second copy.
  - **Code:** push after every work session.
- **History:** the decisions, results and lessons are in `docs/results.md`.
