# CLAUDE.md — Agent Trust Layer

## Project goal

Build a small, calibrated, fine-tuned judge model that reads an AI agent's execution trace (steps, tool calls, tool results, final answer) and predicts whether the agent failed and which failure classes (F1–F8) occurred. It replaces a frontier-LLM-as-judge at a fraction of the cost. The judge drives three decisions (auto-pass, human review, auto-fail) and powers a CI regression gate, monitoring and an optional router.

This is a portfolio project for senior ML interviews: **measured, honest results beat clever code.** Full design: `docs/plan.md`. Read the relevant section before starting work on a phase, and do not start a phase whose previous gate (section 11) is not met.

## Compute: Colab T4 only

Training and GPU inference run on a **Google Colab T4** (16 GB VRAM, Turing). Claude Code runs locally and has no GPU, so:

- **Precision:** fp16 only. Never use bf16 (`bf16=True`, `torch.bfloat16`, `bnb_4bit_compute_dtype=torch.bfloat16`). Use `torch.float16`.
- **Attention:** no FlashAttention-2 (unsupported on Turing). Use `attn_implementation="sdpa"` or `"eager"`.
- **fp16 stability:** use the Trainer's gradient scaling (`fp16=True`); guard against NaN losses and log them.
- **Memory:** SLM training uses 4-bit QLoRA, gradient checkpointing, batch size 1–2 with gradient accumulation. Default context budget is 4,096 tokens for the SLM and 512 for the encoder. Do not propose models above ~3B for training without flagging the memory risk.
- **Colab sessions die:** every training script must checkpoint regularly to a configurable `output_dir` (Google Drive or the HF Hub) and resume with `--resume`.
- **Notebooks are launchers only.** Files in `notebooks/` clone the repo, `pip install -e .`, set secrets, and call CLI entry points. All logic lives in `src/atl/`.
- **Local smoke tests:** every training and eval script must run end to end on CPU with `--smoke` (tiny model such as a `hf-internal-testing/tiny-random-*` checkpoint, ~20 examples, 1–2 steps). Verify with `--smoke` locally before saying GPU code is ready.

## The gold test set rule (non-negotiable)

The gold test set is the hand-verified data in `data/gold/` (and any split named `test`, `gold`, `heldout_model` or `ood`). It is what makes every reported number trustworthy.

- **Never** load the gold or held-out sets in training, hyperparameter search, prompt tuning, threshold selection, calibration or early stopping. Use `train`, `val` and `calib` splits for those.
- Gold data is read **only** by `src/atl/eval/final_eval.py`, through the guarded loader in `src/atl/data/splits.py`. Do not add other code paths that read it.
- **Never** modify, relabel, regenerate, re-split or delete gold data or its hash file (`data/gold/MANIFEST.sha256`). If something seems wrong with it, stop and tell me.
- Splits are by **task ID**, never by trace. Do not change split logic without asking.
- If a request would break any of these rules, say so instead of doing it.

## Coding conventions

- **Python 3.11**, `src/` layout, package `atl`. Dependencies in `pyproject.toml`; pin versions used on Colab.
- **Style:** `ruff check` and `ruff format` (line length 100). Type hints on all public functions. Short functions and docstrings on modules and public functions.
- **Data models:** pydantic v2 for traces, labels and API payloads. The trace schema in `src/atl/traces/schema.py` is the contract; change it only with a version bump.
- **Versioning:** the trace serialiser (`judge/serialise.py`) carries a `SERIALISER_VERSION`. Any change to its output bumps the version. Model artefacts record the serialiser version and the data version they were trained on.
- **Config, not constants:** experiment settings live in YAML under `configs/`. CLI with `typer`; each script takes `--config`.
- **Reproducibility:** set seeds everywhere (`atl.utils.seed_everything`). Log configs, metrics and artefacts to Weights & Biases.
- **Logging:** use the `logging` module, not `print`.
- **Tests:** `pytest` in `tests/`, CPU-only, the whole suite under ~1 minute. Every new module gets tests; metrics, splits and calibration code need tests with hand-checked expected values.
- **LLM API calls** (agent runs, LLM labeller): always cached on disk keyed by the hash of model + prompt + inputs, with retries and a cost log. Never re-run paid calls that are already cached.
- **Secrets:** API keys come from environment variables (`.env` locally, Colab secrets on Colab). Never hard-code or commit them.
- **Data:** `data/` is gitignored and synced from the HF Hub or Drive. Never commit datasets, model weights or checkpoints.
- **Metrics:** report per-class results, not only aggregates, with bootstrap confidence intervals where sample sizes are small.
- **Commits:** small and focused, Conventional Commits style (`feat:`, `fix:`, `test:`, `docs:`, `exp:`).

## Working style

- Before writing code for a phase, propose a short plan (files, functions, tests) and wait for my OK on anything non-trivial.
- Prefer simple, readable code over abstractions. This code will be explained in interviews.
- When results look surprisingly good, suspect leakage first and check the splits.
- Record notable findings, failures and decisions in `docs/results.md` as we go. They are interview material.
