# Agent Trust Layer

A small, calibrated judge model that reads an AI agent's execution trace and predicts whether the
agent failed and which failure classes (F1–F8) occurred. See [docs/plan.md](docs/plan.md).

## Setup

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev,agent]"   # `agent` adds tau2-bench; omit it for judge-only work
pytest
```

Copy `.env.example` to `.env` and fill in the API keys you have.
