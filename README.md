# Agent Trust Layer

A small, calibrated judge model that reads an AI agent's execution trace and predicts whether the
agent failed and which failure classes (F1–F8) occurred. See [docs/plan.md](docs/plan.md).

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
