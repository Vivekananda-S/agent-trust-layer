"""Static guard: only src/atl/agent/ may import tau2, so the judge never depends on it."""

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
ALLOWED_DIR = Path("src/atl/agent")
IMPORT = re.compile(r"^\s*(import tau2|from tau2[\s.])", re.MULTILINE)


def test_tau2_imported_only_in_agent_package() -> None:
    offenders = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        rel = path.relative_to(ROOT)
        if rel.is_relative_to(ALLOWED_DIR):
            continue
        if IMPORT.search(path.read_text(encoding="utf-8")):
            offenders.append(str(rel))
    assert not offenders, f"tau2 imported outside {ALLOWED_DIR}: {offenders}"
