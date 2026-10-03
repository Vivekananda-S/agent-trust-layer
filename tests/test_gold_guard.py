"""Static guard: only splits.py and final_eval.py may reference protected-data access.

Turns the CLAUDE.md gold test set rule into a check that fails CI.
"""

from pathlib import Path

ROOT = Path(__file__).parents[1]
SCANNED_DIRS = ["src", "notebooks", "configs"]
ALLOWED = {
    Path("src/atl/data/splits.py"),
    Path("src/atl/eval/final_eval.py"),
}
FORBIDDEN = ["load_gold", "load_protected_split", "data/gold", "MANIFEST.sha256"]


def test_only_allowed_files_touch_protected_data() -> None:
    offenders = []
    for d in SCANNED_DIRS:
        for path in sorted((ROOT / d).rglob("*")):
            if path.suffix not in {".py", ".ipynb", ".yaml", ".yml"}:
                continue
            rel = path.relative_to(ROOT)
            if rel in ALLOWED:
                continue
            text = path.read_text(encoding="utf-8")
            offenders += [f"{rel}: {s}" for s in FORBIDDEN if s in text]
    assert not offenders, "protected-data access outside final_eval:\n" + "\n".join(offenders)
