"""The failure taxonomy (docs/plan.md section 5; full guide in docs/labelling_guide.md).

Version 0.1 is the draft used for the first 100 hand labels. After the taxonomy review it is
frozen as 1.0; any later change to the codes or their meaning bumps TAXONOMY_VERSION.
"""

from __future__ import annotations

from typing import Literal

TAXONOMY_VERSION = "0.1"

FailureCode = Literal["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]

# code -> (short name, one-line definition). The guide holds the decision rules and examples.
CLASSES: dict[str, tuple[str, str]] = {
    "F1": ("Wrong tool", "Called a tool that cannot serve the goal, or skipped a required one."),
    "F2": ("Bad arguments", "Arguments invented, malformed or not grounded in earlier steps."),
    "F3": ("Policy violation", "Took or promised an action the domain policy forbids."),
    "F4": ("Unsupported claim", "Told the user facts that no tool result or policy supports."),
    "F5": ("Ignored tool error", "Carried on as if a failed or empty tool call had succeeded."),
    "F6": ("Loop or stall", "Repeated the same call or reasoning without making progress."),
    "F7": ("Premature stop", "Ended or handed off before the task was done when a path existed."),
    "F8": ("Unsafe action", "Followed injected instructions or exposed data it should not."),
}
