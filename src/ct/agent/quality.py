"""Evaluator for agent synthesis outputs.

Checks that a synthesis report meets minimum quality criteria before it is
accepted by the benchmark gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class SynthesisQuality:
    ok: bool
    issues: list[str] = field(default_factory=list)


_STEP_REF = re.compile(r"\[step:(\d+)]")
_NEXT_STEP_LINE = re.compile(r"^\d+\.\s+Run\s+\S+", re.MULTILINE)


def evaluate_synthesis_quality(
    synthesis: str,
    *,
    completed_step_ids: Iterable[int] | None = None,
    require_key_evidence: bool = True,
    min_next_steps: int = 2,
    max_next_steps: int = 3,
) -> SynthesisQuality:
    """Score a synthesis report and return pass/fail with issues.

    Checks performed:
    * Key Evidence items must cite at least one completed step via ``[step:N]``.
    * Suggested Next Steps must propose actionable tool invocations
      (``Run <tool.name> …``) within the allowed count range.
    """
    issues: list[str] = []
    valid_ids = set(completed_step_ids) if completed_step_ids else set()

    evidence_block = _extract_section(synthesis, "Key Evidence")
    next_steps_block = _extract_section(synthesis, "Suggested Next Steps")

    if require_key_evidence:
        if not evidence_block:
            issues.append("Missing '## Key Evidence' section")
        else:
            refs = _STEP_REF.findall(evidence_block)
            if not refs:
                issues.append("Key Evidence contains no step citations")
            elif valid_ids:
                cited = {int(r) for r in refs}
                ungrounded = cited - valid_ids
                if ungrounded:
                    issues.append(
                        f"Key Evidence cites uncompleted steps: {sorted(ungrounded)}"
                    )

    actionable = _NEXT_STEP_LINE.findall(next_steps_block) if next_steps_block else []
    n = len(actionable)
    if n < min_next_steps:
        issues.append(
            f"Too few actionable next steps ({n} < {min_next_steps})"
        )
    elif n > max_next_steps:
        issues.append(
            f"Too many next steps ({n} > {max_next_steps})"
        )

    return SynthesisQuality(ok=len(issues) == 0, issues=issues)


def _extract_section(text: str, heading: str) -> str | None:
    """Return the body text under ``## <heading>`` up to the next ``##``."""
    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}\s*\n(.*?)(?=^##\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else None
