"""
Synthesis quality evaluation for ct agent outputs.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import re


@dataclass
class SynthesisQuality:
    """Structured result from synthesis quality evaluation."""

    ok: bool
    issues: list[str]
    metadata: dict[str, Any] | None = None


_STEP_REF_PATTERN = re.compile(r"\[step:(\d+)\]", re.IGNORECASE)
_TOOL_PATTERN = re.compile(r"\b\w+\.\w+\b")


def _extract_section(text: str, header: str) -> str:
    """Return the text under a markdown header up to the next header."""
    pattern = re.compile(rf"^##\s+{re.escape(header)}\s*$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return ""
    start = match.end()
    next_header = re.search(r"^##\s+", text[start:], re.MULTILINE)
    end = start + next_header.start() if next_header else len(text)
    return text[start:end].strip()


def _parse_numbered_lines(section: str) -> list[str]:
    """Extract markdown numbered-list lines from a section."""
    lines = []
    for line in section.splitlines():
        if re.match(r"^\s*\d+\.\s+", line):
            lines.append(line.strip())
    return lines


def evaluate_synthesis_quality(
    synthesis: str,
    *,
    completed_step_ids: Iterable[int] | None = None,
    require_key_evidence: bool = True,
    min_next_steps: int = 2, 
    max_next_steps: int = 3,
) -> SynthesisQuality:
    """Evaluate a synthesized answer for basic structural quality.

    This is designed to be a very basic grader for synthesis quality.

    Current checks:
    - Key evidence section exists and cites at least one completed step.
    - Suggested next steps section has between min_next_steps and max_next_steps numbered items.
    - At least one suggested next step references a concrete tool.
    """
    issues: list[str] = []
    completed = {int(s) for s in (completed_step_ids or [])}

    text = synthesis or ""

    # Key Evidence section and step references
    key_section = _extract_section(text, "Key Evidence")
    key_lines = [ln.strip() for ln in key_section.splitlines() if ln.strip().startswith("-")]

    if require_key_evidence and not key_lines:
        issues.append("missing_key_evidence_bullets")
    else:
        cited_steps: set[int] = set()
        for line in key_lines:
            for m in _STEP_REF_PATTERN.finditer(line):
                cited_steps.add(int(m.group(1)))

        if require_key_evidence and not cited_steps:
            issues.append("key_evidence_missing_step_references")
        elif require_key_evidence and not (cited_steps & completed):
            issues.append("key_evidence_not_linked_to_completed_steps")

    # Suggested Next Steps count and specificity
    next_section = _extract_section(text, "Suggested Next Steps")
    next_lines = _parse_numbered_lines(next_section)

    n_steps = len(next_lines)
    if n_steps < min_next_steps:
        issues.append("too_few_next_steps")
    if n_steps > max_next_steps:
        issues.append("too_many_next_steps")

    # At least one step should reference a concrete tool (e.g. genomics.coloc)
    has_tool_like_step = any(_TOOL_PATTERN.search(line) for line in next_lines)
    if n_steps > 0 and not has_tool_like_step:
        issues.append("next_steps_not_tool_specific")

    ok = not issues

    return SynthesisQuality(
        ok=ok,
        issues=issues,
        metadata={
            "n_key_lines": len(key_lines),
            "n_next_steps": n_steps,
        },
    )
