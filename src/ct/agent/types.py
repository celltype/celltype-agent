"""
Core data types for the agent pipeline.

Defines Plan, Step, Clarification, and ExecutionResult — the data structures
shared across the planner, executor, runner, and UI layers.
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Plan & Step
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """A single research step in a plan."""
    id: int
    description: str = ""
    tool: str = ""
    tool_args: dict = field(default_factory=dict)
    depends_on: list[int] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[dict] = None


@dataclass
class Clarification:
    """Planner needs more information from the user before it can plan."""
    question: str
    missing: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class Plan:
    """A structured research plan."""
    query: str
    steps: list[Step] = field(default_factory=list)
    context: dict = field(default_factory=dict)

    def pending_steps(self) -> list[Step]:
        return [s for s in self.steps if s.status == "pending"]

    def ready_steps(self) -> list[Step]:
        """Steps whose dependencies are all completed."""
        completed_ids = {s.id for s in self.steps if s.status == "completed"}
        return [
            s for s in self.steps
            if s.status == "pending" and all(d in completed_ids for d in s.depends_on)
        ]

    def is_complete(self) -> bool:
        return all(s.status in ("completed", "failed") for s in self.steps)

    def summary(self) -> str:
        lines = [f"Plan: {self.query}", ""]
        for s in self.steps:
            status_icon = {"pending": " ", "running": ">", "completed": "+", "failed": "!"}
            icon = status_icon.get(s.status, "?")
            deps = f" (after {s.depends_on})" if s.depends_on else ""
            lines.append(f"  [{icon}] {s.id}. {s.description} [{s.tool}]{deps}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Result of executing a complete research plan."""
    plan: Plan
    summary: str = ""
    raw_results: dict = field(default_factory=dict)
    duration_s: float = 0.0
    iterations: int = 1
    metadata: dict = field(default_factory=dict)

    def _metadata_header(self) -> list[str]:
        """Build metadata header lines from self.metadata."""
        md = self.metadata
        if not md:
            return []
        lines = [
            "<!--",
            "  Report Metadata (machine-readable provenance)",
        ]
        for key in ("query", "timestamp", "model", "execution_time_s",
                     "tool_success_rate", "profile", "ct_version"):
            if key in md:
                lines.append(f"  {key}: {md[key]}")
        lines.append("-->")
        lines.append("")
        lines.append("| Metadata | Value |")
        lines.append("|----------|-------|")
        if "timestamp" in md:
            lines.append(f"| Generated | {md['timestamp']} |")
        if "model" in md:
            lines.append(f"| Model | {md['model']} |")
        if "execution_time_s" in md:
            lines.append(f"| Execution Time | {md['execution_time_s']:.1f}s |")
        if "tool_success_rate" in md:
            lines.append(f"| Tool Success Rate | {md['tool_success_rate']} |")
        if "profile" in md:
            lines.append(f"| Profile | {md['profile']} |")
        if "ct_version" in md:
            lines.append(f"| ct Version | {md['ct_version']} |")
        lines.append("")
        return lines

    def _quality_scorecard(self) -> list[str]:
        """Build quality scorecard footer from plan steps and metadata."""
        lines = ["## Quality Scorecard", ""]
        lines.append("### Tools Executed")
        lines.append("")
        for step in self.plan.steps:
            status_icon = "PASS" if step.status == "completed" else "FAIL"
            lines.append(f"- `{step.tool}`: {status_icon}")
        lines.append("")

        md = self.metadata
        if md.get("confidence_tier"):
            lines.append(f"**Confidence Tier:** {md['confidence_tier']}")
            lines.append("")
        if md.get("grounding_result"):
            lines.append(f"**Grounding Validation:** {md['grounding_result']}")
            lines.append("")

        data_sources = set()
        for step in self.plan.steps:
            if step.status == "completed" and step.result:
                if isinstance(step.result, dict):
                    for src in step.result.get("data_sources", []):
                        data_sources.add(src)
                    tool_name = step.tool
                    if "." in tool_name:
                        data_sources.add(tool_name.split(".")[0])
        if data_sources:
            lines.append("### Data Sources Referenced")
            lines.append("")
            for src in sorted(data_sources):
                lines.append(f"- {src}")
            lines.append("")

        return lines

    def to_markdown(self) -> str:
        """Generate a markdown report from the execution results."""
        lines = []
        lines.extend(self._metadata_header())
        lines.extend([
            f"# Research Report: {self.plan.query}",
            "",
            f"*Generated by celltype-agent in {self.duration_s:.1f}s*",
            "",
            "---",
            "",
            self.summary,
            "",
            "---",
            "",
            "## Detailed Step Results",
            "",
        ])
        for step in self.plan.steps:
            status = "completed" if step.status == "completed" else "FAILED"
            lines.append(f"### Step {step.id}: {step.description} [{status}]")
            lines.append(f"Tool: `{step.tool}`")
            lines.append("")
            if step.result:
                if isinstance(step.result, dict) and "summary" in step.result:
                    lines.append(step.result["summary"])
                else:
                    lines.append(f"```\n{step.result}\n```")
            lines.append("")

        if self.metadata:
            lines.extend(self._quality_scorecard())

        return "\n".join(lines)
