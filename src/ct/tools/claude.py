"""
Claude reasoning tools and Claude Code integration for ct.

Uses Claude as a reasoning engine for open-ended questions that don't fit
pre-built tools, and delegates complex coding/editing tasks to Claude Code CLI.
"""

import subprocess
from pathlib import Path

from ct.tools import registry


REASON_SYSTEM_PROMPT = """\
You are a drug discovery research expert embedded in celltype-agent, \
an autonomous research agent.

You are being called as a reasoning tool — the planner determined that \
this question requires expert scientific reasoning rather than a pre-built \
computational tool or code execution.

{context_section}

Guidelines:
1. Be specific and quantitative where possible
2. Cite mechanisms, pathways, and known biology
3. Distinguish between established facts and hypotheses
4. If data would strengthen your answer, say which datasets/analyses to run next
5. Structure your response with clear sections
6. Keep your response focused and under 800 words
"""

COMPARE_SYSTEM_PROMPT = """\
You are a drug discovery research expert embedded in celltype-agent.

You are being called to compare and evaluate options. Provide a structured \
comparison with clear criteria, trade-offs, and a recommendation.

{context_section}

Guidelines:
1. Use a structured format (table or criteria-based comparison)
2. Be specific about advantages and disadvantages of each option
3. Provide a clear recommendation with rationale
4. Note any caveats or conditions that would change the recommendation
"""

SUMMARIZE_SYSTEM_PROMPT = """\
You are a drug discovery research expert embedded in celltype-agent.

You are being called to synthesize and summarize information. Distill the \
key findings into a concise, actionable summary.

{context_section}

Guidelines:
1. Lead with the most important finding
2. Use bullet points for clarity
3. Highlight actionable next steps
4. Note any gaps or uncertainties in the data
5. Keep the summary under 500 words
"""


def _build_context_section(prior_results: dict = None) -> str:
    """Format prior step results as context for the reasoning LLM."""
    if not prior_results:
        return ""

    lines = ["You have access to results from prior analysis steps:\n"]
    for step_id, result in prior_results.items():
        if isinstance(result, dict):
            summary = result.get("summary", str(result)[:500])
        else:
            summary = str(result)[:500]
        lines.append(f"- Step {step_id}: {summary}")

    return "\n".join(lines)


@registry.register(
    name="claude.reason",
    description="Expert reasoning about drug discovery questions using Claude",
    category="claude",
    parameters={
        "goal": "The question or reasoning task to address",
        "context": "Additional context (e.g., prior findings, constraints)",
    },
    usage_guide=(
        "Use when the query requires expert scientific reasoning, interpretation, "
        "or hypothesis generation that no pre-built tool covers. Good for: "
        "mechanism-of-action reasoning, experimental design advice, literature "
        "interpretation, risk assessment rationale, strategic recommendations. "
        "Do NOT use for tasks that a pre-built tool handles (data retrieval, "
        "similarity search, etc.) — those are faster and cheaper."
    ),
)
def reason(goal: str, context: str = "", _session=None,
           _prior_results=None, **kwargs) -> dict:
    """Use Claude for expert reasoning on drug discovery questions."""
    if _session is None:
        return {
            "summary": "Reasoning unavailable: no active session.",
            "error": "No session provided.",
        }

    llm = _session.get_llm()
    context_section = _build_context_section(_prior_results)

    user_msg = f"Question: {goal}"
    if context:
        user_msg += f"\n\nAdditional context: {context}"

    system = REASON_SYSTEM_PROMPT.format(context_section=context_section)

    from ct.ui.status import ThinkingStatus
    try:
        with ThinkingStatus(_session.console, "reasoning"):
            response = llm.chat(
                system=system,
                messages=[{"role": "user", "content": user_msg}],
                temperature=0.3,
                max_tokens=2048,
            )
    except Exception as e:
        return {
            "summary": f"LLM reasoning failed: {e}",
            "error": str(e),
        }

    return {
        "summary": response.content,
        "model": getattr(response, "model", "unknown"),
        "usage": getattr(response, "usage", None),
    }


@registry.register(
    name="claude.compare",
    description="Compare and evaluate multiple options (compounds, targets, strategies)",
    category="claude",
    parameters={
        "goal": "What to compare and the decision to make",
        "options": "Comma-separated list of options to compare",
        "criteria": "Evaluation criteria (optional)",
    },
    usage_guide=(
        "Use when the user needs to choose between options — compounds, targets, "
        "indications, strategies, CROs, etc. Provides structured comparison with "
        "a recommendation. Combine with pre-built tools first to gather data, "
        "then use claude.compare to interpret and decide."
    ),
)
def compare(goal: str, options: str = "", criteria: str = "",
            _session=None, _prior_results=None, **kwargs) -> dict:
    """Compare multiple options using Claude's reasoning."""
    if _session is None:
        return {
            "summary": "Comparison unavailable: no active session.",
            "error": "No session provided.",
        }

    llm = _session.get_llm()
    context_section = _build_context_section(_prior_results)

    user_msg = f"Decision: {goal}"
    if options:
        user_msg += f"\n\nOptions to compare: {options}"
    if criteria:
        user_msg += f"\n\nEvaluation criteria: {criteria}"

    system = COMPARE_SYSTEM_PROMPT.format(context_section=context_section)

    from ct.ui.status import ThinkingStatus
    try:
        with ThinkingStatus(_session.console, "comparing"):
            response = llm.chat(
                system=system,
                messages=[{"role": "user", "content": user_msg}],
                temperature=0.2,
                max_tokens=2048,
            )
    except Exception as e:
        return {
            "summary": f"LLM comparison failed: {e}",
            "error": str(e),
        }

    return {
        "summary": response.content,
        "model": getattr(response, "model", "unknown"),
        "usage": getattr(response, "usage", None),
    }


@registry.register(
    name="claude.summarize",
    description="Synthesize and summarize research findings into actionable insights",
    category="claude",
    parameters={
        "goal": "What to summarize and the intended audience/purpose",
        "content": "Text content to summarize (optional if prior results available)",
    },
    usage_guide=(
        "Use after multiple analysis steps to distill key findings. Good for: "
        "executive summaries, decision briefs, literature synthesis. Typically "
        "used as a final step after data-gathering tools have run."
    ),
)
def summarize(goal: str, content: str = "", _session=None,
              _prior_results=None, **kwargs) -> dict:
    """Summarize and synthesize research findings."""
    if _session is None:
        return {
            "summary": "Summarization unavailable: no active session.",
            "error": "No session provided.",
        }

    llm = _session.get_llm()
    context_section = _build_context_section(_prior_results)

    user_msg = f"Summarize: {goal}"
    if content:
        user_msg += f"\n\nContent to summarize:\n{content}"

    system = SUMMARIZE_SYSTEM_PROMPT.format(context_section=context_section)

    from ct.ui.status import ThinkingStatus
    try:
        with ThinkingStatus(_session.console, "summarizing"):
            response = llm.chat(
                system=system,
                messages=[{"role": "user", "content": user_msg}],
                temperature=0.2,
                max_tokens=1500,
            )
    except Exception as e:
        return {
            "summary": f"LLM summarization failed: {e}",
            "error": str(e),
        }

    return {
        "summary": response.content,
        "model": getattr(response, "model", "unknown"),
        "usage": getattr(response, "usage", None),
    }


@registry.register(
    name="claude.code",
    description="Delegate a coding task to Claude Code (file editing, refactoring, debugging, test writing)",
    category="claude",
    parameters={
        "task": "Description of the coding task (be specific: which files, what changes, what to test)",
        "allowed_tools": "Claude Code tools to allow (default: 'Read,Edit,Write,Bash,Glob,Grep')",
        "max_budget": "Max spend in USD (default: 1.0)",
    },
    usage_guide=(
        "Use for complex coding tasks that need iterative edit-test-fix cycles: "
        "refactoring code, writing tests, debugging scripts, modifying config files, "
        "building analysis pipelines. Claude Code handles the full read→edit→test→fix loop. "
        "Do NOT use for simple single-shot file reads or edits — use files.* tools instead. "
        "Do NOT use for drug discovery research — use ct's specialized tools."
    ),
)
def code(task: str, allowed_tools: str = "Read,Edit,Write,Bash,Glob,Grep",
         max_budget: float = 1.0, _session=None, _prior_results=None,
         **kwargs) -> dict:
    """Delegate a coding task to Claude Code CLI.

    Spawns `claude -p` in non-interactive mode with permission bypass and
    a budget cap. Returns Claude Code's output as the tool result.
    """
    import os
    import shutil

    enabled = str(os.environ.get("CT_ENABLE_CLAUDE_CODE", "")).strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if _session is not None and hasattr(_session, "config"):
        enabled = bool(_session.config.get("agent.enable_claude_code_tool", False))

    if not enabled:
        return {
            "summary": (
                "claude.code is disabled by policy (opt-in). "
                "Enable with: ct config set agent.enable_claude_code_tool true"
            ),
            "error": "disabled_by_policy",
        }

    claude_path = shutil.which("claude")
    if not claude_path:
        return {
            "summary": "Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code",
            "error": "claude_not_found",
        }

    # Build context from prior results so Claude Code knows what ct has already done
    context_parts = []
    if _prior_results:
        for step_id, result in _prior_results.items():
            if isinstance(result, dict):
                summary = result.get("summary", "")[:300]
            else:
                summary = str(result)[:300]
            if summary:
                context_parts.append(f"- Step {step_id}: {summary}")

    full_prompt = task
    if context_parts:
        full_prompt = (
            f"Context from prior research steps:\n"
            + "\n".join(context_parts)
            + f"\n\nTask: {task}"
        )

    cmd = [
        claude_path,
        "-p", full_prompt,
        "--output-format", "text",
        "--permission-mode", "bypassPermissions",
        "--allowed-tools", allowed_tools,
        "--max-budget-usd", str(max_budget),
        "--no-session-persistence",
    ]

    try:
        if _session:
            from ct.ui.status import ThinkingStatus
            with ThinkingStatus(_session.console, "coding"):
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(Path.cwd()),
                    timeout=300,
                )
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(Path.cwd()),
                timeout=300,
            )
    except subprocess.TimeoutExpired:
        return {
            "summary": "Claude Code timed out after 5 minutes.",
            "error": "timeout",
        }
    except Exception as e:
        return {
            "summary": f"Failed to run Claude Code: {e}",
            "error": str(e),
        }

    output = result.stdout.strip()
    stderr = result.stderr.strip()

    # Truncate very long output
    if len(output) > 15000:
        output = output[:15000] + "\n... [truncated]"

    if result.returncode == 0 and output:
        summary = output[:2000]
        if len(output) > 2000:
            summary += "..."
        return {
            "summary": summary,
            "full_output": output,
            "exit_code": result.returncode,
        }
    elif output:
        return {
            "summary": f"Claude Code finished (exit {result.returncode}): {output[:1000]}",
            "full_output": output,
            "stderr": stderr[:2000] if stderr else "",
            "exit_code": result.returncode,
        }
    else:
        return {
            "summary": f"Claude Code produced no output (exit {result.returncode}). Stderr: {stderr[:500]}",
            "error": "no_output",
            "stderr": stderr[:2000] if stderr else "",
            "exit_code": result.returncode,
        }
