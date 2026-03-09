"""
Interactive terminal for ct.

Provides a REPL-style interface for continuous research sessions.
"""

import random
import re
import shlex
import subprocess
import time
import threading
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from ct.ui.markdown import LeftMarkdown
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import has_completions
from prompt_toolkit.formatted_text import HTML, ANSI
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from pathlib import Path


@dataclass
class MentionCandidate:
    """A candidate item for the @ mention autocomplete."""
    name: str          # e.g., "target.coessentiality" or "depmap"
    kind: str          # "tool", "database", or "file"
    category: str      # e.g., "target", "dataset", "file"
    description: str   # truncated for display


# Slash commands available in the interactive terminal
SLASH_COMMANDS = {
    "/help": "Show command reference with examples",
    "/tools": "List all tools with status (stable/experimental)",
    "/model": "Switch LLM model/provider interactively",
    "/settings": "Configure UI and agent preferences",
    "/config": "Show active runtime configuration",
    "/keys": "Show API key setup status by service",
    "/doctor": "Run readiness diagnostics and fix hints",
    "/usage": "Show session token/cost usage",
    "/copy": "Copy the last answer to clipboard",
    "/export": "Export current session transcript to markdown",
    "/notebook": "Export current session as Jupyter notebook (.ipynb)",
    "/compact": "Compress session context for longer runs",
    "/agents": "Run a query with N parallel research agents",
    "/sessions": "List recent saved sessions",
    "/resume": "Resume a previous session by id/index",
    "/case-study": "Run/list curated case studies (/case-study list)",
    "/plan": "Toggle plan mode — preview & approve before executing",
    "/clear": "Clear the screen",
    "/exit": "Exit the terminal",
}

# Models available for switching, grouped by provider
AVAILABLE_MODELS = {
    "anthropic": [
        ("claude-sonnet-4-5-20250929", "Sonnet 4.5", "$3/$15 per M tokens — fast, great for most queries"),
        ("claude-haiku-4-5-20251001", "Haiku 4.5", "$0.80/$4 per M tokens — fastest, cheapest"),
        ("claude-opus-4-6", "Opus 4.6", "$15/$75 per M tokens — most capable, use for complex reasoning"),
    ],
    "openai": [
        ("gpt-4o", "GPT-4o", "$2.50/$10 per M tokens"),
        ("gpt-4o-mini", "GPT-4o Mini", "$0.15/$0.60 per M tokens"),
    ],
}

from ct.ui.suggestions import DEFAULT_SUGGESTIONS


# ---------------------------------------------------------------------------
# @ Mention: datasets and completer
# ---------------------------------------------------------------------------

DATASET_CANDIDATES = [
    ("depmap", "dataset", "DepMap CRISPR/model data"),
    ("prism", "dataset", "PRISM drug sensitivity"),
    ("l1000", "dataset", "L1000 gene expression signatures"),
    ("proteomics", "dataset", "Proteomics log2FC matrix"),
    ("msigdb", "dataset", "MSigDB gene sets"),
    ("string", "dataset", "STRING protein interaction network"),
]

KNOWN_DATASETS = frozenset(d[0] for d in DATASET_CANDIDATES)


def _get_workflow_names() -> frozenset[str]:
    """Lazily load workflow names."""
    try:
        from ct.agent.workflows import WORKFLOWS
        return frozenset(WORKFLOWS.keys())
    except Exception:
        return frozenset()


def extract_mentions(text: str):
    """Parse @mentions from input text.

    Returns:
        tuple of (cleaned_query, tool_names, dataset_names, workflow_names)
    """
    dataset_names_set = {d[0] for d in DATASET_CANDIDATES}
    workflow_names_set = _get_workflow_names()
    tool_pattern = re.compile(r"@(\w+\.\w+)")
    word_pattern = re.compile(r"@(\w+)")

    tools = []
    datasets = []
    workflows = []

    # Find @category.tool_name mentions first
    for m in tool_pattern.finditer(text):
        tools.append(m.group(1))

    # Find @dataset and @workflow mentions (single word, no dot)
    cleaned = tool_pattern.sub("", text)
    for m in word_pattern.finditer(cleaned):
        name = m.group(1)
        if name in dataset_names_set:
            datasets.append(name)
        elif name in workflow_names_set:
            workflows.append(name)

    # Strip all recognized @mentions from query
    query = re.sub(r"@\w+(?:\.\w+)?", "", text).strip()
    # Collapse multiple spaces
    query = re.sub(r"\s{2,}", " ", query)

    return query, tools, datasets, workflows


def build_mention_context(tools: list[str], datasets: list[str], workflows: list[str] | None = None) -> str:
    """Build context string from extracted mentions for planner injection."""
    parts = []
    if tools:
        tool_list = ", ".join(tools)
        parts.append(
            f"User specifically requested these tools: {tool_list}. "
            f"You MUST include these tools in your plan."
        )
    if datasets:
        for ds in datasets:
            desc = next(
                (d[2] for d in DATASET_CANDIDATES if d[0] == ds), ds
            )
            parts.append(f"User requested dataset: {ds} ({desc}).")
    if workflows:
        try:
            from ct.agent.workflows import WORKFLOWS
            for wf_name in workflows:
                wf = WORKFLOWS.get(wf_name)
                if wf:
                    steps = ", ".join(s["tool"] for s in wf.get("steps", []))
                    parts.append(
                        f"User requested workflow '{wf_name}': {wf['description']}. "
                        f"Follow this tool sequence: {steps}"
                    )
        except Exception:
            pass
    return "\n".join(parts)



def _extract_llm_suggestions(synthesis_text: str) -> list[str]:
    """Extract follow-up suggestions from the LLM synthesis output.

    Looks for a 'Suggested Next Steps' section and extracts bullet/numbered items.
    Handles various formats: **"quoted text"**, plain bullets, numbered lists.
    """
    suggestions = []
    in_section = False

    for line in synthesis_text.split("\n"):
        stripped = line.strip()

        # Detect the suggested next steps section
        if "suggested next" in stripped.lower() or "follow-up" in stripped.lower():
            if stripped.startswith("#") or stripped.startswith("**"):
                in_section = True
                continue

        if in_section:
            # Stop at next heading (not related to suggestions)
            if stripped.startswith("#") and "suggested" not in stripped.lower() and "follow" not in stripped.lower():
                break
            # Extract bullet items (-, *, 1., 2., etc.)
            if stripped and (stripped[0] in "-*" or (len(stripped) > 1 and stripped[0].isdigit() and stripped[1] in ".)")):
                # Remove bullet prefix
                text = stripped.lstrip("-*0123456789.) ").strip()
                # Extract quoted text from **"..."** or "..." patterns
                quoted = re.findall(r'["\u201c]([^"\u201d]+)["\u201d]', text)
                if quoted:
                    # Use the longest quoted string (the actual query)
                    text = max(quoted, key=len)
                else:
                    # Remove markdown formatting
                    text = text.strip("`").strip("*").strip("_")
                # Skip if it's a header or too short
                if len(text) > 10 and not text.startswith("#"):
                    suggestions.append(text)

    return suggestions[:5]





# prompt_toolkit style — dim ghost text, colored prompt, dark completion menu
PT_STYLE = Style.from_dict({
    "prompt": "bold #50fa7b",
    "placeholder": "#555555",
    "bottom-toolbar": "#888888 bg:#1a1a2e",
    # Completion menu — dark background so mention colors stay readable
    "completion-menu": "bg:#1a1a2e #cccccc",
    "completion-menu.completion": "bg:#1a1a2e #cccccc",
    "completion-menu.completion.current": "bg:#333355 #ffffff bold",
    "completion-menu.meta.completion": "bg:#1a1a2e #888888",
    "completion-menu.meta.completion.current": "bg:#333355 #aaaaaa",
    "scrollbar.background": "bg:#1a1a2e",
    "scrollbar.button": "bg:#333355",
    # Mention kind colors
    "mention-tool": "#00d7ff",     # cyan for tool mentions
    "mention-dataset": "#50fa7b",  # green for dataset mentions
    "mention-file": "#ffd700",     # yellow for file mentions
    "mention-workflow": "#ff79c6",  # pink for workflow mentions
})


class SlashCompleter(Completer):
    """Autocomplete slash commands when input starts with /."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/"):
            for cmd, desc in SLASH_COMMANDS.items():
                if cmd.startswith(text):
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display_meta=desc,
                    )


class MentionCompleter(Completer):
    """Autocomplete tools, datasets, and files when input contains @.

    Supports tabbed filtering via TABS (All / Tools / DB / Files).
    Candidates are ``(name, category, description, kind)`` tuples where
    *kind* is ``"tool"``, ``"dataset"``, or ``"file"``.
    """

    TABS = ["All", "Tools", "DB", "Files", "Flows"]
    _TAB_FILTERS = {
        0: None,          # All
        1: "tool",        # Tools
        2: "dataset",     # DB
        3: "file",        # Files
        4: "workflow",    # Flows
    }

    def __init__(self, candidates: list[tuple[str, str, str, str]] | None = None):
        self.candidates = candidates or []
        self._active_tab = 0

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        # Find the last @ in the text
        at_pos = text.rfind("@")
        if at_pos < 0:
            return

        partial = text[at_pos + 1:].lower()
        replace_len = len(text) - at_pos  # replace from @ onwards

        # Filter by active tab
        kind_filter = self._TAB_FILTERS.get(self._active_tab)

        # Group by category for ordering
        by_category: dict[str, list[tuple]] = {}
        for name, category, description, kind in self.candidates:
            if kind_filter and kind != kind_filter:
                continue
            # Case-insensitive substring match against name, category, description
            if (partial in name.lower()
                    or partial in category.lower()
                    or partial in description.lower()):
                by_category.setdefault(category, []).append(
                    (name, description, kind)
                )

        # Style mapping per kind
        styles = {
            "tool": "class:mention-tool",
            "dataset": "class:mention-dataset",
            "file": "class:mention-file",
            "workflow": "class:mention-workflow",
        }

        for category in sorted(by_category):
            for name, description, kind in sorted(by_category[category]):
                yield Completion(
                    f"@{name}",
                    start_position=-replace_len,
                    display_meta=description,
                    style=styles.get(kind, ""),
                )


class MergedCompleter(Completer):
    """Delegates to SlashCompleter for / and MentionCompleter for @."""

    def __init__(self, slash: Completer, mention: MentionCompleter):
        self._slash = slash
        self._mention = mention

    @property
    def mention_completer(self) -> MentionCompleter:
        return self._mention

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.lstrip().startswith("/"):
            yield from self._slash.get_completions(document, complete_event)
        elif "@" in text:
            yield from self._mention.get_completions(document, complete_event)


# ---------------------------------------------------------------------------
# Plan preview rendering
# ---------------------------------------------------------------------------

def render_plan_preview(plan, console=None):
    """Render a plan as a Rich Panel for user approval.

    Args:
        plan: A Plan object with .steps (each having id, tool, description,
              tool_args, depends_on).
        console: Optional Rich Console. Defaults to a new Console().

    Returns:
        The rendered Text (for testing) or prints to console.
    """
    from rich.text import Text
    from ct.ui.traces import format_args

    if console is None:
        console = Console()

    lines = Text()
    lines.append("Research Plan\n\n", style="bold")

    for step in plan.steps:
        # Dependency indicator
        deps = getattr(step, "depends_on", []) or []
        dep_str = ""
        if deps:
            dep_str = f" (after step {', '.join(str(d) for d in deps)})"

        lines.append(f"  {step.id}. ", style="bold cyan")
        lines.append(step.tool or "", style="cyan")
        if dep_str:
            lines.append(dep_str, style="dim")
        lines.append("\n")

        # Description
        desc = getattr(step, "description", "") or ""
        if desc:
            lines.append(f"     {desc}\n", style="")

        # Key args
        args = getattr(step, "tool_args", {}) or {}
        args_str = format_args(args)
        if args_str:
            lines.append(f"     {args_str}\n", style="dim")

    console.print(Panel(lines, border_style="cyan", title="Plan Preview"))
    return lines


def _build_key_bindings(terminal):
    """Key bindings: Tab accepts ghost suggestion, Ctrl+C double-tap to exit,
    Ctrl+O toggle verbose, Ctrl+J insert newline."""
    kb = KeyBindings()

    @kb.add("tab")
    def _accept_suggestion(event):
        buf = event.app.current_buffer
        if not buf.text:
            idx = terminal._suggestion_idx % len(terminal._suggestions)
            buf.insert_text(terminal._suggestions[idx])
        else:
            buf.start_completion()

    @kb.add("c-c")
    def _handle_ctrl_c(event):
        buf = event.app.current_buffer
        now = time.time()
        if now - terminal._last_interrupt < 0.5:
            # Double Ctrl+C — signal exit
            event.app.exit(result="__EXIT__")
        else:
            terminal._last_interrupt = now
            terminal._show_exit_hint = True
            buf.reset()
            event.app.invalidate()

            def _clear_hint():
                time.sleep(0.5)
                terminal._show_exit_hint = False
                try:
                    if event.app.is_running:
                        event.app.invalidate()
                except Exception:
                    pass

            threading.Thread(target=_clear_hint, daemon=True).start()

    @kb.add("c-o")
    def _toggle_verbose(event):
        """Toggle verbose mode mid-session."""
        terminal.session.verbose = not terminal.session.verbose
        state = "ON" if terminal.session.verbose else "OFF"
        terminal._verbose_hint = f"Verbose {state}"
        event.app.invalidate()

        def _clear_hint():
            time.sleep(2.0)
            terminal._verbose_hint = None
            try:
                if event.app.is_running:
                    event.app.invalidate()
            except Exception:
                pass

        threading.Thread(target=_clear_hint, daemon=True).start()

    @kb.add("c-j")
    def _insert_newline(event):
        """Insert a newline for multi-line input."""
        event.app.current_buffer.insert_text("\n")

    @kb.add("escape", "enter")
    def _insert_newline_alt(event):
        """Option+Enter / Alt+Enter inserts newline."""
        event.app.current_buffer.insert_text("\n")

    @kb.add("enter", filter=has_completions)
    def _accept_first_completion(event):
        """When completions are visible on a / command, accept the current
        (or first) completion and submit."""
        buf = event.app.current_buffer
        cs = buf.complete_state
        if buf.text.lstrip().startswith("/") and cs and cs.completions:
            # If nothing is selected yet, jump to the first completion
            if cs.complete_index is None:
                buf.go_to_completion(0)
                cs = buf.complete_state  # refresh after navigation
            if cs and cs.current_completion:
                buf.apply_completion(cs.current_completion)
            buf.validate_and_handle()
        else:
            # Non-slash: just submit normally
            buf.cancel_completion()
            buf.validate_and_handle()

    @kb.add("right", filter=has_completions)
    def _mention_tab_right(event):
        """Switch to next mention tab while completions are visible."""
        completer = terminal._merged_completer
        if completer is None:
            return
        mc = completer.mention_completer
        mc._active_tab = (mc._active_tab + 1) % len(mc.TABS)
        buf = event.app.current_buffer
        buf.cancel_completion()
        buf.start_completion()

    @kb.add("left", filter=has_completions)
    def _mention_tab_left(event):
        """Switch to previous mention tab while completions are visible."""
        completer = terminal._merged_completer
        if completer is None:
            return
        mc = completer.mention_completer
        mc._active_tab = (mc._active_tab - 1) % len(mc.TABS)
        buf = event.app.current_buffer
        buf.cancel_completion()
        buf.start_completion()

    return kb


class InteractiveTerminal:
    """Interactive research session terminal."""

    def __init__(self, config=None, verbose=False):
        from ct.agent.session import Session
        self.session = Session(config=config, verbose=verbose, mode="interactive")
        self.console = Console()
        self.history_file = Path.home() / ".ct" / "history"
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self._last_interrupt = 0.0
        self._show_exit_hint = False
        self._verbose_hint = None
        self._last_response = None  # Last synthesis text for /copy
        self._suggestions = list(DEFAULT_SUGGESTIONS)
        random.shuffle(self._suggestions)
        self._suggestion_idx = 0
        # Build @ mention completer with tool + dataset + file candidates
        mention_candidates = self._build_mention_candidates()
        self._merged_completer = MergedCompleter(
            slash=SlashCompleter(),
            mention=MentionCompleter(mention_candidates),
        )
        self._prompt_session = PromptSession(
            history=FileHistory(str(self.history_file)),
            completer=self._merged_completer,
            complete_while_typing=True,
            style=PT_STYLE,
            key_bindings=_build_key_bindings(self),
            multiline=False,  # Ctrl+J / Alt+Enter for newlines
        )
        # Auto-highlight (not apply) the first completion for slash commands
        # so the dropdown shows which item will be accepted on Enter.
        def _auto_highlight_first(buf):
            if (buf.text.lstrip().startswith("/")
                    and buf.complete_state
                    and buf.complete_state.complete_index is None
                    and buf.complete_state.completions):
                # Set the index directly — this highlights without changing text
                buf.complete_state.go_to_index(0)

        self._prompt_session.default_buffer.on_completions_changed += _auto_highlight_first

    def _build_mention_candidates(self) -> list[tuple[str, str, str, str]]:
        """Build the candidate list for @ mention completion.

        Returns (name, category, description, kind) tuples.
        """
        candidates = []
        # Add datasets
        for name, category, description in DATASET_CANDIDATES:
            candidates.append((name, category, description, "dataset"))
        # Add tools from registry (lazy load)
        try:
            from ct.tools import registry, ensure_loaded
            ensure_loaded()
            for tool in registry.list_tools():
                candidates.append(
                    (tool.name, tool.category, tool.description[:80], "tool")
                )
        except Exception:
            pass  # Registry not available — datasets still work
        # Add workflow candidates
        try:
            from ct.agent.workflows import WORKFLOWS
            for wf_name, wf in WORKFLOWS.items():
                n_steps = len(wf.get("steps", []))
                candidates.append(
                    (wf_name, "workflow", f"{wf['description']} ({n_steps} steps)", "workflow")
                )
        except Exception:
            pass  # Workflows not available
        # Add file candidates from configured data directory
        try:
            data_base = self.session.config.get("data.base", "")
            if data_base:
                data_path = Path(data_base)
                if data_path.is_dir():
                    for f in sorted(data_path.rglob("*")):
                        if f.is_file() and not f.name.startswith("."):
                            candidates.append(
                                (f.name, "file", str(f.relative_to(data_path)), "file")
                            )
        except Exception:
            pass  # Best-effort file scanning
        return candidates

    def _current_placeholder(self):
        """Return the current ghost suggestion as dim placeholder text."""
        text = self._suggestions[self._suggestion_idx % len(self._suggestions)]
        return HTML(f'<style fg="#555555">{text}</style>')

    def _advance_suggestion(self):
        """Move to next ghost suggestion."""
        self._suggestion_idx = (self._suggestion_idx + 1) % len(self._suggestions)

    def _update_suggestions(self, query: str, plan=None, result=None):
        """Replace suggestions with contextual follow-ups based on last query.

        Uses LLM-suggested follow-ups extracted from the synthesis output.
        """
        suggestions = []

        # Extract LLM-suggested follow-ups from synthesis
        if result and hasattr(result, 'summary') and result.summary:
            llm_suggestions = _extract_llm_suggestions(result.summary)
            suggestions.extend(llm_suggestions)

        if suggestions:
            self._suggestions = suggestions[:5]
            self._suggestion_idx = 0
        else:
            self._advance_suggestion()

    def _model_display_name(self, model_id: str = None) -> str:
        """Get a short display name for a model ID."""
        model_id = model_id or self.session.current_model
        names = {
            "claude-sonnet-4-5-20250929": "Sonnet 4.5",
            "claude-haiku-4-5-20251001": "Haiku 4.5",
            "claude-opus-4-6": "Opus 4.6",
            "gpt-4o": "GPT-4o",
            "gpt-4o-mini": "GPT-4o Mini",
        }
        return names.get(model_id, model_id)

    def _mention_completing(self) -> bool:
        """Check if @ mention completions are currently active."""
        try:
            buf = self._prompt_session.app.current_buffer
            if buf.complete_state and "@" in buf.text:
                return True
        except Exception:
            pass
        return False

    def _bottom_toolbar(self):
        if self._show_exit_hint:
            return HTML('<style fg="#888888">  Press Ctrl+C again to exit</style>')
        if self._verbose_hint:
            return HTML(f'<style fg="#50fa7b">  {self._verbose_hint}</style>')

        # Show tab bar when @ mention completions are active
        if self._mention_completing():
            mc = self._merged_completer.mention_completer
            tabs = []
            for i, label in enumerate(mc.TABS):
                if i == mc._active_tab:
                    tabs.append(f'<style fg="#50fa7b" bg="#333333"><b>[{label}]</b></style>')
                else:
                    tabs.append(f'<style fg="#555555"> {label} </style>')
            tab_bar = "  ".join(tabs)
            return HTML(f'  {tab_bar}  <style fg="#555555">·  ←/→ switch tab</style>')

        model = self._model_display_name()
        verbose = '<style fg="#555555">  </style><style fg="#1a1a2e" bg="#50fa7b"> verbose </style>' if self.session.verbose else ""
        plan = '<style fg="#555555">  </style><style fg="#1a1a2e" bg="#ff79c6"> plan mode </style>' if self.session.config.get("agent.plan_preview", False) else ""
        gpu_cloud = '<style fg="#555555">  ·  GPU tools:</style> <style fg="#50fa7b">CellType Cloud</style>'
        return HTML(f'  <style fg="#ffffff" bg="#50a0ff"> {model} </style>{verbose}{plan}<style fg="#555555">  ? for commands  ·  Ctrl+O verbose</style>{gpu_cloud}')

    def run(self, initial_context: dict = None, resume_id: str = None):
        """Run the interactive session."""
        from ct.agent.loop import AgentLoop

        context = initial_context or {}
        term_width = self.console.width

        # AgentLoop persists across queries — holds trajectory for multi-turn memory
        if resume_id:
            try:
                if resume_id == "last":
                    self.agent = AgentLoop.resume_latest(self.session)
                else:
                    self.agent = AgentLoop.resume(self.session, resume_id)
                n = len(self.agent.trajectory.turns)
                title = self.agent.trajectory.title or "untitled"
                self.console.print(f"  [green]Resumed session[/green] [bold]{self.agent.trajectory.session_id}[/bold] — {title} ({n} turns)")
                self.console.print()
            except FileNotFoundError as e:
                self.console.print(f"  [yellow]{e}[/yellow]")
                self.agent = AgentLoop(self.session)
        else:
            self.agent = AgentLoop(self.session)

        while True:
            try:
                # Separator line above prompt
                self.console.print(f"[#333333]{'─' * term_width}[/]")

                query = self._prompt_session.prompt(
                    [("class:prompt", "❯ ")],
                    bottom_toolbar=self._bottom_toolbar,
                    placeholder=self._current_placeholder(),
                ).strip()
                self._show_exit_hint = False
            except EOFError:
                self.console.print("\nGoodbye.")
                break

            # Handle double Ctrl+C exit signal from key binding
            if query == "__EXIT__":
                self.console.print("Goodbye.")
                break

            if not query:
                self._advance_suggestion()
                continue

            # Handle slash commands and plain commands
            cmd = query.lower()

            # Auto-resolve partial slash commands — first match wins
            # (e.g. "/mod" → "/model", "/co" → "/config")
            if cmd.startswith("/") and cmd not in SLASH_COMMANDS:
                prefix = cmd.split()[0]  # handle "/export file.md" → "/export"
                matches = [c for c in SLASH_COMMANDS if c.startswith(prefix)]
                if matches:
                    cmd = matches[0] + cmd[len(prefix):]
                    query = matches[0] + query[len(prefix):]
            if cmd in ("exit", "quit", "q", "/exit", "/quit"):
                self.console.print("Goodbye.")
                break
            if cmd in ("help", "/help", "?"):
                self._show_help()
                self._advance_suggestion()
                continue
            if cmd in ("tools", "/tools"):
                from ct.tools import registry, ensure_loaded, tool_load_errors
                ensure_loaded()
                self.console.print(registry.list_tools_table())
                errors = tool_load_errors()
                if errors:
                    names = ", ".join(sorted(errors.keys())[:8])
                    extra = "" if len(errors) <= 8 else f" (+{len(errors) - 8} more)"
                    self.console.print(
                        f"[yellow]Warning:[/yellow] {len(errors)} tool module(s) failed to load: "
                        f"{names}{extra}"
                    )
                self._advance_suggestion()
                continue
            if cmd in ("model", "/model"):
                self._switch_model()
                self._advance_suggestion()
                continue
            if cmd in ("settings", "/settings"):
                self._change_settings()
                self._advance_suggestion()
                continue
            if cmd in ("plan", "/plan"):
                self._toggle_plan_mode()
                self._advance_suggestion()
                continue
            if cmd in ("usage", "/usage"):
                self._show_usage()
                self._advance_suggestion()
                continue
            if cmd in ("config", "/config"):
                from ct.agent.config import Config
                self.console.print(Config.load().to_table())
                self._advance_suggestion()
                continue
            if cmd in ("keys", "/keys"):
                from ct.agent.config import Config
                self.console.print(Config.load().keys_table())
                self._advance_suggestion()
                continue
            if cmd in ("doctor", "/doctor"):
                from ct.agent.doctor import has_errors, run_checks, to_table
                checks = run_checks(self.session.config, session=self.session)
                self.console.print(to_table(checks))
                if has_errors(checks):
                    self.console.print("  [red]Blocking issues found.[/red]")
                else:
                    self.console.print("  [green]No blocking issues found.[/green]")
                self._advance_suggestion()
                continue
            if cmd in ("clear", "/clear"):
                self.console.clear()
                self._advance_suggestion()
                continue
            if cmd in ("copy", "/copy"):
                self._copy_last_response()
                continue
            if cmd.startswith("/export"):
                parts = query.split(maxsplit=1)
                filename = parts[1] if len(parts) > 1 else None
                self._export_session(filename)
                continue
            if cmd.startswith("/notebook"):
                parts = query.split(maxsplit=1)
                filename = parts[1] if len(parts) > 1 else None
                self._export_notebook(filename)
                continue
            if cmd.startswith("/compact"):
                parts = query.split(maxsplit=1)
                instructions = parts[1] if len(parts) > 1 else None
                self._compact_context(instructions)
                continue
            if cmd in ("sessions", "/sessions"):
                self._list_sessions()
                continue
            if cmd.startswith("/resume"):
                parts = query.split(maxsplit=1)
                sid = parts[1].strip() if len(parts) > 1 else None
                self._resume_session(sid)
                continue
            if cmd.startswith("/agents"):
                self._handle_agents_command(query, context)
                continue
            if cmd.startswith("/case-study"):
                self._handle_case_study_command(query, context)
                continue

            # ! prefix — shell command
            if query.startswith("!"):
                self._run_shell(query[1:].strip())
                continue

            # "continue" — resume interrupted synthesis or continue conversation
            if cmd in ("continue", "go on", "keep going"):
                if self.agent._last_plan is not None:
                    self.console.print(f"  [cyan]Continuing synthesis...[/cyan]\n")
                    try:
                        result = self.agent.continue_synthesis()
                        self.console.print()
                    except KeyboardInterrupt:
                        self.console.print("\n  [dim]Interrupted.[/dim]")
                        continue
                    if result is not None:
                        self._last_response = result.summary
                        self._update_suggestions(
                            self.agent._last_query or query, result.plan, result,
                        )
                    continue
                # No interrupted state — fall through to normal query
                # (planner will use session history to understand context)

            # Execute query via AgentLoop (observe-replan loop + trajectory)
            # Synthesis is streamed to stdout in real-time by the executor.
            try:
                self.console.print()
                result = self._run_with_clarification(query, context)
                self.console.print()
            except KeyboardInterrupt:
                self.console.print("\n  [yellow]Interrupted.[/yellow]")
                continue

            if result is not None:
                self._last_response = result.summary
                self._update_suggestions(query, result.plan, result)

    def _run_with_clarification(self, query: str, context: dict):
        """Run a query, handling clarification requests interactively."""
        from ct.agent.loop import ClarificationNeeded

        run_context = dict(context)

        # Extract @mentions and inject into context
        cleaned_query, mention_tools, mention_datasets, mention_workflows = extract_mentions(query)
        if mention_tools or mention_datasets or mention_workflows:
            mention_ctx = build_mention_context(mention_tools, mention_datasets, mention_workflows)
            run_context["mention_context"] = mention_ctx
            query = cleaned_query

        max_clarifications = 3  # Prevent infinite clarification loops

        for _ in range(max_clarifications):
            try:
                return self.agent.run(query, run_context)
            except ClarificationNeeded as e:
                clar = e.clarification
                self.console.print(f"  [cyan]{clar.question}[/cyan]")
                if clar.suggestions:
                    self.console.print(f"  [dim]e.g. {', '.join(clar.suggestions[:3])}[/dim]")

                try:
                    answer = self._prompt_session.prompt(
                        [("class:prompt", "  ❯ ")],
                    ).strip()
                except (EOFError, KeyboardInterrupt):
                    self.console.print("  [dim]Cancelled.[/dim]")
                    return None

                if not answer:
                    self.console.print("  [dim]Cancelled.[/dim]")
                    return None

                # Add the answer to context using the missing parameter name
                if clar.missing:
                    run_context[clar.missing[0]] = answer
                # Also append to the query so the planner gets full context
                query = f"{query} — {answer}"

        return self.agent.run(query, run_context)

    def _switch_model(self):
        """Interactive model switcher."""
        provider = self.session.config.get("llm.provider", "anthropic")
        models = AVAILABLE_MODELS.get(provider, [])
        current = self.session.current_model

        self.console.print(f"\n  [cyan]Current model:[/cyan] {self._model_display_name()} ({current})")
        self.console.print(f"  [cyan]Provider:[/cyan] {provider}\n")

        if not models:
            self.console.print(f"  [yellow]No model options configured for provider '{provider}'[/yellow]")
            return

        for i, (model_id, display, desc) in enumerate(models, 1):
            marker = " [green]*[/green]" if model_id == current else "  "
            self.console.print(f"  {marker} [{i}] {display} — [dim]{desc}[/dim]")

        self.console.print()

        try:
            choice = self._prompt_session.prompt(
                [("class:prompt", "  Select model (number): ")],
            ).strip()
        except (EOFError, KeyboardInterrupt):
            return

        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(models):
            self.console.print("  [dim]Cancelled.[/dim]")
            return

        idx = int(choice) - 1
        model_id, display, _ = models[idx]

        if model_id == current:
            self.console.print(f"  [dim]Already using {display}.[/dim]")
            return

        self.session.set_model(model_id)
        self.session.config.save()  # Persist to ~/.ct/config.json
        self.console.print(f"  [green]Switched to {display}[/green] ({model_id})")

    def _getch(self):
        """Read a single character from standard input without requiring Enter."""
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        # Handle ctrl-c (x03) and ctrl-d (x04)
        if ch in ('\x03', '\x04'):
            raise KeyboardInterrupt
        return ch

    def _change_settings(self):
        """Interactive settings configuration menu."""
        from ct.agent.config import Config, AGENT_PROFILE_PRESETS
        from ct.ui.status import SPINNERS
        
        cfg = Config.load()
        
        while True:
            self.console.print("\n  [cyan]Settings Menu[/cyan]")
            self.console.print("  [1] UI Loading Spinner")
            self.console.print("  [2] Agent Profile (Research/Pharma/Enterprise)")
            self.console.print("  [3] Auto-publish HTML Reports")
            self.console.print("  [0] Done")
            self.console.print("\n  Select option: ", end="")
            
            import sys
            sys.stdout.flush()
            
            try:
                choice = self._getch()
            except KeyboardInterrupt:
                self.console.print()
                return
                
            self.console.print(choice)
            
            if choice == "0":
                break
            elif choice == "1":
                spinners = list(SPINNERS.keys())
                current_spinner = cfg.get("ui.spinner", "dna_helix")
                self.console.print(f"\n  [cyan]UI Loading Spinner[/cyan]")
                for i, spinner_id in enumerate(spinners, 1):
                    marker = " [green]*[/green]" if spinner_id == current_spinner else "  "
                    self.console.print(f"  {marker} [{i}] {spinner_id}")
                self.console.print("\n  Select spinner: ", end="")
                sys.stdout.flush()
                
                try:
                    s_choice = self._getch()
                except KeyboardInterrupt:
                    self.console.print()
                    return
                    
                self.console.print(s_choice)
                
                if s_choice.isdigit() and 1 <= int(s_choice) <= len(spinners):
                    new_spinner = spinners[int(s_choice) - 1]
                    if new_spinner != current_spinner:
                        cfg.set("ui.spinner", new_spinner)
                        cfg.save()
                        self.console.print(f"  [green]Spinner updated to:[/green] {new_spinner}")
                else:
                    self.console.print("  [dim]Cancelled.[/dim]")

            elif choice == "2":
                profiles = list(AGENT_PROFILE_PRESETS.keys())
                current_profile = cfg.get("agent.profile", "research")
                self.console.print(f"\n  [cyan]Agent Profile[/cyan]")
                for i, profile_id in enumerate(profiles, 1):
                    marker = " [green]*[/green]" if profile_id == current_profile else "  "
                    self.console.print(f"  {marker} [{i}] {profile_id}")
                self.console.print("\n  Select profile: ", end="")
                sys.stdout.flush()
                
                try:
                    p_choice = self._getch()
                except KeyboardInterrupt:
                    self.console.print()
                    return
                    
                self.console.print(p_choice)
                
                if p_choice.isdigit() and 1 <= int(p_choice) <= len(profiles):
                    new_profile = profiles[int(p_choice) - 1]
                    if new_profile != current_profile:
                        cfg.set("agent.profile", new_profile)
                        cfg.save()
                        self.console.print(f"  [green]Profile updated to:[/green] {new_profile}")
                else:
                    self.console.print("  [dim]Cancelled.[/dim]")
                    
            elif choice == "3":
                current_html = cfg.get("output.auto_publish_html_interactive", True)
                self.console.print(f"\n  [cyan]Auto-publish HTML Reports[/cyan]")
                self.console.print(f"  Current: [bold]{'Yes' if current_html else 'No'}[/bold]")
                self.console.print("\n  Enable? (y/n): ", end="")
                sys.stdout.flush()
                
                try:
                    h_choice = self._getch().lower()
                except KeyboardInterrupt:
                    self.console.print()
                    return
                    
                self.console.print(h_choice)
                
                if h_choice == "y":
                    cfg.set("output.auto_publish_html_interactive", True)
                    cfg.save()
                    self.console.print(f"  [green]Auto-publish HTML enabled.[/green]")
                elif h_choice == "n":
                    cfg.set("output.auto_publish_html_interactive", False)
                    cfg.save()
                    self.console.print(f"  [green]Auto-publish HTML disabled.[/green]")
                else:
                    self.console.print("  [dim]Cancelled.[/dim]")
            else:
                self.console.print("  [dim]Invalid choice.[/dim]")

    def _toggle_plan_mode(self):
        """Toggle plan mode — agent shows plan for approval before executing."""
        cfg = self.session.config
        current = bool(cfg.get("agent.plan_preview", False))
        cfg.set("agent.plan_preview", not current)
        if not current:
            self.console.print("  [#ff79c6]Plan mode ON[/] — agent will preview its plan before executing")
        else:
            self.console.print("  [dim]Plan mode OFF[/dim] — agent will execute directly")

    def _show_usage(self):
        """Show token usage and cost for this session."""
        llm = self.session.get_llm()
        if not hasattr(llm, 'usage') or not llm.usage.calls:
            self.console.print("  [dim]No LLM calls made yet.[/dim]")
            return
        self.console.print(f"  {llm.usage.summary()}")

    def _copy_last_response(self):
        """Copy the last synthesis response to the system clipboard."""
        if not self._last_response:
            self.console.print("  [dim]No response to copy yet.[/dim]")
            return

        try:
            proc = subprocess.run(
                ["pbcopy"], input=self._last_response.encode(),
                capture_output=True, timeout=5,
            )
            if proc.returncode == 0:
                preview = self._last_response[:80].replace("\n", " ")
                self.console.print(f"  [green]Copied to clipboard.[/green] [dim]{preview}...[/dim]")
            else:
                # Fallback for non-macOS
                self.console.print(f"  [yellow]Clipboard not available. Use /export instead.[/yellow]")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.console.print(f"  [yellow]Clipboard not available. Use /export instead.[/yellow]")

    def _export_session(self, filename: str = None):
        """Export the session transcript to a markdown file."""
        if not hasattr(self, 'agent') or not self.agent.trajectory.turns:
            self.console.print("  [dim]No session data to export yet.[/dim]")
            return

        output_dir = Path.cwd() / "exports"
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename:
            path = output_dir / filename
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = output_dir / f"session_{ts}.md"

        lines = ["# ct Session Export\n"]
        lines.append(f"*Exported {time.strftime('%Y-%m-%d %H:%M')}*\n")
        lines.append(f"*Model: {self._model_display_name()}*\n\n---\n")

        for i, turn in enumerate(self.agent.trajectory.turns, 1):
            lines.append(f"## Query {i}\n")
            lines.append(f"**Q:** {turn.query}\n")
            lines.append(f"**A:** {turn.answer}\n")
            if turn.entities:
                lines.append(f"*Entities: {', '.join(turn.entities)}*\n")
            if turn.tools_used:
                lines.append(f"*Tools: {', '.join(turn.tools_used)}*\n")
            lines.append("\n---\n")

        path.write_text("\n".join(lines))
        self.console.print(f"  [green]Exported to[/green] {path}")

    def _export_notebook(self, filename: str = None):
        """Export the current session trace as a Jupyter notebook."""
        if not hasattr(self, 'agent') or not hasattr(self.agent, 'trace_store'):
            self.console.print("  [dim]No trace data available.[/dim]")
            return

        trace_store = self.agent.trace_store
        if not trace_store.path.exists():
            self.console.print("  [dim]No trace data yet. Run a query first.[/dim]")
            return

        try:
            from ct.reports.notebook import trace_to_notebook, save_notebook
        except ImportError:
            self.console.print("  [red]nbformat required.[/red] pip install nbformat")
            return

        nb = trace_to_notebook(trace_store.path)

        output_dir = Path.cwd() / "exports"
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename:
            path = output_dir / filename
        else:
            import re
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", trace_store.session_id).strip("_")
            path = output_dir / f"session_{slug}.ipynb"

        save_notebook(nb, path)
        self.console.print(f"  [green]Notebook exported to[/green] {path}")
        self.console.print(f"  [dim]Open with: jupyter lab {path}[/dim]")

    def _compact_context(self, instructions: str = None):
        """Summarize session trajectory to free context window space."""
        if not hasattr(self, 'agent') or not self.agent.trajectory.turns:
            self.console.print("  [dim]Nothing to compact yet.[/dim]")
            return

        n_turns = len(self.agent.trajectory.turns)
        if n_turns <= 2:
            self.console.print("  [dim]Session too short to compact.[/dim]")
            return

        # Build a summary of the session using the LLM
        context = self.agent.trajectory.context_for_planner()
        focus = f"\nFocus: {instructions}" if instructions else ""
        prompt = (
            f"Summarize this research session into a brief paragraph that preserves "
            f"key findings, entities, and conclusions. Be specific about results and numbers.{focus}\n\n"
            f"{context}"
        )

        try:
            llm = self.session.get_llm()
            response = llm.chat(
                system="You are a research session summarizer. Be concise but preserve specific results.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1200,
            )
            summary = response.content if hasattr(response, "content") else str(response)
            if not summary.strip():
                raise ValueError("Summarizer returned empty output")

            # Replace all turns except the last one with a single summary turn
            from ct.agent.trajectory import Turn
            last_turn = self.agent.trajectory.turns[-1]
            summary_turn = Turn(
                query="[session summary]",
                answer=summary,
                entities=list(self.agent.trajectory.entities()),
                tools_used=[],
                timestamp=time.time(),
            )
            self.agent.trajectory.turns = [summary_turn, last_turn]
            self.console.print(f"  [green]Compacted[/green] {n_turns} turns → 2 (summary + last)")
        except Exception as e:
            self.console.print(f"  [red]Compact failed:[/red] {e}")

    def _run_shell(self, cmd: str):
        """Execute a shell command and display output."""
        if not cmd:
            self.console.print("  [dim]Usage: !<command>  (e.g., !ls .)[/dim]")
            return

        from ct.tools.shell import _is_blocked
        blocked_reason = _is_blocked(cmd)
        if blocked_reason:
            self.console.print(f"  [yellow]Command blocked:[/yellow] {blocked_reason}")
            return

        try:
            args = shlex.split(cmd, posix=True)
        except ValueError as e:
            self.console.print(f"  [red]Invalid command syntax:[/red] {e}")
            return

        # Expand user-home shorthand for convenience when not using a shell.
        args = [str(Path(arg).expanduser()) if arg.startswith("~") else arg for arg in args]

        try:
            result = subprocess.run(
                args,
                shell=False,
                cwd=str(Path.cwd()),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.stdout:
                self.console.print(result.stdout.rstrip())
            if result.stderr:
                self.console.print(f"[red]{result.stderr.rstrip()}[/red]")
            if result.returncode != 0 and not result.stderr:
                self.console.print(f"  [dim]Exit code: {result.returncode}[/dim]")
        except subprocess.TimeoutExpired:
            self.console.print("  [yellow]Command timed out (30s limit).[/yellow]")
        except Exception as e:
            self.console.print(f"  [red]Error: {e}[/red]")

    def _list_sessions(self):
        """Show recent saved sessions."""
        from ct.agent.trajectory import Trajectory
        sessions = Trajectory.list_sessions()
        if not sessions:
            self.console.print("  [dim]No saved sessions.[/dim]")
            return

        self.console.print(f"\n  [cyan]Recent sessions:[/cyan]\n")
        for i, s in enumerate(sessions[:10], 1):
            title = s.get("title", "untitled")[:60]
            sid = s.get("session_id", "?")
            n = s.get("n_turns", 0)
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(s.get("created_at", 0)))
            current = " [green]*[/green]" if hasattr(self, 'agent') and self.agent.trajectory.session_id == sid else "  "
            self.console.print(f"  {current}[{i}] [bold]{sid}[/bold] — {title} ({n} turns, {ts})")

        self.console.print(f"\n  [dim]Use /resume <id> or /resume <number> to restore.[/dim]")

    def _resume_session(self, identifier: str = None):
        """Resume a previous session."""
        from ct.agent.loop import AgentLoop
        from ct.agent.trajectory import Trajectory

        sessions = Trajectory.list_sessions()
        if not sessions:
            self.console.print("  [dim]No saved sessions.[/dim]")
            return

        if identifier is None:
            # Show picker
            self._list_sessions()
            try:
                choice = self._prompt_session.prompt(
                    [("class:prompt", "  Select session: ")],
                ).strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not choice:
                return
            identifier = choice

        # Resolve: number → session from list, or direct ID
        if identifier.isdigit():
            idx = int(identifier) - 1
            if 0 <= idx < len(sessions):
                session_id = sessions[idx]["session_id"]
            else:
                self.console.print("  [dim]Invalid number.[/dim]")
                return
        elif identifier == "last":
            session_id = sessions[0]["session_id"]
        else:
            session_id = identifier

        try:
            self.agent = AgentLoop.resume(self.session, session_id)
            n = len(self.agent.trajectory.turns)
            title = self.agent.trajectory.title or "untitled"
            self.console.print(f"  [green]Resumed[/green] [bold]{session_id}[/bold] — {title} ({n} turns)")

            # Show last turn as context
            if self.agent.trajectory.turns:
                last = self.agent.trajectory.turns[-1]
                preview = last.answer[:150].replace("\n", " ")
                self.console.print(f"  [dim]Last: {last.query}[/dim]")
                self.console.print(f"  [dim]→ {preview}...[/dim]")
        except FileNotFoundError:
            self.console.print(f"  [yellow]Session '{session_id}' not found.[/yellow]")

    def _handle_agents_command(self, query: str, context: dict):
        """Handle /agents N [query] command."""
        parts = query.split(maxsplit=2)
        # /agents N query  or  /agents N
        if len(parts) < 2 or not parts[1].isdigit():
            self.console.print(
                "  [dim]Usage: /agents N [query]  "
                "(e.g., /agents 3 profile lenalidomide)[/dim]"
            )
            return

        n_threads = int(parts[1])
        if n_threads < 1:
            self.console.print("  [dim]Need at least 1 agent.[/dim]")
            return

        if len(parts) > 2:
            agent_query = parts[2]
        else:
            # Prompt for query
            try:
                agent_query = self._prompt_session.prompt(
                    [("class:prompt", "  Research question: ")],
                ).strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print("  [dim]Cancelled.[/dim]")
                return
            if not agent_query:
                self.console.print("  [dim]Cancelled.[/dim]")
                return

        self._run_orchestrated(agent_query, context, n_threads)

    def _run_orchestrated(self, query: str, context: dict, n_threads: int):
        """Run a query using the multi-agent orchestrator."""
        from ct.agent.orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            self.session,
            n_threads=n_threads,
            trajectory=self.agent.trajectory if hasattr(self, 'agent') else None,
        )

        try:
            self.console.print()
            result = orchestrator.run(query, context)
            self.console.print()

            if result is not None:
                self._last_response = result.summary
                self._update_suggestions(query, result.merged_plan, result)
        except KeyboardInterrupt:
            self.console.print("\n  [yellow]Interrupted.[/yellow]")
        except Exception as e:
            self.console.print(f"\n  [red]Orchestrator error:[/red] {e}")

    def _handle_case_study_command(self, query: str, context: dict):
        """Handle /case-study <id> or /case-study list."""
        from ct.agent.case_studies import CASE_STUDIES, run_case_study

        parts = query.split(maxsplit=1)
        arg = parts[1].strip() if len(parts) > 1 else ""

        if not arg or arg == "list":
            from rich.table import Table

            table = Table(title="Case Studies")
            table.add_column("ID", style="cyan")
            table.add_column("Drug")
            table.add_column("Threads", style="dim")
            table.add_column("Description")
            for case_id, case in CASE_STUDIES.items():
                table.add_row(
                    case_id,
                    case.name,
                    str(len(case.thread_goals)),
                    case.description[:80] + ("..." if len(case.description) > 80 else ""),
                )
            self.console.print(table)
            self.console.print(
                "\n  [dim]Usage: /case-study <id>  (e.g., /case-study revlimid)[/dim]"
            )
            return

        case_id = arg.split()[0].lower()
        if case_id not in CASE_STUDIES:
            available = ", ".join(sorted(CASE_STUDIES.keys()))
            self.console.print(
                f"  [red]Unknown case study '{case_id}'.[/red] Available: {available}"
            )
            return

        case = CASE_STUDIES[case_id]
        self.console.print(
            f"\n  [cyan]Case Study:[/cyan] [bold]{case.name}[/bold]"
            f"\n  [dim]{case.description}[/dim]\n"
        )

        try:
            result = run_case_study(self.session, case_id)
            self.console.print()

            if result is not None:
                self._last_response = result.summary
                self._update_suggestions(case.compound, result.merged_plan, result)
        except KeyboardInterrupt:
            self.console.print("\n  [yellow]Interrupted.[/yellow]")
        except Exception as e:
            self.console.print(f"\n  [red]Case study error:[/red] {e}")

    def _show_help(self):
        command_lines = ["**Slash Commands:**"]
        for command in sorted(SLASH_COMMANDS.keys()):
            command_lines.append(f"- `{command}` — {SLASH_COMMANDS[command]}")

        help_text = (
            "**Usage:**\n"
            "- Type any research question to investigate.\n"
            "- `!command` — run one shell command safely (no pipes/chaining; e.g., `!ls .`).\n"
            + "\n".join(command_lines)
            + "\n\n"
            "**Shortcuts:**\n"
            "- `Ctrl+O` — toggle verbose mode\n"
            "- `Ctrl+J` or `Alt+Enter` — insert newline (multi-line input)\n"
            "- `Tab` — accept ghost suggestion\n"
            "- `Ctrl+C` × 2 — exit\n"
            "\n"
            "**Examples:**\n"
            '- `find top genetically supported Parkinson targets`\n'
            '- `/agents 3 find repurposing hypotheses for ulcerative colitis`\n'
            '- `/case-study list` then `/case-study revlimid`\n'
            '- `ct report publish` (from shell) to convert latest markdown report to HTML.'
        )
        self.console.print(Panel(
            LeftMarkdown(help_text),
            title="ct Help",
            border_style="cyan",
        ))
