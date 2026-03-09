"""
Session management: holds config, LLM clients, and shared state for a ct session.
"""

import time
from pathlib import Path
from rich.console import Console

from ct.agent.config import Config


class Session:
    """Manages state for a ct research session."""

    def __init__(self, config: Config = None, verbose: bool = False, mode: str = "batch"):
        self.config = config or Config.load()
        self.verbose = verbose
        self.mode = mode  # "interactive" or "batch"
        self.console = Console()
        self._llm = None
        self._scratchpad = []
        self._tool_health_failures: dict[str, list[float]] = {}
        self._tool_health_suppressed_until: dict[str, float] = {}

    def get_llm(self):
        """Get or create the LLM client based on config."""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def _create_llm(self):
        """Create LLM client from config."""
        from ct.models.llm import LLMClient

        provider = self.config.get("llm.provider", "anthropic")
        model = self.config.get("llm.model", None)
        api_key = self.config.llm_api_key(provider)

        return LLMClient(
            provider=provider,
            model=model,
            api_key=api_key,
        )

    def set_model(self, model: str, provider: str = None):
        """Switch the LLM model mid-session. Resets the client."""
        if provider:
            self.config.set("llm.provider", provider)
        self.config.set("llm.model", model)
        self._llm = None  # Force re-creation on next get_llm()

    @property
    def current_model(self) -> str:
        """Return the current model name."""
        if self._llm:
            return self._llm.model
        return self.config.get("llm.model") or "claude-opus-4-6"

    def log(self, message: str):
        """Log to scratchpad (for debugging/transparency)."""
        self._scratchpad.append(message)
        if self.verbose:
            self.console.print(f"  [dim]{message}[/dim]")

    def save_scratchpad(self, path: Path):
        """Save scratchpad to file for debugging."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self._scratchpad))

    # --- Runtime tool-health tracking (for planner suppression) ---

    def _tool_health_enabled(self) -> bool:
        return bool(self.config.get("agent.tool_health_enabled", True))

    def _tool_failure_window_seconds(self) -> int:
        return int(self.config.get("agent.tool_health_failure_window_s", 1800))

    def _tool_fail_threshold(self) -> int:
        return int(self.config.get("agent.tool_health_fail_threshold", 2))

    def _tool_suppress_seconds(self) -> int:
        return int(self.config.get("agent.tool_health_suppress_seconds", 900))

    def _is_transient_tool_error(self, error_text: str) -> bool:
        text = str(error_text or "").lower()
        transient_markers = (
            "timeout",
            "timed out",
            "connection",
            "dns",
            "service unavailable",
            "rate limit",
            "429",
            "500",
            "502",
            "503",
            "504",
            "expected json",
            "invalid json response",
            "temporarily unavailable",
        )
        return any(marker in text for marker in transient_markers)

    def record_tool_success(self, tool_name: str):
        """Clear runtime failure pressure after a successful execution."""
        if not tool_name:
            return
        self._tool_health_failures.pop(tool_name, None)
        self._tool_health_suppressed_until.pop(tool_name, None)

    def record_tool_failure(self, tool_name: str, error_text: str = ""):
        """Record transient tool failures and suppress flaky tools temporarily."""
        if not self._tool_health_enabled() or not tool_name:
            return
        if not self._is_transient_tool_error(error_text):
            return

        now = time.time()
        window_s = max(60, self._tool_failure_window_seconds())
        threshold = max(1, self._tool_fail_threshold())
        suppress_s = max(60, self._tool_suppress_seconds())

        history = [t for t in self._tool_health_failures.get(tool_name, []) if now - t <= window_s]
        history.append(now)
        self._tool_health_failures[tool_name] = history

        if len(history) >= threshold:
            self._tool_health_suppressed_until[tool_name] = now + suppress_s

    def tool_health_suppressed_tools(self) -> set[str]:
        """Return tools currently suppressed due to repeated transient failures."""
        if not self._tool_health_enabled():
            return set()
        now = time.time()
        suppressed = set()
        expired = []
        for name, until in self._tool_health_suppressed_until.items():
            if now < until:
                suppressed.add(name)
            else:
                expired.append(name)
        for name in expired:
            self._tool_health_suppressed_until.pop(name, None)
            self._tool_health_failures.pop(name, None)
        return suppressed
