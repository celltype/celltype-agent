"""
Configuration management for ct.

Config is stored at ~/.ct/config.json and manages:
- LLM provider settings (Anthropic, OpenAI, local models)
- Data directory paths (DepMap, PRISM, etc.)
- Output preferences
- Tool-specific settings
"""

import json
import os
import logging
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Load .env from current dir and project root
load_dotenv()
load_dotenv(Path(__file__).resolve().parents[3] / ".env")  # repo root

from rich.table import Table

CONFIG_DIR = Path.home() / ".ct"
CONFIG_FILE = CONFIG_DIR / "config.json"
VALID_LLM_PROVIDERS = frozenset({"anthropic", "openai", "local", "gluelm"})
logger = logging.getLogger("ct.config")

DEFAULTS = {
    "llm.provider": "anthropic",
    "llm.model": "claude-opus-4-6",
    "llm.api_key": None,
    "llm.openai_api_key": None,
    "llm.temperature": 0.1,
    "data.base": str(CONFIG_DIR / "data"),
    "data.depmap": None,
    "data.prism": None,
    "data.l1000": None,
    "data.msigdb": None,
    "data.alphafold": None,
    "data.string": None,
    "data.proteomics": None,

    "api.data_endpoint": None,
    "api.clue_key": None,

    "output.format": "markdown",
    "output.verbose": False,
    "output.auto_publish_html_interactive": True,
    "output.auto_publish_html_batch": False,

    "ui.spinner": "benzene_breathing",

    "models.gluelm": None,
    "models.deepternary": None,
    "models.boltz2": None,

    "api.ibm_rxn_key": None,
    "api.lens_key": None,

    "notification.sendgrid_api_key": None,
    "notification.from_email": None,
    "notification.auto_send": False,

    "compute.lambda_api_key": None,
    "compute.runpod_api_key": None,
    "compute.default_provider": "lambda",
    "compute.mode": "cloud",

    "cloud.endpoint": "https://api.celltype.com",
    "cloud.dashboard_url": "https://cloud.celltype.com",

    "gpu.name": None,
    "gpu.vram_mb": None,
    "gpu.setup_completed": False,

    "sandbox.timeout": 30,
    "sandbox.output_dir": str(Path.cwd() / "outputs"),
    "sandbox.max_retries": 2,

    "agent.max_iterations": 3,
    "agent.enable_experimental_tools": False,
    "agent.observer_model": None,
    "agent.executor_max_retries": 2,
    "agent.executor_loop_limit": 50,
    "agent.observer_confidence_threshold": 0.8,
    "agent.synthesis_max_tokens": 8192,
    "agent.enforce_grounded_synthesis": True,
    "agent.enforce_claim_content_validation": True,
    "agent.confidence_scoring_enabled": True,
    "agent.min_step_success_rate": 0.5,
    "agent.require_key_evidence_section": True,
    "agent.allow_creative_hypotheses": True,
    "agent.max_hypotheses": 3,
    "agent.grounding_repair_retries": 1,
    "agent.log_evidence_store": True,
    "agent.memory_retrieval_enabled": True,
    "agent.memory_retrieval_limit": 3,
    "agent.verifier_model": None,
    "agent.verifier_provider": None,
    "agent.verifier_repair_retries": 1,
    "agent.quality_gate_enabled": True,
    "agent.quality_gate_strict": False,
    "agent.quality_gate_repair_retries": 1,
    "agent.quality_gate_repair_non_strict": False,
    "agent.quality_gate_min_next_steps": 2,
    "agent.quality_gate_max_next_steps": 3,
    "agent.synthesis_style": "standard",
    "agent.profile": "research",
    "agent.enable_claude_code_tool": False,
    "agent.parallel_default_count": 3,
    "agent.parallel_auto_suggest": True,
    "agent.parallel_max_threads": 5,
    "agent.planner_max_tools": 90,
    "agent.planner_compact_tool_descriptions": True,
    "agent.tool_health_enabled": True,
    "agent.tool_health_fail_threshold": 2,
    "agent.tool_health_failure_window_s": 1800,
    "agent.tool_health_suppress_seconds": 900,
    "agent.preflight_validation_enabled": True,

    "enterprise.enforce_policy": False,
    "enterprise.audit_enabled": True,
    "enterprise.audit_dir": str(Path.home() / ".ct" / "audit"),
    "enterprise.blocked_tools": "",
    "enterprise.blocked_categories": "",
    "enterprise.require_tool_allowlist": False,
    "enterprise.tool_allowlist": "",
    "enterprise.max_cost_usd_per_query": 0.0,

    "knowledge.enable_substrate": True,
    "knowledge.auto_ingest_evidence": True,
    "knowledge.substrate_path": str(Path.home() / ".ct" / "knowledge" / "substrate.json"),
    "knowledge.schema_monitor_enabled": False,

    "ops.base_dir": str(Path.home() / ".ct" / "ops"),
}

AGENT_PROFILE_PRESETS = {
    "research": {
        "agent.enforce_grounded_synthesis": True,
        "agent.enforce_claim_content_validation": True,
        "agent.require_key_evidence_section": True,
        "agent.allow_creative_hypotheses": True,
        "agent.quality_gate_enabled": True,
        "agent.quality_gate_strict": False,
        "agent.quality_gate_repair_retries": 1,
        "agent.quality_gate_repair_non_strict": False,
        "agent.synthesis_style": "standard",
        "agent.memory_retrieval_enabled": True,
        "agent.enable_claude_code_tool": False,
        "enterprise.enforce_policy": False,
        "enterprise.blocked_tools": "",
        "enterprise.blocked_categories": "",
        "enterprise.require_tool_allowlist": False,
    },
    "enterprise": {
        "agent.enforce_grounded_synthesis": True,
        "agent.enforce_claim_content_validation": True,
        "agent.require_key_evidence_section": True,
        "agent.allow_creative_hypotheses": False,
        "agent.quality_gate_enabled": True,
        "agent.quality_gate_strict": True,
        "agent.quality_gate_repair_retries": 2,
        "agent.quality_gate_repair_non_strict": True,
        "agent.synthesis_style": "standard",
        "agent.memory_retrieval_enabled": True,
        "agent.enable_claude_code_tool": False,
        "enterprise.enforce_policy": True,
        "enterprise.blocked_tools": "shell.run,files.delete_file,claude.code",
        "enterprise.blocked_categories": "",
        "enterprise.require_tool_allowlist": False,
    },
    "pharma": {
        "agent.enforce_grounded_synthesis": True,
        "agent.enforce_claim_content_validation": True,
        "agent.require_key_evidence_section": True,
        "agent.allow_creative_hypotheses": False,
        "agent.quality_gate_enabled": True,
        "agent.quality_gate_strict": True,
        "agent.quality_gate_repair_retries": 2,
        "agent.quality_gate_repair_non_strict": True,
        "agent.quality_gate_min_next_steps": 3,
        "agent.quality_gate_max_next_steps": 3,
        "agent.synthesis_style": "pharma",
        "agent.memory_retrieval_enabled": True,
        "agent.enable_claude_code_tool": False,
        "enterprise.enforce_policy": False,
        "enterprise.blocked_tools": "",
        "enterprise.blocked_categories": "",
        "enterprise.require_tool_allowlist": False,
    },
}


API_KEYS = {
    "api.ibm_rxn_key": {
        "name": "IBM RXN",
        "env_var": "IBM_RXN_API_KEY",
        "description": "AI-powered retrosynthesis (chemistry.retrosynthesis)",
        "url": "https://rxn.res.ibm.com",
        "free": True,
    },
    "api.lens_key": {
        "name": "Lens.org",
        "env_var": "LENS_API_KEY",
        "description": "Patent search (literature.patent_search)",
        "url": "https://www.lens.org/lens/user/subscriptions",
        "free": True,
    },
    "notification.sendgrid_api_key": {
        "name": "SendGrid",
        "env_var": "SENDGRID_API_KEY",
        "description": "Email sending (notification.send_email)",
        "url": "https://sendgrid.com",
        "free": True,
    },
    "compute.lambda_api_key": {
        "name": "Lambda Labs",
        "env_var": "LAMBDA_API_KEY",
        "description": "GPU compute jobs (compute.submit_job)",
        "url": "https://cloud.lambdalabs.com",
        "free": False,
    },
    "compute.runpod_api_key": {
        "name": "RunPod",
        "env_var": "RUNPOD_API_KEY",
        "description": "GPU compute jobs (compute.submit_job)",
        "url": "https://www.runpod.io",
        "free": False,
    },
}


def _validate_config(config_dict: dict) -> list[str]:
    """Validate a config dict and return a list of warning/error messages.

    Checks:
    - Type correctness (numeric, bool, string)
    - Range validity (positive integers, minimums)
    - Interdependency warnings (pharma + quality_gate_strict)
    - Unknown keys (possible typos)
    """
    warnings: list[str] = []

    # --- Unknown keys ---
    known_keys = set(DEFAULTS.keys())
    for key in config_dict:
        if key not in known_keys:
            warnings.append(f"Unknown config key '{key}' (possible typo)")

    # --- Type checks ---
    for key, value in config_dict.items():
        if key not in DEFAULTS or value is None:
            continue
        default = DEFAULTS[key]
        if default is None:
            continue

        expected_type = type(default)
        if expected_type == bool:
            if not isinstance(value, bool):
                warnings.append(
                    f"Type error: '{key}' should be bool, got {type(value).__name__} ({value!r})"
                )
        elif expected_type == int:
            if not isinstance(value, (int, float)):
                warnings.append(
                    f"Type error: '{key}' should be int, got {type(value).__name__} ({value!r})"
                )
        elif expected_type == float:
            if not isinstance(value, (int, float)):
                warnings.append(
                    f"Type error: '{key}' should be float, got {type(value).__name__} ({value!r})"
                )
        elif expected_type == str:
            if not isinstance(value, str):
                warnings.append(
                    f"Type error: '{key}' should be str, got {type(value).__name__} ({value!r})"
                )

    # --- Range checks ---
    def _check_positive_int(key: str, label: str):
        val = config_dict.get(key)
        if val is not None and isinstance(val, (int, float)) and val <= 0:
            warnings.append(f"Range error: '{key}' ({label}) must be > 0, got {val}")

    def _check_min(key: str, minimum: int, label: str):
        val = config_dict.get(key)
        if val is not None and isinstance(val, (int, float)) and val < minimum:
            warnings.append(
                f"Range error: '{key}' ({label}) must be >= {minimum}, got {val}"
            )

    _check_positive_int("agent.max_iterations", "max iterations")
    _check_positive_int("agent.executor_max_retries", "executor max retries")
    _check_positive_int("agent.executor_loop_limit", "executor loop limit")
    _check_positive_int("agent.parallel_max_threads", "parallel max threads")
    _check_min("agent.synthesis_max_tokens", 512, "synthesis max tokens")
    _check_min("sandbox.timeout", 1, "sandbox timeout")

    # --- Interdependency checks ---
    profile = config_dict.get("agent.profile")
    if profile == "pharma":
        qg_strict = config_dict.get(
            "agent.quality_gate_strict",
            DEFAULTS.get("agent.quality_gate_strict"),
        )
        if qg_strict is False or qg_strict == 0:
            warnings.append(
                "Interdependency warning: profile is 'pharma' but "
                "agent.quality_gate_strict is false (recommended: true)"
            )

    return warnings


class Config:
    """ct configuration manager."""

    def __init__(self, data: dict = None):
        self._data = data or {}
        self._env_loaded_keys: set[str] = set()

    def __repr__(self) -> str:
        """Safe repr that masks API keys and secrets."""
        safe = {}
        for k, v in self._data.items():
            if ("api_key" in k or "secret" in k or k.startswith("api.")) and v:
                safe[k] = str(v)[:4] + "..." if len(str(v)) > 4 else "***"
            else:
                safe[k] = v
        return f"Config({safe})"

    @classmethod
    def load(cls) -> "Config":
        """Load config from disk, creating defaults if needed."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    logger.warning(
                        "Invalid config format in %s (expected JSON object), ignoring file",
                        CONFIG_FILE,
                    )
                    data = {}
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to read config file %s: %s", CONFIG_FILE, exc)
                data = {}
        else:
            data = {}

        # Migrate legacy global output dir default to workspace-local output dir.
        legacy_output_dir = str(Path.home() / ".ct" / "outputs")
        if data.get("sandbox.output_dir") == legacy_output_dir:
            data["sandbox.output_dir"] = str(Path.cwd() / "outputs")

        # Check environment variables
        env_mappings = {
            "ANTHROPIC_API_KEY": "llm.api_key",
            "OPENAI_API_KEY": "llm.openai_api_key",
            "CT_DATA_DIR": "data.base",
            "CT_LLM_PROVIDER": "llm.provider",
            "CT_LLM_MODEL": "llm.model",
            "IBM_RXN_API_KEY": "api.ibm_rxn_key",
            "LENS_API_KEY": "api.lens_key",
            "SENDGRID_API_KEY": "notification.sendgrid_api_key",
            "LAMBDA_API_KEY": "compute.lambda_api_key",
            "RUNPOD_API_KEY": "compute.runpod_api_key",
            "CT_DATA_ENDPOINT": "api.data_endpoint",
            "CLUE_API_KEY": "api.clue_key",
        }
        for env_var, config_key in env_mappings.items():
            val = os.environ.get(env_var)
            if val and config_key not in data:
                data[config_key] = val

        cfg = cls(data)
        # Track keys loaded from environment so they're masked in __repr__/logs
        cfg._env_loaded_keys = {
            config_key for env_var, config_key in env_mappings.items()
            if os.environ.get(env_var) and config_key in data
        }

        # Run validation and log warnings (never crash)
        issues = _validate_config(data)
        for issue in issues:
            logger.warning("Config validation: %s", issue)

        return cfg

    def validate(self) -> list[str]:
        """Run schema validation on current config data. Returns list of issues."""
        return _validate_config(self._data)

    def save(self):
        """Save config to disk."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(self._data, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value, falling back to defaults."""
        return self._data.get(key, DEFAULTS.get(key, default))

    def set(self, key: str, value: Any):
        """Set a config value."""
        if key == "agent.profile":
            profile = str(value).strip().lower()
            if profile not in AGENT_PROFILE_PRESETS:
                valid = ", ".join(sorted(AGENT_PROFILE_PRESETS.keys()))
                raise ValueError(
                    f"Invalid agent.profile '{value}'. Valid profiles: {valid}"
                )
            for preset_key, preset_val in AGENT_PROFILE_PRESETS[profile].items():
                self._data[preset_key] = preset_val
            self._data["agent.profile"] = profile
            return

        # Type coercion
        if key in DEFAULTS and DEFAULTS[key] is not None:
            expected_type = type(DEFAULTS[key])
            if expected_type == bool:
                value = value.lower() in ("true", "1", "yes") if isinstance(value, str) else bool(value)
            elif expected_type == float:
                value = float(value)
            elif expected_type == int:
                value = int(value)

        self._data[key] = value

    def llm_api_key(self, provider: Optional[str] = None) -> Optional[str]:
        """Get the best API key for the selected provider."""
        provider = (provider or self.get("llm.provider", "anthropic")).lower()
        if provider == "openai":
            return self.get("llm.openai_api_key") or self.get("llm.api_key")
        return self.get("llm.api_key")

    def llm_preflight_issue(self) -> Optional[str]:
        """Return a human-readable LLM config issue, or None when ready."""
        provider_raw = self.get("llm.provider", "anthropic")
        provider = str(provider_raw or "").strip().lower()
        if not provider:
            return "llm.provider is empty. Set it with: ct config set llm.provider anthropic"

        if provider not in VALID_LLM_PROVIDERS:
            valid = ", ".join(sorted(VALID_LLM_PROVIDERS))
            return (
                f"Unsupported llm.provider '{provider}'. "
                f"Valid providers: {valid}. Set it with: ct config set llm.provider <provider>"
            )

        if provider in {"local", "gluelm"}:
            if not self.get("llm.model"):
                return (
                    f"llm.model is required for provider '{provider}'. "
                    "Set it with: ct config set llm.model <model-id-or-path>"
                )
            return None

        if self.llm_api_key(provider):
            return None

        # Azure AI Foundry: Foundry-specific env vars are valid Anthropic auth
        if provider == "anthropic" and (
            os.environ.get("ANTHROPIC_FOUNDRY_API_KEY")
            or os.environ.get("ANTHROPIC_FOUNDRY_RESOURCE")
        ):
            return None

        if provider == "openai":
            return (
                "OpenAI API key not configured. Set OPENAI_API_KEY or run:\n"
                "  ct config set llm.openai_api_key <key>"
            )

        return (
            "Anthropic API key not configured. Set ANTHROPIC_API_KEY or run:\n"
            "  ct config set llm.api_key <key>\n"
            "For Azure AI Foundry: set ANTHROPIC_FOUNDRY_API_KEY and "
            "ANTHROPIC_FOUNDRY_RESOURCE"
        )

    def keys_table(self) -> Table:
        """Render API key status as a rich table."""
        table = Table(title="API Keys", caption="Set: ct config set <key> <value>  |  Or: export ENV_VAR=<value>")
        table.add_column("Service", style="bold")
        table.add_column("Status")
        table.add_column("Unlocks", style="dim")
        table.add_column("Config Key", style="cyan dim")
        table.add_column("Sign Up", style="dim")

        for config_key, info in API_KEYS.items():
            val = self.get(config_key)
            if val:
                status = "[green]configured[/green]"
            else:
                status = "[red]not set[/red]"

            free_tag = " (free)" if info["free"] else ""
            table.add_row(
                info["name"],
                status,
                info["description"],
                config_key,
                info["url"] + free_tag,
            )

        return table

    def to_table(self) -> Table:
        """Render config as a rich table."""
        table = Table(title="ct Configuration")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Source", style="dim")

        all_keys = sorted(set(list(DEFAULTS.keys()) + list(self._data.keys())))
        for key in all_keys:
            if key in self._data:
                val = self._data[key]
                source = "config"
            elif key in DEFAULTS:
                val = DEFAULTS[key]
                source = "default"
            else:
                continue

            # Mask sensitive values (API keys, secrets)
            display_val = str(val)
            is_sensitive = "api_key" in key or "secret" in key or key.startswith("api.")
            if is_sensitive and val and len(str(val)) > 8:
                display_val = str(val)[:4] + "..." + str(val)[-4:]

            table.add_row(key, display_val, source)

        return table
