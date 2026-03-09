"""
Tool manifest loader — single source of truth for GPU/high-memory tool configuration.

Loads tool_manifest.yaml and provides programmatic access to tool configs,
allowed tools, GPU profiles, cost rates, and environment specs.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ct.cloud.manifest")

SUPPORTED_VERSIONS = {"1.0"}
VALID_GPU_TYPES = {"T4", "L4", "A10G", "A100", "H100"}
REQUIRED_SECTIONS = {"hardware", "environment", "cost", "execution"}

# Module-level cache
_manifest_cache: Optional[dict] = None


def _find_manifest_path() -> Path:
    """Locate tool_manifest.yaml, searching from package root upward."""
    # Try relative to this file first (src/ct/cloud/ -> repo root)
    pkg_dir = Path(__file__).resolve().parent.parent.parent.parent
    candidate = pkg_dir / "tool_manifest.yaml"
    if candidate.exists():
        return candidate

    # Try cwd
    cwd_candidate = Path.cwd() / "tool_manifest.yaml"
    if cwd_candidate.exists():
        return cwd_candidate

    raise FileNotFoundError(
        f"tool_manifest.yaml not found. Expected at {candidate} or {cwd_candidate}"
    )


def validate_manifest(manifest: dict) -> list[str]:
    """Validate manifest structure and return list of errors (empty = valid)."""
    errors = []

    # Check version
    version = manifest.get("version")
    if not version:
        errors.append("Missing top-level 'version' field")
    elif version not in SUPPORTED_VERSIONS:
        errors.append(
            f"Unsupported manifest version '{version}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_VERSIONS))}"
        )

    tools = manifest.get("tools")
    if not tools:
        errors.append("Missing or empty 'tools' section")
        return errors

    for tool_name, tool_config in tools.items():
        # Check required sections
        for section in REQUIRED_SECTIONS:
            if section not in tool_config:
                errors.append(f"{tool_name}: missing required section '{section}'")

        hw = tool_config.get("hardware", {})
        cpu_only = hw.get("cpu_only", False)
        gpu_type = hw.get("gpu_type")

        # CPU-only must not have gpu_type
        if cpu_only and gpu_type:
            errors.append(
                f"{tool_name}: cpu_only=true but gpu_type='{gpu_type}' is set"
            )

        # GPU tools must have valid gpu_type
        if not cpu_only and gpu_type and gpu_type not in VALID_GPU_TYPES:
            errors.append(
                f"{tool_name}: invalid gpu_type '{gpu_type}'. "
                f"Valid types: {', '.join(sorted(VALID_GPU_TYPES))}"
            )

        # Cost checks
        cost = tool_config.get("cost", {})
        if cost.get("per_second_base", 0) < 0:
            errors.append(f"{tool_name}: per_second_base must be non-negative")
        if cost.get("markup", 0) < 0:
            errors.append(f"{tool_name}: markup must be non-negative")

    return errors


def load_manifest(path: Optional[str] = None) -> dict:
    """Load and validate tool_manifest.yaml.

    Args:
        path: Optional explicit path to manifest file.

    Returns:
        Parsed manifest dict.

    Raises:
        FileNotFoundError: If manifest file not found.
        ValueError: If manifest is invalid.
    """
    global _manifest_cache

    if _manifest_cache is not None and path is None:
        return _manifest_cache

    import yaml

    manifest_path = Path(path) if path else _find_manifest_path()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    errors = validate_manifest(manifest)
    if errors:
        raise ValueError(
            f"Invalid manifest ({manifest_path}):\n" +
            "\n".join(f"  - {e}" for e in errors)
        )

    if path is None:
        _manifest_cache = manifest

    return manifest


def get_tool_config(tool_name: str) -> Optional[dict]:
    """Get configuration for a specific tool."""
    manifest = load_manifest()
    return manifest.get("tools", {}).get(tool_name)


def get_allowed_tools() -> set[str]:
    """Get set of all tool names defined in the manifest."""
    manifest = load_manifest()
    return set(manifest.get("tools", {}).keys())


def get_allowed_profiles() -> set[str]:
    """Get set of all GPU profiles used by manifest tools."""
    manifest = load_manifest()
    profiles = set()
    for tool_config in manifest.get("tools", {}).values():
        profile = tool_config.get("gpu_profile")
        if profile:
            profiles.add(profile)
    return profiles


def get_cost_per_second(tool_name: str) -> float:
    """Get cost per second for a tool (base * markup)."""
    config = get_tool_config(tool_name)
    if not config:
        return 0.0
    cost = config.get("cost", {})
    base = cost.get("per_second_base", 0.0)
    markup = cost.get("markup", 1.0)
    return base * markup


def get_environment_spec(tool_name: str) -> Optional[dict]:
    """Get environment specification for a tool."""
    config = get_tool_config(tool_name)
    if not config:
        return None
    return config.get("environment")


def clear_cache():
    """Clear the manifest cache (useful for testing)."""
    global _manifest_cache
    _manifest_cache = None
