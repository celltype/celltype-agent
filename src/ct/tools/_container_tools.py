"""
Container tool loader — auto-discovers and registers tools from subdirectories.

Each tool that runs in a container is a subdirectory of src/ct/tools/ containing:
  - tool.yaml:          Tool metadata (name, description, compute requirements, etc.)
  - Dockerfile:         How to build the container
  - implementation.py:  The actual inference code (runs inside the container)

To add a new tool:
  1. Create src/ct/tools/<tool_name>/
  2. Add tool.yaml, Dockerfile, implementation.py
  3. That's it — the tool is auto-discovered and registered.
"""

import logging
from pathlib import Path

from ct.tools import registry

logger = logging.getLogger("ct.tools.container_tools")

# Tool directories live alongside the .py tool modules
_TOOLS_DIR = Path(__file__).parent


def _find_tool_dirs() -> list[Path]:
    """Find all subdirectories containing tool.yaml."""
    dirs = []
    if _TOOLS_DIR.is_dir():
        for child in sorted(_TOOLS_DIR.iterdir()):
            if child.is_dir() and (child / "tool.yaml").exists():
                dirs.append(child)
    return dirs


def _load_tool_yaml(path: Path) -> dict:
    """Load and validate a tool.yaml file."""
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "name" not in data:
        raise ValueError(f"Invalid tool.yaml: {path}")
    return data


def _make_placeholder_fn(tool_name: str, params: dict):
    """Create a placeholder function for a container tool.

    The real execution happens via ComputeRouter -> LocalRunner (Docker) or CloudClient (Modal).
    This function body is never called for container tools — the router intercepts.
    """
    def _placeholder(**kwargs):
        return {"summary": f"{tool_name} (dispatched to compute)."}
    _placeholder.__name__ = tool_name.replace(".", "_")
    return _placeholder


def load_container_tools() -> int:
    """Discover and register all container tools from tool.yaml files.

    Returns number of tools loaded.
    """
    tool_dirs = _find_tool_dirs()
    loaded = 0

    for tool_dir in tool_dirs:
        try:
            config = _load_tool_yaml(tool_dir / "tool.yaml")
        except Exception as e:
            logger.warning("Failed to load %s/tool.yaml: %s", tool_dir.name, e)
            continue

        name = config["name"]
        compute = config.get("compute", {})
        params = config.get("parameters", {})
        fn = _make_placeholder_fn(name, params)

        registry.register(
            name=name,
            description=config.get("description", ""),
            category=config.get("category", name.split(".")[0]),
            parameters=params,
            usage_guide=config.get("usage_guide", ""),
            requires_gpu=compute.get("requires_gpu", False),
            gpu_profile=compute.get("gpu_profile", ""),
            estimated_cost=compute.get("estimated_cost_usd", 0.0),
            docker_image=config.get("docker_image", ""),
            min_vram_gb=compute.get("min_vram_gb", 32),
            cpu_only=compute.get("cpu_only", False),
            num_gpus=compute.get("num_gpus", 1),
        )(fn)

        loaded += 1

    if loaded:
        logger.debug("Loaded %d container tools", loaded)
    return loaded
