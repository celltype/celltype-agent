"""
Tool registry for ct.

Each tool is a Python function decorated with @tool that the agent can invoke.
Tools are organized by category (target, structure, chemistry, etc.).
"""

from dataclasses import dataclass, field
import importlib
import logging
from typing import Callable, Optional
from rich.table import Table


EXPERIMENTAL_CATEGORIES = frozenset({"compute", "cro"})
_TOOL_MODULES = (
    "target",
    "structure",
    "chemistry",
    "expression",
    "viability",
    "biomarker",
    "combination",
    "clinical",
    "intel",
    "translational",
    "regulatory",
    "pk",
    "report",
    "literature",
    "safety",
    "cro",
    "compute",
    "experiment",
    "notification",
    "code",
    "files",
    "claude",
    "network",
    "genomics",
    "statistics",
    "repurposing",
    "design",
    "singlecell",
    "protein",
    "imaging",
    "data_api",
    "parity",
    "ops",
    "dna",
    "omics",
    "shell",
    "cellxgene",
    "clue",
    "remote_data",
    # Container tools (esmfold, boltz2, etc.) loaded from tool.yaml by _container_tools.py
)


@dataclass
class Tool:
    """A registered tool that the agent can invoke."""
    name: str                  # e.g., "target.neosubstrate_score"
    description: str           # Human-readable description
    category: str              # e.g., "target", "structure", "chemistry"
    function: Callable         # The actual Python function
    parameters: dict = field(default_factory=dict)  # Parameter descriptions
    requires_data: list = field(default_factory=list)  # Required datasets
    usage_guide: str = ""      # When/why to use this tool (injected into planner prompt)
    requires_gpu: bool = False  # Whether this tool needs GPU compute
    gpu_profile: str = ""       # Modal compute class (e.g., "structure", "docking")
    estimated_cost: float = 0.0  # Estimated cost in USD per run
    docker_image: str = ""      # Docker image for local GPU execution
    min_vram_gb: int = 0        # Minimum GPU VRAM in GB (base, small input)
    min_ram_gb: int = 0         # Minimum system RAM in GB (for CPU-only tools)
    cpu_only: bool = False      # CPU-only high-memory tool (no GPU needed)
    num_gpus: int = 1           # Number of GPUs required
    vram_estimate_fn: Callable = None  # fn(**kwargs) -> int GB, estimates VRAM for given input

    def estimate_vram_gb(self, **kwargs) -> int:
        """Estimate VRAM needed for given input. Falls back to min_vram_gb."""
        if self.vram_estimate_fn:
            try:
                return self.vram_estimate_fn(**kwargs)
            except Exception:
                pass
        return self.min_vram_gb

    def run(self, **kwargs):
        """Execute the tool."""
        return self.function(**kwargs)


class ToolRegistry:
    """Central registry of all available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, category: str,
                 parameters: dict = None, requires_data: list = None,
                 usage_guide: str = "",
                 requires_gpu: bool = False, gpu_profile: str = "",
                 estimated_cost: float = 0.0, docker_image: str = "",
                 min_vram_gb: int = 0, min_ram_gb: int = 0,
                 cpu_only: bool = False, num_gpus: int = 1,
                 vram_estimate_fn: Callable = None):
        """Decorator to register a function as a tool."""
        def decorator(func):
            self._tools[name] = Tool(
                name=name,
                description=description,
                category=category,
                function=func,
                parameters=parameters or {},
                requires_data=requires_data or [],
                usage_guide=usage_guide,
                requires_gpu=requires_gpu,
                gpu_profile=gpu_profile,
                estimated_cost=estimated_cost,
                docker_image=docker_image,
                min_vram_gb=min_vram_gb,
                min_ram_gb=min_ram_gb,
                cpu_only=cpu_only,
                num_gpus=num_gpus,
                vram_estimate_fn=vram_estimate_fn,
            )
            return func
        return decorator

    def get_tool(self, name: str) -> Optional[Tool]:
        """Look up a tool by name."""
        return self._tools.get(name)

    def list_tools(self, category: str = None) -> list[Tool]:
        """List all tools, optionally filtered by category."""
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return sorted(tools, key=lambda t: t.name)

    def list_tools_table(self) -> Table:
        """Render tool list as a rich table."""
        table = Table(title="ct Tools")
        table.add_column("Tool", style="cyan")
        table.add_column("Status")
        table.add_column("Description")
        table.add_column("Data Required", style="dim")

        for tool in self.list_tools():
            data_str = ", ".join(tool.requires_data) if tool.requires_data else "-"
            if tool.name == "claude.code":
                status = "[yellow]guarded (opt-in)[/yellow]"
            elif tool.cpu_only and tool.min_ram_gb > 0:
                cost_str = f" ~${tool.estimated_cost:.2f}" if tool.estimated_cost else ""
                status = f"[magenta]cloud/cpu {tool.min_ram_gb}GB RAM{cost_str}[/magenta]"
            elif tool.requires_gpu and tool.num_gpus > 1:
                cost_str = f" ~${tool.estimated_cost:.2f}" if tool.estimated_cost else ""
                status = f"[magenta]cloud/multi-gpu x{tool.num_gpus}{cost_str}[/magenta]"
            elif tool.requires_gpu:
                cost_str = f" ~${tool.estimated_cost:.2f}" if tool.estimated_cost else ""
                status = f"[magenta]cloud/gpu{cost_str}[/magenta]"
            elif tool.category in EXPERIMENTAL_CATEGORIES:
                status = "[yellow]experimental / TODO[/yellow]"
            else:
                status = "[green]stable[/green]"
            table.add_row(tool.name, status, tool.description, data_str)

        return table

    def categories(self) -> list[str]:
        """List all tool categories."""
        return sorted(set(t.category for t in self._tools.values()))

    def tool_descriptions_for_llm(
        self,
        exclude_categories: set[str] | None = None,
        exclude_tools: set[str] | None = None,
    ) -> str:
        """Generate tool descriptions for the LLM planner."""
        exclude_categories = exclude_categories or set()
        exclude_tools = exclude_tools or set()
        lines = []
        for cat in self.categories():
            if cat in exclude_categories:
                continue
            cat_tools = [t for t in self.list_tools(cat) if t.name not in exclude_tools]
            if not cat_tools:
                continue
            lines.append(f"\n## {cat}")
            for tool in cat_tools:
                params = ", ".join(f"{k}: {v}" for k, v in tool.parameters.items())
                lines.append(f"- **{tool.name}**({params}): {tool.description}")
                if tool.usage_guide:
                    lines.append(f"  USE WHEN: {tool.usage_guide}")
                if tool.cpu_only and tool.min_ram_gb > 0:
                    cost_str = f"~${tool.estimated_cost:.2f}/run" if tool.estimated_cost else "variable"
                    lines.append(f"  CPU: High-memory tool requiring {tool.min_ram_gb}GB+ RAM ({cost_str})")
                elif tool.requires_gpu:
                    cost_str = f"~${tool.estimated_cost:.2f}/run" if tool.estimated_cost else "variable"
                    vram_str = f", {tool.min_vram_gb}GB+ VRAM" if tool.min_vram_gb else ""
                    gpu_str = f", {tool.num_gpus}x GPU" if tool.num_gpus > 1 else ""
                    lines.append(f"  GPU: Requires CellType Cloud or local GPU ({cost_str}{vram_str}{gpu_str})")
                if tool.category in EXPERIMENTAL_CATEGORIES:
                    lines.append("  NOTE: Experimental/TODO category. Outputs may be placeholder or limited.")
        return "\n".join(lines)


# Global registry instance
registry = ToolRegistry()


# Import tool modules to trigger registration
def _load_tools():
    """Import all tool modules to register them."""
    logger = logging.getLogger("ct.tools")
    errors = {}

    for module_name in _TOOL_MODULES:
        import_name = f"ct.tools.{module_name}"
        try:
            importlib.import_module(import_name)
        except Exception as exc:
            errors[module_name] = str(exc)
            logger.warning("Failed to load tool module %s: %s", import_name, exc)

    return errors


# Lazy loading — tools are registered on first access
_loaded = False
_load_errors: dict[str, str] = {}


def ensure_loaded():
    global _loaded
    global _load_errors
    if not _loaded:
        _load_errors = _load_tools()
        # Load container tools from src/ct/tools/*/tool.yaml
        try:
            from ct.tools._container_tools import load_container_tools
            load_container_tools()
        except Exception as e:
            _load_errors["container_tools"] = str(e)
        _loaded = True


def tool_load_errors() -> dict[str, str]:
    """Return module import failures from tool loading."""
    return dict(_load_errors)
