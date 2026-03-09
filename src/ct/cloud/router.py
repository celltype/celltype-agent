"""
Compute router — decides whether a GPU tool runs locally or on CellType Cloud.

Config-driven with auto-detection of local GPU availability and VRAM.
"""

import logging
import subprocess
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("ct.cloud.router")


# ─── GPU detection ─────────────────────────────────────────────

@dataclass
class GPUInfo:
    """Detected local GPU with name and VRAM."""
    name: str
    vram_mb: int

    @property
    def vram_gb(self) -> int:
        return self.vram_mb // 1024


# Session-scoped cache
_gpu_info_cache: Optional[list[GPUInfo]] = None


def _detect_local_gpu_info() -> list[GPUInfo]:
    """Detect local NVIDIA GPUs with name and VRAM.

    Returns list of GPUInfo (empty if no GPUs). Cached for session lifetime.
    """
    global _gpu_info_cache
    if _gpu_info_cache is not None:
        return _gpu_info_cache

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            _gpu_info_cache = []
            return _gpu_info_cache

        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                name = parts[0]
                try:
                    vram_mb = int(float(parts[1]))
                except ValueError:
                    vram_mb = 0
                gpus.append(GPUInfo(name=name, vram_mb=vram_mb))
        _gpu_info_cache = gpus
    except (FileNotFoundError, subprocess.TimeoutExpired):
        _gpu_info_cache = []

    logger.debug("Local GPU detection: %s", _gpu_info_cache)
    return _gpu_info_cache


def _detect_local_gpu() -> bool:
    """Check if any CUDA-capable NVIDIA GPU is available locally."""
    return len(_detect_local_gpu_info()) > 0


def get_gpu_tool_compatibility(gpus: list[GPUInfo] | None = None) -> list[dict]:
    """Check which GPU tools can run on the detected local GPU(s).

    Returns list of dicts with tool_name, min_vram_gb, compatible, gpu_name, gpu_vram_gb.
    """
    if gpus is None:
        gpus = _detect_local_gpu_info()

    if not gpus:
        return []

    best_gpu = max(gpus, key=lambda g: g.vram_mb)

    from ct.tools import registry, ensure_loaded
    ensure_loaded()

    results = []
    for tool in registry.list_tools():
        if not tool.requires_gpu:
            continue
        min_vram = getattr(tool, "min_vram_gb", 0)
        results.append({
            "tool_name": tool.name,
            "min_vram_gb": min_vram,
            "compatible": best_gpu.vram_gb >= min_vram if min_vram > 0 else True,
            "gpu_name": best_gpu.name,
            "gpu_vram_gb": best_gpu.vram_gb,
        })
    return results


# ─── Docker checks ─────────────────────────────────────────────

def _check_docker() -> tuple[bool, str]:
    """Check if Docker and NVIDIA Container Toolkit are available.

    Returns (ok, error_message).
    """
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False, (
                "Docker is required for local GPU execution. "
                "Install Docker and the NVIDIA Container Toolkit, or switch to cloud "
                "with `ct config set compute.mode cloud`."
            )
    except FileNotFoundError:
        return False, (
            "Docker is required for local GPU execution. "
            "Install Docker and the NVIDIA Container Toolkit, or switch to cloud "
            "with `ct config set compute.mode cloud`."
        )

    # Check for NVIDIA Container Toolkit
    try:
        result = subprocess.run(
            ["nvidia-container-cli", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        pass  # nvidia-container-cli not found but docker --gpus may still work

    return True, ""


# ─── Router ────────────────────────────────────────────────────

class ComputeRouter:
    """Routes GPU tool calls to cloud or local execution based on config."""

    def __init__(self, config=None):
        self._config = config
        self._cloud_client = None
        self._local_runner = None

    @property
    def config(self):
        if self._config is None:
            from ct.agent.config import Config
            self._config = Config.load()
        return self._config

    def _get_mode(self) -> str:
        """Get the compute mode: cloud, local, or auto."""
        return str(self.config.get("compute.mode", "cloud")).lower()

    def _resolve_mode(self) -> str:
        """Resolve 'auto' mode to either 'cloud' or 'local'."""
        mode = self._get_mode()
        if mode == "auto":
            return "local" if _detect_local_gpu() else "cloud"
        return mode

    def route(self, tool, **kwargs) -> dict:
        """Route a GPU or high-memory tool call to the appropriate execution backend."""
        # CPU-only high-memory tools are still dispatched via the router
        if not tool.requires_gpu and not getattr(tool, 'cpu_only', False):
            return tool.run(**kwargs)

        mode = self._resolve_mode()

        if mode == "local":
            return self._route_local(tool, **kwargs)
        else:
            return self._route_cloud(tool, **kwargs)

    def _route_cloud(self, tool, **kwargs) -> dict:
        """Route to CellType Cloud."""
        from ct.cloud.auth import is_logged_in, get_token

        if not is_logged_in():
            logger.warning(
                "[Skipped] %s — not logged in for CellType Cloud. "
                "Run `ct login` to access CellType Cloud, or "
                "`ct config set compute.mode local` to use your own GPU.",
                tool.name,
            )
            return {
                "summary": (
                    f"[Skipped] {tool.name} — GPU tool requires CellType Cloud authentication. "
                    "Run `ct login` to access CellType Cloud, or "
                    "`ct config set compute.mode local` to use your own GPU."
                ),
                "skipped": True,
                "reason": "not_authenticated",
            }

        try:
            token = get_token()
        except RuntimeError as e:
            logger.warning("[Skipped] %s — %s", tool.name, e)
            return {
                "summary": f"[Skipped] {tool.name} — {e}",
                "skipped": True,
                "reason": "token_expired",
            }

        # Dispatch to cloud client
        from ct.cloud.client import CloudClient

        if self._cloud_client is None:
            endpoint = self.config.get("cloud.endpoint", "https://api.celltype.com")
            self._cloud_client = CloudClient(endpoint=endpoint)

        try:
            return self._cloud_client.submit_and_wait(
                tool_name=tool.name,
                gpu_profile=tool.gpu_profile,
                estimated_cost=tool.estimated_cost,
                token=token,
                **kwargs,
            )
        except Exception as e:
            logger.warning("[Skipped] %s — Cloud error: %s", tool.name, e)
            return {
                "summary": (
                    f"[Skipped] {tool.name} — CellType Cloud error: {e}. "
                    "Try again later, or switch to local GPU with "
                    "`ct config set compute.mode local`."
                ),
                "skipped": True,
                "reason": "cloud_error",
            }

    def _try_cloud_fallback(self, tool, reason_msg: str, **kwargs) -> dict:
        """Handle a tool that can't run locally.

        - auto mode: transparent fallback to cloud
        - local mode: return a 'needs_user_prompt' result so the MCP handler
          can prompt the user on the main thread
        """
        mode = self._get_mode()

        if mode == "auto":
            return self._route_cloud(tool, **kwargs)

        # Explicit local mode — we can't prompt from here (worker thread).
        # Return a special result that the MCP handler will intercept.
        return {
            "summary": reason_msg,
            "needs_user_prompt": True,
            "prompt_message": reason_msg,
            "prompt_tool_name": tool.name,
        }

    def route_cloud_for_tool(self, tool, **kwargs) -> dict:
        """Public method to route a specific tool to cloud.

        Called by the MCP handler after the user approves cloud fallback.
        """
        return self._route_cloud(tool, **kwargs)

    def _route_local(self, tool, **kwargs) -> dict:
        """Route to local Docker execution with VRAM-aware fallback."""
        gpus = _detect_local_gpu_info()

        # ── No GPU at all ──
        if not gpus:
            mode = self._get_mode()
            if mode == "auto":
                return self._route_cloud(tool, **kwargs)
            return self._try_cloud_fallback(
                tool,
                f"No local GPU detected — {tool.name} cannot run locally.",
                **kwargs,
            )

        best_gpu = max(gpus, key=lambda g: g.vram_mb)

        # ── Check VRAM for this specific tool + input ──
        # Use input-aware estimate if available, otherwise fall back to min_vram_gb
        if hasattr(tool, "estimate_vram_gb"):
            estimated_vram = tool.estimate_vram_gb(**kwargs)
        else:
            estimated_vram = getattr(tool, "min_vram_gb", 0)
        if estimated_vram > 0 and best_gpu.vram_gb < estimated_vram:
            mode = self._get_mode()
            if mode == "auto":
                return self._route_cloud(tool, **kwargs)
            # mode=local but GPU too small — try cloud fallback
            reason = (
                f"{tool.name} needs ~{estimated_vram}GB VRAM for this input "
                f"(your {best_gpu.name} has {best_gpu.vram_gb}GB)."
            )
            return self._try_cloud_fallback(tool, reason, **kwargs)

        # ── GPU is sufficient — check Docker ──
        docker_ok, docker_err = _check_docker()
        if not docker_ok:
            logger.warning("[Skipped] %s — %s", tool.name, docker_err)
            return {
                "summary": f"[Skipped] {tool.name} — {docker_err}",
                "skipped": True,
                "reason": "docker_missing",
            }

        from ct.cloud.local_runner import LocalRunner

        if self._local_runner is None:
            self._local_runner = LocalRunner()

        try:
            return self._local_runner.run(tool, **kwargs)
        except Exception as e:
            logger.warning("[Skipped] %s — Local execution error: %s", tool.name, e)
            return {
                "summary": f"[Skipped] {tool.name} — Local execution error: {e}",
                "skipped": True,
                "reason": "local_error",
            }
