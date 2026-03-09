"""
Local GPU execution engine — runs Docker containers with --gpus all.

For users who have their own NVIDIA GPUs and want to run tools locally.
"""

import json
import logging
import os
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from ct.cloud.structure_inputs import inline_structure_file_args

logger = logging.getLogger("ct.cloud.local_runner")

DEFAULT_WORKSPACE = Path.home() / ".ct" / "gpu-workspace"


class LocalRunner:
    """Execute GPU tools locally via Docker with NVIDIA GPU passthrough."""

    def __init__(self, workspace: Optional[Path] = None):
        self.workspace = workspace or DEFAULT_WORKSPACE
        self._session_id = uuid.uuid4().hex[:12]
        self._session_dir = self.workspace / self._session_id

    def _ensure_workspace(self) -> Path:
        """Create session workspace directory."""
        self._session_dir.mkdir(parents=True, exist_ok=True)
        return self._session_dir

    def _ensure_image(self, image: str, tool=None) -> None:
        """Build Docker image from docker-images/ if not present locally."""
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            return  # Image exists

        # Image not found — build from docker-images/
        build_dir = self._find_docker_build_dir(tool)
        if build_dir:
            logger.info("Building image %s from %s...", image, build_dir)
            print(f"  Building {image} (first run — this may take a few minutes)...")

            # Copy tool_entrypoint.py if missing
            entrypoint = build_dir / "tool_entrypoint.py"
            if not entrypoint.exists():
                shared = build_dir.parent / "tool_entrypoint.py"
                if shared.exists():
                    import shutil
                    shutil.copy2(shared, entrypoint)

            build_result = subprocess.run(
                ["docker", "build", "-t", image, str(build_dir)],
                timeout=3600,
            )
            if build_result.returncode != 0:
                raise RuntimeError(f"Failed to build {image}")
        else:
            # No local build dir — try pulling from registry
            logger.info("Pulling image %s...", image)
            print(f"  Pulling image {image}...")
            subprocess.run(
                ["docker", "pull", image],
                timeout=600,
                check=True,
            )

    def _find_docker_build_dir(self, tool) -> "Optional[Path]":
        """Find the tool directory containing the Dockerfile.

        Tool directories live at src/ct/tools/<dir_name>/.
        """
        if tool is None:
            return None

        _DIR_MAP = {
            "structure.alphafold2_multimer": "alphafold2-multimer",
            "design.evo2_protein_design": "evo2-protein-design",
            "genomics.msa_search": "msa-search",
        }
        tool_name = getattr(tool, "name", "")
        dir_name = _DIR_MAP.get(tool_name, tool_name.split(".")[-1])

        # Tools live in src/ct/tools/<dir_name>/
        tools_dir = Path(__file__).parent.parent / "tools"
        for candidate in [
            tools_dir / dir_name,
            Path.cwd() / "src" / "ct" / "tools" / dir_name,
        ]:
            if candidate.is_dir() and (candidate / "Dockerfile").exists():
                return candidate

        return None

    def _get_timeout(self, tool) -> int:
        """Get per-tool timeout from manifest, with fallback."""
        try:
            from ct.cloud.manifest import get_tool_config
            config = get_tool_config(tool.name)
            if config:
                return config.get("execution", {}).get("timeout_s", 600)
        except Exception:
            pass
        return 600

    # Weight mounts are handled by _get_cache_mounts() which mounts
    # the host cache directories. No per-model mount logic needed.

    def _get_cache_mounts(self) -> list[str]:
        """Mount host model caches into the container.

        All tools download weights on first run and cache them on the host.
        Subsequent runs load from cache (no network required).
        """
        mounts = []
        home = Path.home()

        cache_mappings = {
            # HuggingFace cache (ESMFold, ESM2, Evo2, DiffDock)
            "HF_HOME": (home / ".cache" / "huggingface", "/root/.cache/huggingface"),
            # Torch hub cache
            "TORCH_HOME": (home / ".cache" / "torch", "/root/.cache/torch"),
            # Boltz-2 model cache
            "BOLTZ_CACHE": (home / ".cache" / "boltz", "/root/.boltz"),
            # RFDiffusion model weights
            "RFDIFFUSION_CACHE": (home / ".cache" / "rfdiffusion", "/root/.cache/rfdiffusion"),
            # OpenFold3 model weights
            "OPENFOLD3_CACHE": (home / ".cache" / "openfold3", "/root/.openfold3"),
            # OpenFold/AF2 model params
            "OPENFOLD_CACHE": (home / ".cache" / "openfold", "/root/.cache/openfold"),
            # ProteinMPNN weights
            "PROTEINMPNN_CACHE": (home / ".cache" / "proteinmpnn", "/root/.cache/proteinmpnn"),
            # DiffDock score/confidence models
            "DIFFDOCK_CACHE": (home / ".cache" / "diffdock", "/root/.cache/diffdock"),
        }

        for env_var, (host_path, container_path) in cache_mappings.items():
            resolved = Path(os.environ.get(env_var, host_path))
            resolved.mkdir(parents=True, exist_ok=True)
            mounts.extend(["-v", f"{resolved}:{container_path}"])

        return mounts

    def _get_gpu_flags(self, tool) -> list[str]:
        """Get GPU flags based on tool requirements."""
        cpu_only = getattr(tool, 'cpu_only', False)
        num_gpus = getattr(tool, 'num_gpus', 1)
        min_ram_gb = getattr(tool, 'min_ram_gb', 0)

        if cpu_only:
            # CPU-only: use memory limit, no GPU
            flags = []
            if min_ram_gb > 0:
                flags.extend(["--memory", f"{min_ram_gb}g"])
            return flags
        elif num_gpus > 1:
            # Multi-GPU: specify device IDs
            devices = ",".join(str(i) for i in range(num_gpus))
            return ["--gpus", f'"device={devices}"']
        else:
            # Standard single-GPU
            return ["--gpus", "all"]

    def run(self, tool, **kwargs) -> dict:
        """Execute a GPU tool in a Docker container.

        Args:
            tool: Tool object with docker_image, name, etc.
            **kwargs: Tool arguments.

        Returns:
            Tool result dict.
        """
        if not tool.docker_image:
            return {
                "summary": f"[Skipped] {tool.name} — no Docker image configured.",
                "skipped": True,
                "reason": "no_docker_image",
            }

        # Weights are downloaded automatically on first run inside the container.
        # The host cache dirs are mounted via _get_cache_mounts() so downloads persist.

        workspace = self._ensure_workspace()
        self._ensure_image(tool.docker_image, tool=tool)

        # Strip internal kwargs
        tool_args = {
            k: v for k, v in kwargs.items()
            if not k.startswith("_")
        }
        tool_args = inline_structure_file_args(tool.name, tool_args, logger=logger)

        # Write input args to workspace
        input_file = workspace / "input.json"
        input_file.write_text(json.dumps(tool_args))

        # Build Docker command with appropriate GPU/memory flags
        gpu_flags = self._get_gpu_flags(tool)
        cache_mounts = self._get_cache_mounts()
        timeout = self._get_timeout(tool)

        cmd = [
            "docker", "run", "--rm",
        ] + gpu_flags + [
            "-v", f"{workspace}:/workspace",
        ] + cache_mounts + [
            "-e", f"TOOL_NAME={tool.name}",
            "-e", "INPUT_FILE=/workspace/input.json",
            "-e", "OUTPUT_FILE=/workspace/output.json",
            tool.docker_image,
        ]

        logger.info("Running: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                return {
                    "summary": f"[Failed] {tool.name} — container exited with code {result.returncode}: {stderr[:500]}",
                    "error": stderr,
                }

            # Read output
            output_file = workspace / "output.json"
            if output_file.exists():
                output_data = json.loads(output_file.read_text())
                return output_data if isinstance(output_data, dict) else {"summary": str(output_data)}
            else:
                stdout = result.stdout.strip()
                return {"summary": stdout[:2000] if stdout else f"{tool.name} completed (no output file)."}

        except subprocess.TimeoutExpired:
            return {
                "summary": f"[Timeout] {tool.name} — container execution timed out ({timeout}s).",
                "skipped": True,
                "reason": "timeout",
            }

    def cleanup(self, max_age_hours: int = 24) -> int:
        """Remove old session directories. Returns count of removed dirs."""
        import time

        if not self.workspace.exists():
            return 0

        removed = 0
        now = time.time()
        cutoff = max_age_hours * 3600

        for d in self.workspace.iterdir():
            if d.is_dir():
                try:
                    age = now - d.stat().st_mtime
                    if age > cutoff:
                        import shutil
                        shutil.rmtree(d)
                        removed += 1
                except Exception as e:
                    logger.warning("Failed to clean up %s: %s", d, e)

        return removed
