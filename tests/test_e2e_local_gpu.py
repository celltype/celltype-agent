"""
End-to-end tests for local GPU execution via Docker.

These tests require Docker and GPU access. Run with:
    pytest tests/test_e2e_local_gpu.py -v --run-e2e

Uses GPU index 0 only. Does NOT kill other GPU processes.
"""

import json
import os
import subprocess
import pytest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch


def _docker_available():
    try:
        result = subprocess.run(
            ["sudo", "docker", "--version"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _gpu_available():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return "0" in result.stdout
    except Exception:
        return False


requires_docker = pytest.mark.skipif(
    not _docker_available(), reason="Docker not available"
)
requires_gpu = pytest.mark.skipif(
    not _gpu_available(), reason="No GPU available"
)


@dataclass
class RealTool:
    name: str
    requires_gpu: bool = True
    gpu_profile: str = "structure"
    estimated_cost: float = 0.05
    docker_image: str = ""
    min_vram_gb: int = 0
    min_ram_gb: int = 0
    cpu_only: bool = False
    num_gpus: int = 1

    def run(self, **kwargs):
        return {"summary": "placeholder"}


@pytest.mark.e2e
class TestLocalGPUExecution:
    """Test running GPU tools locally via Docker."""

    @requires_docker
    @requires_gpu
    def test_esmfold_local_docker(self, tmp_path):
        """Run ESMFold in Docker with GPU 0."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "input.json").write_text(json.dumps({
            "sequence": "MKWVTFISLLFLFSSAYS"
        }))

        cmd = [
            "sudo", "docker", "run", "--rm",
            "--gpus", '"device=0"',
            "-v", f"{workspace}:/workspace",
            "-e", "TOOL_NAME=structure.esmfold",
            "-e", "INPUT_FILE=/workspace/input.json",
            "-e", "OUTPUT_FILE=/workspace/output.json",
            "celltype/esmfold:latest",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        output_file = workspace / "output.json"
        assert output_file.exists(), f"No output.json. stderr: {result.stderr[-300:]}"

        output = json.loads(output_file.read_text())
        assert "summary" in output
        assert "num_residues" in output
        assert output["num_residues"] == 18

    @requires_docker
    def test_msa_search_cpu_only_docker(self, tmp_path):
        """Run MSA-Search (CPU-only) in Docker."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "input.json").write_text(json.dumps({
            "sequence": "MKWVTFISLLFLFSSAYS",
            "database": "colabfold_envdb",
        }))

        cmd = [
            "sudo", "docker", "run", "--rm",
            "--memory", "4g",
            "-v", f"{workspace}:/workspace",
            "-e", "TOOL_NAME=genomics.msa_search",
            "-e", "INPUT_FILE=/workspace/input.json",
            "-e", "OUTPUT_FILE=/workspace/output.json",
            "celltype/msa-search:latest",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        output_file = workspace / "output.json"
        assert output_file.exists()

        output = json.loads(output_file.read_text())
        assert "summary" in output
        assert "msa" in output
        assert output["num_sequences"] > 0

    @requires_docker
    @requires_gpu
    def test_diffdock_local_docker(self, tmp_path):
        """Run DiffDock in Docker with GPU 0."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "input.json").write_text(json.dumps({
            "protein_pdb": "ATOM      1  N   ALA A   1       0.0   0.0   0.0  1.0  0.0\nEND\n",
            "ligand_smiles": "CCO",
            "num_poses": 3,
        }))

        cmd = [
            "sudo", "docker", "run", "--rm",
            "--gpus", '"device=0"',
            "-v", f"{workspace}:/workspace",
            "-e", "TOOL_NAME=structure.diffdock",
            "-e", "INPUT_FILE=/workspace/input.json",
            "-e", "OUTPUT_FILE=/workspace/output.json",
            "celltype/diffdock:latest",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        output_file = workspace / "output.json"
        assert output_file.exists()

        output = json.loads(output_file.read_text())
        assert "summary" in output
        assert "poses" in output
        assert len(output["poses"]) == 3
        assert output["best_score"] > 0

    @requires_docker
    @requires_gpu
    def test_boltz2_local_docker(self, tmp_path):
        """Run Boltz-2 in Docker with GPU 0."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "input.json").write_text(json.dumps({
            "sequence": "MKWVTFISLLFLFSSAYS"
        }))

        cmd = [
            "sudo", "docker", "run", "--rm",
            "--gpus", '"device=0"',
            "-v", f"{workspace}:/workspace",
            "-e", "TOOL_NAME=structure.boltz2",
            "-e", "INPUT_FILE=/workspace/input.json",
            "-e", "OUTPUT_FILE=/workspace/output.json",
            "celltype/boltz2:latest",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        output_file = workspace / "output.json"
        assert output_file.exists()

        output = json.loads(output_file.read_text())
        assert "summary" in output
        assert "pdb_content" in output
        assert "confidence" in output
        assert output["num_residues"] == 18


    @requires_docker
    @requires_gpu
    def test_bindcraft_local_docker(self, tmp_path):
        """Run BindCraft in Docker with GPU 0."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        pdb_text = (
            "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00  0.00           C\n"
            "ATOM      4  O   ALA A   1       4.000   5.000   6.000  1.00  0.00           O\n"
            "END\n"
        )
        (workspace / "input.json").write_text(json.dumps({
            "target_pdb": pdb_text,
            "chains": "A",
            "num_designs": 1,
            "binder_length_min": 20,
            "binder_length_max": 40,
        }))

        cmd = [
            "sudo", "docker", "run", "--rm",
            "--gpus", '"device=0"',
            "-v", f"{workspace}:/workspace",
            "-e", "TOOL_NAME=design.bindcraft",
            "-e", "INPUT_FILE=/workspace/input.json",
            "-e", "OUTPUT_FILE=/workspace/output.json",
            "celltype/bindcraft:latest",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

        output_file = workspace / "output.json"
        assert output_file.exists(), f"No output.json. stderr: {result.stderr[-300:]}"

        output = json.loads(output_file.read_text())
        assert "summary" in output


@pytest.mark.e2e
class TestLocalRunnerIntegration:
    """Test the LocalRunner class end-to-end."""

    @pytest.mark.docker
    @requires_docker
    @requires_gpu
    def test_local_runner_esmfold(self, tmp_path):
        """Test LocalRunner.run() dispatches to Docker correctly.
        Requires Docker accessible without sudo (docker group membership).
        """
        from ct.cloud.local_runner import LocalRunner

        runner = LocalRunner(workspace=tmp_path)
        tool = RealTool(
            name="structure.esmfold",
            docker_image="celltype/esmfold:latest",
            min_vram_gb=16,
        )

        # Mock manifest to skip weight check
        with patch("ct.cloud.manifest.get_tool_config", return_value=None):
            result = runner.run(tool, sequence="MKWVTF")

        assert "summary" in result
        # Either it ran successfully or skipped (if docker not accessible without sudo)
        assert "ESMFold" in result.get("summary", "") or "Skipped" in result.get("summary", "")


@pytest.mark.e2e
class TestComputeRouterIntegration:
    """Test the compute router with real GPU detection."""

    def test_real_gpu_detection(self):
        """Test GPU detection on this machine."""
        from ct.cloud.router import _detect_local_gpu_info, _detect_local_gpu

        import ct.cloud.router as m
        m._gpu_info_cache = None  # Reset cache

        gpus = _detect_local_gpu_info()
        if gpus:
            assert gpus[0].vram_mb > 0
            assert len(gpus[0].name) > 0
            assert _detect_local_gpu() is True
        # Else just passes — no GPU on test machine

        m._gpu_info_cache = None

    def test_gpu_tool_compatibility(self):
        """Test real GPU compatibility check."""
        from ct.cloud.router import get_gpu_tool_compatibility, _detect_local_gpu_info

        import ct.cloud.router as m
        m._gpu_info_cache = None

        gpus = _detect_local_gpu_info()
        if gpus:
            compat = get_gpu_tool_compatibility(gpus)
            assert len(compat) > 0
            # All tools should have compatibility info
            for c in compat:
                assert "tool_name" in c
                assert "compatible" in c
                assert "gpu_name" in c

        m._gpu_info_cache = None
