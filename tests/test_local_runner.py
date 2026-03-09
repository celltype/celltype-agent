"""Tests for local GPU runner — mock Docker subprocess calls."""

from dataclasses import dataclass
from unittest.mock import patch, MagicMock
import json
import pytest


@pytest.fixture(autouse=True)
def mock_manifest_for_runner():
    """Mock manifest lookups so weight check doesn't block tests."""
    with patch("ct.cloud.manifest.get_tool_config", return_value=None):
        yield


@dataclass
class FakeTool:
    name: str = "structure.esmfold"
    requires_gpu: bool = True
    gpu_profile: str = "structure"
    estimated_cost: float = 0.05
    docker_image: str = "celltype/esmfold:latest"
    min_vram_gb: int = 0
    min_ram_gb: int = 0
    cpu_only: bool = False
    num_gpus: int = 1

    def run(self, **kwargs):
        return {"summary": "local run"}


class TestLocalRunner:
    """Test Docker-based local GPU execution."""

    def test_run_with_output(self, tmp_path):
        from ct.cloud.local_runner import LocalRunner

        runner = LocalRunner(workspace=tmp_path)

        # Mock docker image inspect (image exists)
        # Mock docker run (success)
        with patch("ct.cloud.local_runner.subprocess.run") as mock_run:
            # First call: docker image inspect (image exists)
            inspect_result = MagicMock(returncode=0)
            # Second call: docker run (success)
            run_result = MagicMock(returncode=0, stdout="done", stderr="")

            mock_run.side_effect = [inspect_result, run_result]

            # Pre-create output file
            session_dir = runner._session_dir
            session_dir.mkdir(parents=True, exist_ok=True)
            output_file = session_dir / "output.json"
            output_file.write_text(json.dumps({"summary": "Structure predicted"}))

            result = runner.run(FakeTool(), sequence="MKWVTF")
            assert result["summary"] == "Structure predicted"

    def test_run_container_failure(self, tmp_path):
        from ct.cloud.local_runner import LocalRunner

        runner = LocalRunner(workspace=tmp_path)

        with patch("ct.cloud.local_runner.subprocess.run") as mock_run:
            inspect_result = MagicMock(returncode=0)
            run_result = MagicMock(returncode=1, stdout="", stderr="CUDA out of memory")
            mock_run.side_effect = [inspect_result, run_result]

            result = runner.run(FakeTool(), sequence="MKWVTF")
            assert "[Failed]" in result["summary"]

    def test_run_no_docker_image(self, tmp_path):
        from ct.cloud.local_runner import LocalRunner

        runner = LocalRunner(workspace=tmp_path)
        tool = FakeTool(docker_image="")

        result = runner.run(tool, sequence="MKWVTF")
        assert result["skipped"] is True
        assert result["reason"] == "no_docker_image"

    def test_image_pull_when_missing(self, tmp_path):
        from ct.cloud.local_runner import LocalRunner

        runner = LocalRunner(workspace=tmp_path)

        with patch("ct.cloud.local_runner.subprocess.run") as mock_run:
            # Image inspect fails (not found)
            inspect_result = MagicMock(returncode=1)
            # Pull succeeds
            pull_result = MagicMock(returncode=0)
            # Docker run succeeds
            run_result = MagicMock(returncode=0, stdout="", stderr="")

            mock_run.side_effect = [inspect_result, pull_result, run_result]

            runner.run(FakeTool(), sequence="MKWVTF")

            # Should have called docker pull
            calls = mock_run.call_args_list
            assert any("pull" in str(c) for c in calls)

    def test_gpu_passthrough_flags(self, tmp_path):
        from ct.cloud.local_runner import LocalRunner

        runner = LocalRunner(workspace=tmp_path)

        with patch("ct.cloud.local_runner.subprocess.run") as mock_run:
            inspect_result = MagicMock(returncode=0)
            run_result = MagicMock(returncode=0, stdout="", stderr="")
            mock_run.side_effect = [inspect_result, run_result]

            runner.run(FakeTool(), sequence="MKWVTF")

            # Verify docker run was called with --gpus all
            docker_run_call = mock_run.call_args_list[1]
            cmd = docker_run_call[0][0]
            assert "--gpus" in cmd
            assert "all" in cmd

    def test_workspace_mounting(self, tmp_path):
        from ct.cloud.local_runner import LocalRunner

        runner = LocalRunner(workspace=tmp_path)

        with patch("ct.cloud.local_runner.subprocess.run") as mock_run:
            inspect_result = MagicMock(returncode=0)
            run_result = MagicMock(returncode=0, stdout="", stderr="")
            mock_run.side_effect = [inspect_result, run_result]

            runner.run(FakeTool(), sequence="MKWVTF")

            # Verify workspace is mounted
            docker_run_call = mock_run.call_args_list[1]
            cmd = docker_run_call[0][0]
            assert "-v" in cmd
            # Should contain workspace path
            assert any("/workspace" in str(c) for c in cmd)

    def test_cpu_only_no_gpu_flags(self, tmp_path):
        """CPU-only tools should not pass --gpus, should pass --memory."""
        from ct.cloud.local_runner import LocalRunner

        runner = LocalRunner(workspace=tmp_path)
        tool = FakeTool(
            name="genomics.msa_search",
            docker_image="celltype/msa-search:latest",
            requires_gpu=False,
            cpu_only=True,
            min_ram_gb=64,
            num_gpus=0,
        )

        with patch("ct.cloud.local_runner.subprocess.run") as mock_run:
            inspect_result = MagicMock(returncode=0)
            run_result = MagicMock(returncode=0, stdout="", stderr="")
            mock_run.side_effect = [inspect_result, run_result]

            runner.run(tool, sequence="MKWVTF")

            docker_run_call = mock_run.call_args_list[1]
            cmd = docker_run_call[0][0]
            assert "--gpus" not in cmd
            assert "--memory" in cmd
            assert "64g" in cmd

    def test_multi_gpu_device_flags(self, tmp_path):
        """Multi-GPU tools should pass --gpus with device IDs."""
        from ct.cloud.local_runner import LocalRunner

        runner = LocalRunner(workspace=tmp_path)
        tool = FakeTool(
            name="genomics.evo2",
            docker_image="celltype/evo2:latest",
            num_gpus=2,
        )

        with patch("ct.cloud.local_runner.subprocess.run") as mock_run:
            inspect_result = MagicMock(returncode=0)
            run_result = MagicMock(returncode=0, stdout="", stderr="")
            mock_run.side_effect = [inspect_result, run_result]

            runner.run(tool, dna_sequence="ATCG")

            docker_run_call = mock_run.call_args_list[1]
            cmd = docker_run_call[0][0]
            assert "--gpus" in cmd
            assert "device=" in str(cmd)

    def test_cleanup_old_sessions(self, tmp_path):
        from ct.cloud.local_runner import LocalRunner
        import os
        import time

        runner = LocalRunner(workspace=tmp_path)

        # Create old session dir
        old_dir = tmp_path / "old-session"
        old_dir.mkdir()
        (old_dir / "test.json").write_text("{}")

        # Make it old by changing mtime
        old_time = time.time() - 25 * 3600  # 25 hours ago
        os.utime(old_dir, (old_time, old_time))

        removed = runner.cleanup(max_age_hours=24)
        assert removed == 1
        assert not old_dir.exists()

    @pytest.mark.parametrize(
        ("tool_name", "arg_name"),
        [
            ("design.rfdiffusion", "target_pdb"),
            ("design.proteinmpnn", "backbone_pdb"),
            ("structure.diffdock", "protein_pdb"),
        ],
    )
    def test_run_inlines_local_structure_file_args(self, tmp_path, tool_name, arg_name):
        from ct.cloud.local_runner import LocalRunner

        runner = LocalRunner(workspace=tmp_path)
        tool = FakeTool(name=tool_name)
        pdb_path = tmp_path / "input.pdb"
        pdb_content = "ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00  0.00           C\nEND\n"
        pdb_path.write_text(pdb_content, encoding="utf-8")

        with patch("ct.cloud.local_runner.subprocess.run") as mock_run:
            inspect_result = MagicMock(returncode=0)
            run_result = MagicMock(returncode=0, stdout="", stderr="")
            mock_run.side_effect = [inspect_result, run_result]

            runner.run(tool, **{arg_name: str(pdb_path)})

        input_payload = json.loads((runner._session_dir / "input.json").read_text())
        assert input_payload[arg_name] == pdb_content
