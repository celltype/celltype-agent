"""Tests for the compute router."""

from unittest.mock import patch, MagicMock
from dataclasses import dataclass

import pytest


def _reset_cache():
    import ct.cloud.router as m
    m._gpu_info_cache = None


@dataclass
class FakeTool:
    name: str = "test.gpu_tool"
    requires_gpu: bool = True
    gpu_profile: str = "structure"
    estimated_cost: float = 0.10
    docker_image: str = "celltype/test:latest"
    min_vram_gb: int = 0
    min_ram_gb: int = 0
    cpu_only: bool = False
    num_gpus: int = 1

    def run(self, **kwargs):
        return {"summary": "local run"}


class TestGPUInfoDetection:
    """Test nvidia-smi GPU detection with VRAM parsing."""

    def test_detect_gpu_info_single(self):
        from ct.cloud.router import _detect_local_gpu_info
        _reset_cache()

        with patch("ct.cloud.router.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="NVIDIA A100, 81920\n"
            )
            gpus = _detect_local_gpu_info()
            assert len(gpus) == 1
            assert gpus[0].name == "NVIDIA A100"
            assert gpus[0].vram_mb == 81920
            assert gpus[0].vram_gb == 80

        _reset_cache()

    def test_detect_gpu_info_multiple(self):
        from ct.cloud.router import _detect_local_gpu_info
        _reset_cache()

        with patch("ct.cloud.router.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="NVIDIA GeForce RTX 3090, 24576\nNVIDIA GeForce RTX 3060, 12288\n",
            )
            gpus = _detect_local_gpu_info()
            assert len(gpus) == 2
            assert gpus[0].vram_gb == 24
            assert gpus[1].vram_gb == 12

        _reset_cache()

    def test_detect_no_gpu(self):
        from ct.cloud.router import _detect_local_gpu_info
        _reset_cache()

        with patch("ct.cloud.router.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError
            gpus = _detect_local_gpu_info()
            assert gpus == []

        _reset_cache()

    def test_detect_local_gpu_bool_wrapper(self):
        from ct.cloud.router import _detect_local_gpu, GPUInfo
        _reset_cache()

        with patch("ct.cloud.router._detect_local_gpu_info", return_value=[
            GPUInfo(name="A100", vram_mb=81920)
        ]):
            assert _detect_local_gpu() is True

        with patch("ct.cloud.router._detect_local_gpu_info", return_value=[]):
            assert _detect_local_gpu() is False

    def test_detection_cached(self):
        import ct.cloud.router as m
        from ct.cloud.router import GPUInfo, _detect_local_gpu_info

        m._gpu_info_cache = [GPUInfo(name="cached", vram_mb=1024)]
        result = _detect_local_gpu_info()
        assert result[0].name == "cached"

        _reset_cache()


class TestGPUToolCompatibility:
    """Test per-tool VRAM compatibility check."""

    def test_compatibility_all_fit(self):
        from ct.cloud.router import get_gpu_tool_compatibility, GPUInfo

        gpus = [GPUInfo(name="A100", vram_mb=81920)]
        compat = get_gpu_tool_compatibility(gpus)
        assert all(c["compatible"] for c in compat)

    def test_compatibility_some_dont_fit(self):
        from ct.cloud.router import get_gpu_tool_compatibility, GPUInfo

        # All GPU tools require min 32GB — a 24GB GPU should fail
        gpus = [GPUInfo(name="RTX 4090", vram_mb=24576)]  # 24GB
        compat = get_gpu_tool_compatibility(gpus)

        by_name = {c["tool_name"]: c for c in compat}
        # All tools need 32GB, so nothing fits on 24GB
        assert all(not c["compatible"] for c in compat)


class TestComputeRouter:
    """Test routing logic."""

    def test_non_gpu_tool_runs_directly(self):
        from ct.cloud.router import ComputeRouter

        tool = FakeTool(requires_gpu=False)
        router = ComputeRouter()
        result = router.route(tool)
        assert result["summary"] == "local run"

    def test_cloud_mode_not_logged_in(self):
        from ct.cloud.router import ComputeRouter

        mock_config = MagicMock()
        mock_config.get.return_value = "cloud"

        router = ComputeRouter(config=mock_config)

        with patch("ct.cloud.auth.is_logged_in", return_value=False):
            result = router._route_cloud(FakeTool())
            assert result["skipped"] is True
            assert result["reason"] == "not_authenticated"

    def test_local_no_gpu_returns_needs_prompt(self):
        """Local mode, no GPU → returns needs_user_prompt for MCP handler."""
        from ct.cloud.router import ComputeRouter
        _reset_cache()

        mock_config = MagicMock()
        mock_config.get.side_effect = lambda k, d=None: {
            "compute.mode": "local",
        }.get(k, d)

        router = ComputeRouter(config=mock_config)

        with patch("ct.cloud.router._detect_local_gpu_info", return_value=[]):
            result = router._route_local(FakeTool())
            assert result.get("needs_user_prompt") is True
            assert "prompt_message" in result

        _reset_cache()

    def test_auto_no_gpu_falls_back_to_cloud(self):
        """Auto mode: no GPU → fall back to cloud transparently."""
        from ct.cloud.router import ComputeRouter
        _reset_cache()

        mock_config = MagicMock()
        mock_config.get.side_effect = lambda k, d=None: {
            "compute.mode": "auto",
        }.get(k, d)

        router = ComputeRouter(config=mock_config)

        with patch("ct.cloud.router._detect_local_gpu_info", return_value=[]):
            with patch("ct.cloud.auth.is_logged_in", return_value=True):
                with patch("ct.cloud.auth.get_token", return_value="tok"):
                    with patch("ct.cloud.client.CloudClient.submit_and_wait",
                               return_value={"summary": "cloud result"}):
                        result = router._route_local(FakeTool())
                        assert result["summary"] == "cloud result"

        _reset_cache()

    def test_route_cloud_for_tool_works(self):
        """route_cloud_for_tool dispatches to cloud (called by MCP after user approval)."""
        from ct.cloud.router import ComputeRouter
        _reset_cache()

        mock_config = MagicMock()
        mock_config.get.side_effect = lambda k, d=None: {
            "compute.mode": "local",
            "cloud.endpoint": "http://localhost:8000",
        }.get(k, d)

        router = ComputeRouter(config=mock_config)

        with patch("ct.cloud.auth.is_logged_in", return_value=True):
            with patch("ct.cloud.auth.get_token", return_value="tok"):
                with patch("ct.cloud.client.CloudClient.submit_and_wait",
                           return_value={"summary": "cloud result"}):
                    result = router.route_cloud_for_tool(FakeTool())
                    assert result["summary"] == "cloud result"

        _reset_cache()

    def test_local_insufficient_vram_returns_needs_prompt(self):
        """Local mode, VRAM too small → returns needs_user_prompt."""
        from ct.cloud.router import ComputeRouter, GPUInfo
        _reset_cache()

        mock_config = MagicMock()
        mock_config.get.side_effect = lambda k, d=None: {
            "compute.mode": "local",
        }.get(k, d)

        router = ComputeRouter(config=mock_config)
        tool = FakeTool(name="structure.alphafold3", min_vram_gb=40)

        with patch("ct.cloud.router._detect_local_gpu_info",
                    return_value=[GPUInfo(name="RTX 3090", vram_mb=24576)]):
            result = router._route_local(tool)
            assert result.get("needs_user_prompt") is True
            assert "40GB VRAM" in result["prompt_message"]

        _reset_cache()

    def test_auto_insufficient_vram_falls_back_to_cloud(self):
        """Auto mode: VRAM too small → fall back to cloud."""
        from ct.cloud.router import ComputeRouter, GPUInfo
        _reset_cache()

        mock_config = MagicMock()
        mock_config.get.side_effect = lambda k, d=None: {
            "compute.mode": "auto",
        }.get(k, d)

        router = ComputeRouter(config=mock_config)
        tool = FakeTool(min_vram_gb=40)

        with patch("ct.cloud.router._detect_local_gpu_info",
                    return_value=[GPUInfo(name="RTX 3090", vram_mb=24576)]):
            with patch("ct.cloud.auth.is_logged_in", return_value=True):
                with patch("ct.cloud.auth.get_token", return_value="tok"):
                    with patch("ct.cloud.client.CloudClient.submit_and_wait",
                               return_value={"summary": "cloud fallback"}):
                        result = router._route_local(tool)
                        assert result["summary"] == "cloud fallback"

        _reset_cache()

    def test_local_sufficient_vram_proceeds(self):
        from ct.cloud.router import ComputeRouter, GPUInfo
        _reset_cache()

        mock_config = MagicMock()
        mock_config.get.side_effect = lambda k, d=None: {
            "compute.mode": "local",
        }.get(k, d)

        router = ComputeRouter(config=mock_config)
        tool = FakeTool(min_vram_gb=8)

        with patch("ct.cloud.router._detect_local_gpu_info",
                    return_value=[GPUInfo(name="RTX 3090", vram_mb=24576)]):
            with patch("ct.cloud.router._check_docker", return_value=(True, "")):
                with patch("ct.cloud.local_runner.LocalRunner.run",
                           return_value={"summary": "local ok"}):
                    result = router._route_local(tool)
                    assert result["summary"] == "local ok"

        _reset_cache()

    def test_auto_mode_resolves_to_cloud(self):
        from ct.cloud.router import ComputeRouter
        _reset_cache()

        mock_config = MagicMock()
        mock_config.get.side_effect = lambda k, d=None: {
            "compute.mode": "auto",
        }.get(k, d)

        router = ComputeRouter(config=mock_config)

        with patch("ct.cloud.router._detect_local_gpu", return_value=False):
            assert router._resolve_mode() == "cloud"

        _reset_cache()

    def test_auto_mode_resolves_to_local(self):
        from ct.cloud.router import ComputeRouter
        _reset_cache()

        mock_config = MagicMock()
        mock_config.get.side_effect = lambda k, d=None: {
            "compute.mode": "auto",
        }.get(k, d)

        router = ComputeRouter(config=mock_config)

        with patch("ct.cloud.router._detect_local_gpu", return_value=True):
            assert router._resolve_mode() == "local"

        _reset_cache()

    def test_cpu_only_tool_routed(self):
        """CPU-only high-memory tools should be routed (not run directly)."""
        from ct.cloud.router import ComputeRouter
        _reset_cache()

        mock_config = MagicMock()
        mock_config.get.side_effect = lambda k, d=None: {
            "compute.mode": "cloud",
        }.get(k, d)

        router = ComputeRouter(config=mock_config)
        tool = FakeTool(
            name="genomics.msa_search",
            requires_gpu=False,
            cpu_only=True,
            min_ram_gb=64,
        )

        with patch("ct.cloud.auth.is_logged_in", return_value=True):
            with patch("ct.cloud.auth.get_token", return_value="tok"):
                with patch("ct.cloud.client.CloudClient.submit_and_wait",
                           return_value={"summary": "msa result"}):
                    result = router.route(tool)
                    assert result["summary"] == "msa result"

        _reset_cache()
