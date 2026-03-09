"""
End-to-end tests for compute router fallback behavior.

Tests that tools with insufficient local VRAM fall back to cloud correctly.
"""

import pytest
from dataclasses import dataclass
from unittest.mock import patch, MagicMock


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


def _reset_cache():
    import ct.cloud.router as m
    m._gpu_info_cache = None


class TestVRAMFallback:
    """Test that tools exceeding local VRAM fall back properly."""

    def test_openfold3_exceeds_24gb_gpu(self):
        """OpenFold3 needs 80GB VRAM — should fall back on 24GB GPU."""
        from ct.cloud.router import ComputeRouter, GPUInfo
        _reset_cache()

        mock_config = MagicMock()
        mock_config.get.side_effect = lambda k, d=None: {
            "compute.mode": "auto",
        }.get(k, d)

        router = ComputeRouter(config=mock_config)
        tool = FakeTool(name="structure.openfold3", min_vram_gb=80)

        with patch("ct.cloud.router._detect_local_gpu_info",
                    return_value=[GPUInfo(name="RTX 3090", vram_mb=24576)]):
            with patch("ct.cloud.auth.is_logged_in", return_value=True):
                with patch("ct.cloud.auth.get_token", return_value="tok"):
                    with patch("ct.cloud.client.CloudClient.submit_and_wait",
                               return_value={"summary": "cloud result"}) as cloud_mock:
                        result = router._route_local(tool)
                        assert result["summary"] == "cloud result"
                        cloud_mock.assert_called_once()

        _reset_cache()

    def test_esmfold_fits_24gb_gpu(self):
        """ESMFold needs 16GB — should run locally on 24GB GPU."""
        from ct.cloud.router import ComputeRouter, GPUInfo
        _reset_cache()

        mock_config = MagicMock()
        mock_config.get.side_effect = lambda k, d=None: {
            "compute.mode": "local",
        }.get(k, d)

        router = ComputeRouter(config=mock_config)
        tool = FakeTool(name="structure.esmfold", min_vram_gb=16)

        with patch("ct.cloud.router._detect_local_gpu_info",
                    return_value=[GPUInfo(name="RTX 3090", vram_mb=24576)]):
            with patch("ct.cloud.router._check_docker", return_value=(True, "")):
                with patch("ct.cloud.local_runner.LocalRunner.run",
                           return_value={"summary": "local ok"}) as local_mock:
                    result = router._route_local(tool)
                    assert result["summary"] == "local ok"
                    local_mock.assert_called_once()

        _reset_cache()

    def test_h100_runs_all_tools_locally(self):
        """H100 80GB should run all tools locally."""
        from ct.cloud.router import ComputeRouter, GPUInfo
        _reset_cache()

        mock_config = MagicMock()
        mock_config.get.side_effect = lambda k, d=None: {
            "compute.mode": "local",
        }.get(k, d)

        router = ComputeRouter(config=mock_config)

        # Even the biggest tool (OpenFold3 @ 80GB) fits
        tool = FakeTool(name="structure.openfold3", min_vram_gb=80)

        with patch("ct.cloud.router._detect_local_gpu_info",
                    return_value=[GPUInfo(name="H100", vram_mb=81920)]):
            with patch("ct.cloud.router._check_docker", return_value=(True, "")):
                with patch("ct.cloud.local_runner.LocalRunner.run",
                           return_value={"summary": "local h100"}) as local_mock:
                    result = router._route_local(tool)
                    assert result["summary"] == "local h100"

        _reset_cache()

    def test_local_mode_no_gpu_prompts(self):
        """In explicit local mode, no GPU should return needs_user_prompt."""
        from ct.cloud.router import ComputeRouter
        _reset_cache()

        mock_config = MagicMock()
        mock_config.get.side_effect = lambda k, d=None: {
            "compute.mode": "local",
        }.get(k, d)

        router = ComputeRouter(config=mock_config)
        tool = FakeTool(min_vram_gb=16)

        with patch("ct.cloud.router._detect_local_gpu_info", return_value=[]):
            result = router._route_local(tool)
            assert result.get("needs_user_prompt") is True

        _reset_cache()


class TestCPUOnlyFallback:
    """Test CPU-only tool routing (MSA-Search)."""

    def test_cpu_only_routes_to_cloud(self):
        """CPU-only tools should route to cloud in cloud mode."""
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
                           return_value={"summary": "msa cloud"}) as cloud_mock:
                    result = router.route(tool)
                    assert result["summary"] == "msa cloud"

        _reset_cache()

    def test_non_gpu_non_cpu_only_runs_directly(self):
        """Normal tools (not GPU, not cpu_only) should run directly."""
        from ct.cloud.router import ComputeRouter
        _reset_cache()

        mock_config = MagicMock()
        router = ComputeRouter(config=mock_config)
        tool = FakeTool(requires_gpu=False, cpu_only=False)

        result = router.route(tool)
        assert result["summary"] == "local run"

        _reset_cache()
