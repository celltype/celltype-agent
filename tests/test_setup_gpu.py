"""Tests for the GPU setup wizard."""

from unittest.mock import patch, MagicMock, call
import pytest


@pytest.fixture
def mock_cfg():
    """Create a mock Config that tracks set() calls."""
    cfg = MagicMock()
    cfg._data = {}

    def _set(key, value):
        cfg._data[key] = value

    def _get(key, default=None):
        return cfg._data.get(key, default)

    cfg.set = MagicMock(side_effect=_set)
    cfg.get = MagicMock(side_effect=_get)
    cfg.save = MagicMock()
    return cfg


class TestSetupGPUCloudPath:
    """Test: user chooses cloud."""

    def test_cloud_choice_sets_mode(self, mock_cfg):
        from ct.cli import _setup_gpu

        with patch("builtins.input", return_value="y"):
            with patch("ct.cloud.auth.is_logged_in", return_value=True):
                with patch("ct.cloud.auth.get_user_email", return_value="test@x.com"):
                    _setup_gpu(mock_cfg)

        assert mock_cfg._data["compute.mode"] == "cloud"
        assert mock_cfg._data["gpu.setup_completed"] is True
        mock_cfg.save.assert_called()


class TestSetupGPULocalWithGPU:
    """Test: user chooses local, GPU is detected."""

    def test_local_with_sufficient_gpu(self, mock_cfg):
        from ct.cli import _setup_gpu
        from ct.cloud.router import GPUInfo

        gpu = GPUInfo(name="NVIDIA A100", vram_mb=81920)

        with patch("builtins.input", return_value="n"):
            with patch("ct.cloud.router._detect_local_gpu_info", return_value=[gpu]):
                with patch("ct.cloud.router.get_gpu_tool_compatibility", return_value=[
                    {"tool_name": "structure.esmfold", "min_vram_gb": 16,
                     "compatible": True, "gpu_name": "NVIDIA A100", "gpu_vram_gb": 80},
                    {"tool_name": "structure.alphafold3", "min_vram_gb": 40,
                     "compatible": True, "gpu_name": "NVIDIA A100", "gpu_vram_gb": 80},
                ]):
                    with patch("ct.cloud.router._check_docker", return_value=(True, "")):
                        _setup_gpu(mock_cfg)

        assert mock_cfg._data["compute.mode"] == "local"
        assert mock_cfg._data["gpu.name"] == "NVIDIA A100"

    def test_local_with_limited_gpu(self, mock_cfg):
        from ct.cli import _setup_gpu
        from ct.cloud.router import GPUInfo

        gpu = GPUInfo(name="RTX 3090", vram_mb=24576)

        with patch("builtins.input", return_value="n"):
            with patch("ct.cloud.router._detect_local_gpu_info", return_value=[gpu]):
                with patch("ct.cloud.router.get_gpu_tool_compatibility", return_value=[
                    {"tool_name": "structure.diffdock", "min_vram_gb": 8,
                     "compatible": True, "gpu_name": "RTX 3090", "gpu_vram_gb": 24},
                    {"tool_name": "structure.esmfold", "min_vram_gb": 16,
                     "compatible": True, "gpu_name": "RTX 3090", "gpu_vram_gb": 24},
                    {"tool_name": "structure.alphafold3", "min_vram_gb": 40,
                     "compatible": False, "gpu_name": "RTX 3090", "gpu_vram_gb": 24},
                ]):
                    with patch("ct.cloud.router._check_docker", return_value=(True, "")):
                        _setup_gpu(mock_cfg)

        assert mock_cfg._data["compute.mode"] == "local"


class TestSetupGPUNoGPU:
    """Test: user chooses local, no GPU found."""

    def test_no_gpu_fallback_to_cloud(self, mock_cfg):
        from ct.cli import _setup_gpu

        # First "n" for cloud, then "y" for fallback
        inputs = iter(["n", "y"])

        with patch("builtins.input", side_effect=lambda _: next(inputs)):
            with patch("ct.cloud.router._detect_local_gpu_info", return_value=[]):
                with patch("ct.cloud.auth.is_logged_in", return_value=True):
                    with patch("ct.cloud.auth.get_user_email", return_value="t@x.com"):
                        _setup_gpu(mock_cfg)

        assert mock_cfg._data["compute.mode"] == "cloud"

    def test_no_gpu_accept_limitation(self, mock_cfg):
        from ct.cli import _setup_gpu

        # First "n" for cloud, then "n" again (accept no GPU)
        inputs = iter(["n", "n"])

        with patch("builtins.input", side_effect=lambda _: next(inputs)):
            with patch("ct.cloud.router._detect_local_gpu_info", return_value=[]):
                _setup_gpu(mock_cfg)

        assert mock_cfg._data["compute.mode"] == "local"
        assert mock_cfg._data["gpu.setup_completed"] is True
