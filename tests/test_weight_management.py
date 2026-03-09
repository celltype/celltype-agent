"""Tests for model weight management."""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from ct.cloud.manifest import clear_cache
from ct.cloud.weight_downloader import (
    _cache_path_for_model,
    is_cached,
    pull_tool_weights,
)


@pytest.fixture(autouse=True)
def clear_manifest():
    clear_cache()
    yield
    clear_cache()


SAMPLE_MANIFEST = {
    "version": "1.0",
    "tools": {
        "structure.esmfold": {
            "display_name": "ESMFold",
            "category": "structure",
            "gpu_profile": "structure",
            "hardware": {"gpu_type": "A10G", "min_vram_gb": 16, "cpu_only": False},
            "environment": {"python": "3.11", "pip": [], "apt": [], "conda": [], "env_vars": {}},
            "models": [
                {
                    "name": "facebook/esmfold_v1",
                    "source": "huggingface",
                    "size_gb": 4,
                    "cache_path": "/vol/models/esmfold",
                    "required": True,
                }
            ],
            "cost": {"per_second_base": 0.001, "markup": 2.5},
            "execution": {"estimated_duration_s": 90, "timeout_s": 300},
            "docker_image": "celltype/esmfold:latest",
        },
        "structure.openfold3": {
            "display_name": "OpenFold3",
            "category": "structure",
            "gpu_profile": "structure",
            "hardware": {"gpu_type": "A100", "min_vram_gb": 80, "cpu_only": False},
            "environment": {"python": "3.11", "pip": [], "apt": [], "conda": [], "env_vars": {}},
            "models": [
                {
                    "name": "openfold3-weights",
                    "source": "huggingface",
                    "size_gb": 15,
                    "cache_path": "/vol/models/openfold3",
                    "required": True,
                }
            ],
            "databases": [
                {
                    "name": "bfd",
                    "source": "url",
                    "size_gb": 1700,
                    "cache_path": "/vol/databases/bfd",
                    "required": False,
                    "optional": True,
                }
            ],
            "cost": {"per_second_base": 0.001, "markup": 2.5},
            "execution": {"estimated_duration_s": 180, "timeout_s": 900},
            "docker_image": "celltype/openfold3:latest",
        },
    },
}


@pytest.fixture
def manifest_file(tmp_path):
    path = tmp_path / "tool_manifest.yaml"
    path.write_text(yaml.dump(SAMPLE_MANIFEST))
    return path


class TestCachePath:
    def test_sanitizes_name(self):
        model = {"name": "facebook/esmfold_v1"}
        path = _cache_path_for_model(model)
        assert "facebook_esmfold_v1" in str(path)


class TestIsCached:
    def test_not_cached(self, tmp_path):
        model = {"name": "nonexistent/model"}
        with patch("ct.cloud.weight_downloader.get_cache_dir", return_value=tmp_path):
            assert is_cached(model) is False

    def test_cached(self, tmp_path):
        model = {"name": "test_model"}
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        (model_dir / "weights.bin").write_text("data")
        with patch("ct.cloud.weight_downloader.get_cache_dir", return_value=tmp_path):
            assert is_cached(model) is True


class TestPullToolWeights:
    def test_unknown_tool(self, manifest_file):
        from ct.cloud.manifest import load_manifest
        load_manifest(str(manifest_file))
        result = pull_tool_weights("nonexistent.tool")
        assert "error" in result
        assert "not found" in result["summary"]

    def test_already_cached(self, manifest_file, tmp_path):
        from ct.cloud.manifest import load_manifest
        load_manifest(str(manifest_file))

        # Create cached directory
        cache_dir = tmp_path / "models"
        model_dir = cache_dir / "facebook_esmfold_v1"
        model_dir.mkdir(parents=True)
        (model_dir / "weights.bin").write_text("data")

        with patch("ct.cloud.weight_downloader.get_cache_dir", return_value=cache_dir):
            result = pull_tool_weights("structure.esmfold")
            assert "error" not in result
            assert "Already cached" in result["summary"] or "Pulled" in result["summary"]

    def test_skips_optional_databases(self, manifest_file):
        from ct.cloud.manifest import load_manifest
        load_manifest(str(manifest_file))

        with patch("ct.cloud.weight_downloader.download_model") as mock_dl:
            mock_dl.return_value = Path("/tmp/test")
            result = pull_tool_weights("structure.openfold3", include_optional=False)
            assert "error" not in result
            assert "bfd" in str(result.get("skipped", []))
