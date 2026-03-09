"""Tests for tool manifest loader and validation."""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch

from ct.cloud.manifest import (
    load_manifest,
    get_tool_config,
    get_allowed_tools,
    get_allowed_profiles,
    get_cost_per_second,
    get_environment_spec,
    validate_manifest,
    clear_cache,
)


@pytest.fixture(autouse=True)
def clear_manifest_cache():
    """Clear manifest cache before each test."""
    clear_cache()
    yield
    clear_cache()


MINIMAL_MANIFEST = {
    "version": "1.0",
    "tools": {
        "structure.esmfold": {
            "display_name": "ESMFold",
            "category": "structure",
            "gpu_profile": "structure",
            "hardware": {
                "gpu_type": "A10G",
                "min_vram_gb": 16,
                "num_gpus": 1,
                "min_ram_gb": 16,
                "cpu_only": False,
            },
            "environment": {
                "base": "nvidia/cuda:12.1.0-runtime-ubuntu22.04",
                "python": "3.11",
                "cuda": "12.1",
                "pip": ["torch==2.3.1"],
                "apt": [],
                "conda": [],
                "env_vars": {},
            },
            "models": [
                {
                    "name": "facebook/esmfold_v1",
                    "source": "huggingface",
                    "size_gb": 4,
                    "cache_path": "/vol/models/esmfold",
                    "required": True,
                }
            ],
            "cost": {
                "per_second_base": 0.001389,
                "markup": 2.5,
                "estimated_per_run": 0.05,
            },
            "execution": {
                "estimated_duration_s": 90,
                "timeout_s": 300,
                "warm_pool": {"min": 0, "max": 1},
            },
            "docker_image": "celltype/esmfold:latest",
            "modal_function": "predict_structure_esmfold",
            "parameters": {"sequence": "Amino acid sequence"},
            "tags": ["protein", "structure-prediction"],
        }
    },
}


class TestValidateManifest:
    def test_valid_manifest(self):
        errors = validate_manifest(MINIMAL_MANIFEST)
        assert errors == []

    def test_missing_version(self):
        m = {"tools": MINIMAL_MANIFEST["tools"]}
        errors = validate_manifest(m)
        assert any("version" in e for e in errors)

    def test_unsupported_version(self):
        m = {"version": "2.0", "tools": MINIMAL_MANIFEST["tools"]}
        errors = validate_manifest(m)
        assert any("2.0" in e for e in errors)
        assert any("Supported" in e or "Unsupported" in e for e in errors)

    def test_missing_tools(self):
        m = {"version": "1.0"}
        errors = validate_manifest(m)
        assert any("tools" in e.lower() for e in errors)

    def test_missing_hardware_section(self):
        tool = dict(MINIMAL_MANIFEST["tools"]["structure.esmfold"])
        del tool["hardware"]
        m = {"version": "1.0", "tools": {"test.tool": tool}}
        errors = validate_manifest(m)
        assert any("hardware" in e for e in errors)

    def test_invalid_gpu_type(self):
        tool = dict(MINIMAL_MANIFEST["tools"]["structure.esmfold"])
        tool["hardware"] = dict(tool["hardware"])
        tool["hardware"]["gpu_type"] = "V100"
        m = {"version": "1.0", "tools": {"test.tool": tool}}
        errors = validate_manifest(m)
        assert any("V100" in e for e in errors)

    def test_cpu_only_with_gpu_type(self):
        tool = dict(MINIMAL_MANIFEST["tools"]["structure.esmfold"])
        tool["hardware"] = dict(tool["hardware"])
        tool["hardware"]["cpu_only"] = True
        tool["hardware"]["gpu_type"] = "A10G"
        m = {"version": "1.0", "tools": {"test.tool": tool}}
        errors = validate_manifest(m)
        assert any("cpu_only" in e for e in errors)

    def test_negative_cost(self):
        tool = dict(MINIMAL_MANIFEST["tools"]["structure.esmfold"])
        tool["cost"] = dict(tool["cost"])
        tool["cost"]["per_second_base"] = -0.001
        m = {"version": "1.0", "tools": {"test.tool": tool}}
        errors = validate_manifest(m)
        assert any("per_second_base" in e for e in errors)


class TestLoadManifest:
    def test_load_from_file(self, tmp_path):
        manifest_path = tmp_path / "tool_manifest.yaml"
        manifest_path.write_text(yaml.dump(MINIMAL_MANIFEST))
        result = load_manifest(str(manifest_path))
        assert result["version"] == "1.0"
        assert "structure.esmfold" in result["tools"]

    def test_load_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_manifest("/nonexistent/path/manifest.yaml")

    def test_load_invalid_manifest(self, tmp_path):
        bad_manifest = {"version": "1.0", "tools": {
            "bad.tool": {"cost": {"per_second_base": -1}}
        }}
        manifest_path = tmp_path / "tool_manifest.yaml"
        manifest_path.write_text(yaml.dump(bad_manifest))
        with pytest.raises(ValueError, match="Invalid manifest"):
            load_manifest(str(manifest_path))

    def test_load_real_manifest(self):
        """Test that the actual repo manifest loads successfully."""
        repo_root = Path(__file__).resolve().parent.parent
        manifest_path = repo_root / "tool_manifest.yaml"
        if manifest_path.exists():
            result = load_manifest(str(manifest_path))
            assert result["version"] == "1.0"
            assert len(result["tools"]) == 15


class TestGetToolConfig:
    def test_get_existing_tool(self, tmp_path):
        manifest_path = tmp_path / "tool_manifest.yaml"
        manifest_path.write_text(yaml.dump(MINIMAL_MANIFEST))
        load_manifest(str(manifest_path))
        config = get_tool_config("structure.esmfold")
        assert config is not None
        assert config["display_name"] == "ESMFold"
        assert config["hardware"]["gpu_type"] == "A10G"

    def test_get_nonexistent_tool(self, tmp_path):
        manifest_path = tmp_path / "tool_manifest.yaml"
        manifest_path.write_text(yaml.dump(MINIMAL_MANIFEST))
        load_manifest(str(manifest_path))
        config = get_tool_config("nonexistent.tool")
        assert config is None


class TestGetAllowedTools:
    def test_returns_tool_names(self, tmp_path):
        manifest_path = tmp_path / "tool_manifest.yaml"
        manifest_path.write_text(yaml.dump(MINIMAL_MANIFEST))
        load_manifest(str(manifest_path))
        tools = get_allowed_tools()
        assert "structure.esmfold" in tools


class TestGetAllowedProfiles:
    def test_returns_profiles(self, tmp_path):
        manifest_path = tmp_path / "tool_manifest.yaml"
        manifest_path.write_text(yaml.dump(MINIMAL_MANIFEST))
        load_manifest(str(manifest_path))
        profiles = get_allowed_profiles()
        assert "structure" in profiles


class TestGetCostPerSecond:
    def test_returns_cost_with_markup(self, tmp_path):
        manifest_path = tmp_path / "tool_manifest.yaml"
        manifest_path.write_text(yaml.dump(MINIMAL_MANIFEST))
        load_manifest(str(manifest_path))
        cost = get_cost_per_second("structure.esmfold")
        expected = 0.001389 * 2.5
        assert abs(cost - expected) < 1e-6

    def test_unknown_tool_returns_zero(self, tmp_path):
        manifest_path = tmp_path / "tool_manifest.yaml"
        manifest_path.write_text(yaml.dump(MINIMAL_MANIFEST))
        load_manifest(str(manifest_path))
        assert get_cost_per_second("nonexistent.tool") == 0.0


class TestGetEnvironmentSpec:
    def test_returns_env_spec(self, tmp_path):
        manifest_path = tmp_path / "tool_manifest.yaml"
        manifest_path.write_text(yaml.dump(MINIMAL_MANIFEST))
        load_manifest(str(manifest_path))
        env = get_environment_spec("structure.esmfold")
        assert env is not None
        assert any("torch" in p for p in env["pip"])
        assert env["python"] == "3.11"
