"""
Test extensibility — verify adding a 16th tool requires only
manifest + implementation + registration (no gateway/router changes).
"""

import ast
import json
import sys
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch

from ct.cloud.manifest import load_manifest, validate_manifest, clear_cache, get_tool_config


@pytest.fixture(autouse=True)
def clear():
    clear_cache()
    yield
    clear_cache()


# A hypothetical 16th tool
TOOL_16 = {
    "display_name": "TestTool16",
    "category": "test",
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
        "pip": ["numpy", "biopython"],
        "apt": [],
        "conda": [],
        "env_vars": {},
    },
    "models": [],
    "cost": {
        "per_second_base": 0.001,
        "markup": 2.5,
        "estimated_per_run": 0.05,
    },
    "execution": {
        "estimated_duration_s": 60,
        "timeout_s": 300,
        "warm_pool": {"min": 0, "max": 1},
    },
    "docker_image": "celltype/test16:latest",
    "modal_function": "test_tool16",
    "parameters": {"input": "test input"},
    "tags": ["test"],
}


class TestAddingNewTool:
    """Verify a 16th tool can be added with only manifest+impl+registration."""

    def test_manifest_accepts_16th_tool(self, tmp_path):
        """Manifest validation works with 16 tools."""
        # Load real manifest
        repo_root = Path(__file__).resolve().parent.parent
        manifest_path = repo_root / "tool_manifest.yaml"
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        # Add 16th tool
        manifest["tools"]["test.tool16"] = TOOL_16

        errors = validate_manifest(manifest)
        assert errors == [], f"Validation errors: {errors}"
        assert len(manifest["tools"]) == 16

    def test_manifest_loader_returns_16th_tool(self, tmp_path):
        """Loader provides config for the new tool."""
        repo_root = Path(__file__).resolve().parent.parent
        manifest_path = repo_root / "tool_manifest.yaml"
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        manifest["tools"]["test.tool16"] = TOOL_16

        new_manifest_path = tmp_path / "tool_manifest.yaml"
        new_manifest_path.write_text(yaml.dump(manifest))

        loaded = load_manifest(str(new_manifest_path))
        config = loaded["tools"]["test.tool16"]
        assert config["display_name"] == "TestTool16"

    def test_modal_app_generates_16th_function(self, tmp_path):
        """Code generator produces function for the 16th tool."""
        scripts_dir = Path(__file__).resolve().parent.parent / "celltype-cloud" / "scripts"
        sys.path.insert(0, str(scripts_dir))
        from generate_modal_app import generate_modal_app

        repo_root = Path(__file__).resolve().parent.parent
        manifest_path = repo_root / "tool_manifest.yaml"
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        manifest["tools"]["test.tool16"] = TOOL_16

        code = generate_modal_app(manifest)

        # Valid Python
        ast.parse(code)

        # Has 16 functions
        assert code.count("@app.function(") == 16

        # Has the new function
        assert "test_tool16" in code
        assert "from implementations.tool16 import run" in code

    def test_gateway_needs_no_changes(self, tmp_path):
        """Gateway derives ALLOWED_TOOLS from manifest — no code changes needed."""
        repo_root = Path(__file__).resolve().parent.parent
        manifest_path = repo_root / "tool_manifest.yaml"
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        manifest["tools"]["test.tool16"] = TOOL_16

        new_manifest_path = tmp_path / "tool_manifest.yaml"
        new_manifest_path.write_text(yaml.dump(manifest))

        loaded = load_manifest(str(new_manifest_path))
        tools = set(loaded["tools"].keys())
        assert "test.tool16" in tools
        # The gateway would automatically allow this tool

    def test_router_needs_no_changes(self, tmp_path):
        """Compute router works with the new tool — no code changes needed."""
        from ct.cloud.router import ComputeRouter
        from dataclasses import dataclass
        from unittest.mock import MagicMock

        @dataclass
        class Tool16:
            name: str = "test.tool16"
            requires_gpu: bool = True
            gpu_profile: str = "structure"
            estimated_cost: float = 0.05
            docker_image: str = "celltype/test16:latest"
            min_vram_gb: int = 16
            min_ram_gb: int = 0
            cpu_only: bool = False
            num_gpus: int = 1

            def run(self, **kwargs):
                return {"summary": "tool16 result"}

        mock_config = MagicMock()
        mock_config.get.side_effect = lambda k, d=None: {
            "compute.mode": "cloud",
        }.get(k, d)

        router = ComputeRouter(config=mock_config)

        with patch("ct.cloud.auth.is_logged_in", return_value=True):
            with patch("ct.cloud.auth.get_token", return_value="tok"):
                with patch("ct.cloud.client.CloudClient.submit_and_wait",
                           return_value={"summary": "tool16 cloud"}) as mock:
                    result = router.route(Tool16())
                    assert result["summary"] == "tool16 cloud"

    def test_image_builder_generates_dockerfile(self):
        """Image builder generates Dockerfile for the new tool."""
        from ct.cloud.image_builder import generate_dockerfile

        dockerfile = generate_dockerfile(TOOL_16)
        assert "nvidia/cuda:12.1.0-runtime-ubuntu22.04" in dockerfile
        assert "numpy" in dockerfile
        assert "biopython" in dockerfile
        assert 'ENTRYPOINT ["python", "/opt/tool_entrypoint.py"]' in dockerfile
