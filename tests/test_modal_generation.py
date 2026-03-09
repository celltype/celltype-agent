"""Tests for Modal app generation from manifest."""

import ast
import sys
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch


# Add celltype-cloud/scripts to path for import
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "celltype-cloud" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from generate_modal_app import generate_modal_app


SAMPLE_MANIFEST = {
    "version": "1.0",
    "tools": {
        "structure.esmfold": {
            "display_name": "ESMFold",
            "category": "structure",
            "gpu_profile": "structure",
            "hardware": {"gpu_type": "A10G", "min_vram_gb": 16, "cpu_only": False},
            "environment": {
                "python": "3.11",
                "pip": ["torch==2.3.1", "transformers>=4.40"],
                "apt": [],
                "conda": [],
                "env_vars": {},
            },
            "cost": {"per_second_base": 0.001389, "markup": 2.5},
            "execution": {"timeout_s": 300, "warm_pool": {"min": 0}},
            "modal_function": "predict_structure_esmfold",
        },
        "genomics.msa_search": {
            "display_name": "MSA-Search",
            "category": "genomics",
            "gpu_profile": "msa",
            "hardware": {"gpu_type": None, "min_vram_gb": 0, "min_ram_gb": 64, "cpu_only": True},
            "environment": {
                "python": "3.11",
                "pip": ["colabfold", "biopython"],
                "apt": ["mmseqs2"],
                "conda": [],
                "env_vars": {},
            },
            "cost": {"per_second_base": 0.000278, "markup": 2.5},
            "execution": {"timeout_s": 600, "warm_pool": {"min": 0}},
            "modal_function": "genomics_msa_search",
        },
        "genomics.evo2": {
            "display_name": "Evo2-40B",
            "category": "genomics",
            "gpu_profile": "structure",
            "hardware": {"gpu_type": "H100", "min_vram_gb": 80, "cpu_only": False},
            "environment": {
                "python": "3.11",
                "pip": ["torch==2.3.1", "evo-model", "flash-attn>=2.5"],
                "apt": [],
                "conda": [],
                "env_vars": {},
            },
            "cost": {"per_second_base": 0.002778, "markup": 2.5},
            "execution": {"timeout_s": 900, "warm_pool": {"min": 0}},
            "modal_function": "genomics_evo2",
        },
    },
}


class TestGenerateModalApp:
    def test_generated_code_is_valid_python(self):
        code = generate_modal_app(SAMPLE_MANIFEST)
        # Should parse without syntax errors
        ast.parse(code)

    def test_contains_all_tool_functions(self):
        code = generate_modal_app(SAMPLE_MANIFEST)
        assert "predict_structure_esmfold" in code
        assert "genomics_msa_search" in code
        assert "genomics_evo2" in code

    def test_gpu_tools_have_correct_gpu_type(self):
        code = generate_modal_app(SAMPLE_MANIFEST)
        # ESMFold should use A10G
        assert 'gpu="A10G"' in code
        # Evo2 should use H100
        assert 'gpu="H100"' in code

    def test_cpu_only_tool_has_no_gpu(self):
        code = generate_modal_app(SAMPLE_MANIFEST)
        # Find the MSA-Search function definition and check nearby lines
        lines = code.split("\n")
        in_msa_section = False
        msa_lines = []
        for line in lines:
            if "MSA-Search" in line:
                in_msa_section = True
            if in_msa_section:
                msa_lines.append(line)
                if "def genomics_msa_search" in line:
                    break
        msa_block = "\n".join(msa_lines)
        assert "memory=" in msa_block
        assert 'gpu=' not in msa_block

    def test_has_auto_generated_header(self):
        code = generate_modal_app(SAMPLE_MANIFEST)
        assert "DO NOT EDIT" in code
        assert "generate_modal_app.py" in code

    def test_has_dispatcher(self):
        code = generate_modal_app(SAMPLE_MANIFEST)
        assert "def run_gpu_tool" in code
        assert "function_map" in code

    def test_dispatcher_maps_all_tools(self):
        code = generate_modal_app(SAMPLE_MANIFEST)
        assert '"structure.esmfold"' in code
        assert '"genomics.msa_search"' in code
        assert '"genomics.evo2"' in code

    def test_imports_from_implementations(self):
        code = generate_modal_app(SAMPLE_MANIFEST)
        assert "from implementations.esmfold import run" in code
        assert "from implementations.msa_search import run" in code
        assert "from implementations.evo2 import run" in code

    def test_real_manifest_generates_15_functions(self):
        """Test with the actual repo manifest."""
        repo_root = Path(__file__).resolve().parent.parent
        manifest_path = repo_root / "tool_manifest.yaml"
        if not manifest_path.exists():
            pytest.skip("tool_manifest.yaml not found")

        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        code = generate_modal_app(manifest)
        ast.parse(code)  # Valid Python
        assert code.count("@app.function(") == 15
