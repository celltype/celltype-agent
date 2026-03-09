"""Tests for Docker image builder."""

import pytest
from pathlib import Path

from ct.cloud.image_builder import generate_dockerfile, write_dockerfile


GPU_TOOL_CONFIG = {
    "display_name": "ESMFold",
    "hardware": {
        "gpu_type": "A10G",
        "min_vram_gb": 16,
        "cpu_only": False,
    },
    "environment": {
        "base": "nvidia/cuda:12.1.0-runtime-ubuntu22.04",
        "python": "3.11",
        "cuda": "12.1",
        "pip": ["torch==2.3.1", "transformers>=4.40", "accelerate", "biopython", "numpy"],
        "apt": [],
        "conda": [],
        "env_vars": {"TRANSFORMERS_CACHE": "/vol/models/esmfold"},
    },
}

CPU_TOOL_CONFIG = {
    "display_name": "MSA-Search",
    "hardware": {
        "gpu_type": None,
        "min_vram_gb": 0,
        "cpu_only": True,
    },
    "environment": {
        "base": "python:3.11-slim",
        "python": "3.11",
        "cuda": None,
        "pip": ["colabfold", "biopython", "numpy"],
        "apt": ["mmseqs2"],
        "conda": [],
        "env_vars": {"COLABFOLD_DB": "/vol/databases/colabfold"},
    },
}

JAX_TOOL_CONFIG = {
    "display_name": "AlphaFold2",
    "hardware": {
        "gpu_type": "A100",
        "min_vram_gb": 40,
        "cpu_only": False,
    },
    "environment": {
        "base": "nvidia/cuda:11.8.0-runtime-ubuntu22.04",
        "python": "3.11",
        "cuda": "11.8",
        "pip": ["jax[cuda11_pip]==0.4.30", "dm-haiku==0.0.12", "openmm==8.1.1", "biopython", "numpy"],
        "apt": [],
        "conda": [],
        "env_vars": {
            "XLA_FLAGS": "--xla_gpu_cuda_data_dir=/usr/local/cuda",
            "JAX_PLATFORMS": "gpu",
        },
    },
}


class TestGenerateDockerfile:
    def test_gpu_tool_uses_cuda_base(self):
        dockerfile = generate_dockerfile(GPU_TOOL_CONFIG)
        assert "nvidia/cuda:12.1.0-runtime-ubuntu22.04" in dockerfile

    def test_gpu_tool_installs_pip_packages(self):
        dockerfile = generate_dockerfile(GPU_TOOL_CONFIG)
        assert "torch==2.3.1" in dockerfile
        assert "transformers>=4.40" in dockerfile

    def test_gpu_tool_sets_env_vars(self):
        dockerfile = generate_dockerfile(GPU_TOOL_CONFIG)
        assert "TRANSFORMERS_CACHE=/vol/models/esmfold" in dockerfile

    def test_gpu_tool_has_entrypoint(self):
        dockerfile = generate_dockerfile(GPU_TOOL_CONFIG)
        assert 'ENTRYPOINT ["python", "/opt/tool_entrypoint.py"]' in dockerfile

    def test_gpu_tool_copies_implementation(self):
        dockerfile = generate_dockerfile(GPU_TOOL_CONFIG)
        assert "COPY implementation.py /opt/implementation.py" in dockerfile

    def test_cpu_only_uses_python_slim(self):
        dockerfile = generate_dockerfile(CPU_TOOL_CONFIG)
        assert "python:3.11-slim" in dockerfile
        assert "nvidia" not in dockerfile

    def test_cpu_only_installs_apt_packages(self):
        dockerfile = generate_dockerfile(CPU_TOOL_CONFIG)
        assert "mmseqs2" in dockerfile

    def test_cpu_only_has_no_gpu_base(self):
        dockerfile = generate_dockerfile(CPU_TOOL_CONFIG)
        assert "cuda" not in dockerfile.split("FROM")[1].split("\n")[0]

    def test_jax_tool_uses_correct_cuda(self):
        dockerfile = generate_dockerfile(JAX_TOOL_CONFIG)
        assert "nvidia/cuda:11.8.0-runtime-ubuntu22.04" in dockerfile
        assert "jax[cuda11_pip]==0.4.30" in dockerfile

    def test_jax_tool_env_vars(self):
        dockerfile = generate_dockerfile(JAX_TOOL_CONFIG)
        assert "XLA_FLAGS" in dockerfile
        assert "JAX_PLATFORMS=gpu" in dockerfile

    def test_auto_generated_header(self):
        dockerfile = generate_dockerfile(GPU_TOOL_CONFIG)
        assert "DO NOT EDIT" in dockerfile
        assert "image_builder.py" in dockerfile


class TestWriteDockerfile:
    def test_writes_to_directory(self, tmp_path):
        output_dir = tmp_path / "build"
        result = write_dockerfile(GPU_TOOL_CONFIG, output_dir)
        assert result.exists()
        assert result.name == "Dockerfile"
        content = result.read_text()
        assert "nvidia/cuda" in content

    def test_creates_directory_if_needed(self, tmp_path):
        output_dir = tmp_path / "nested" / "build"
        assert not output_dir.exists()
        result = write_dockerfile(GPU_TOOL_CONFIG, output_dir)
        assert output_dir.exists()
        assert result.exists()
