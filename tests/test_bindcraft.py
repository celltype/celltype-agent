"""Tests for BindCraft binder design tool — mocked, no GPU required."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock


SAMPLE_PDB = (
    "HEADER    TEST\n"
    "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N\n"
    "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00  0.00           C\n"
    "ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00  0.00           C\n"
    "ATOM      4  O   ALA A   1       4.000   5.000   6.000  1.00  0.00           O\n"
    "ATOM      5  N   GLY A   2       5.000   6.000   7.000  1.00  0.00           N\n"
    "ATOM      6  CA  GLY A   2       6.000   7.000   8.000  1.00  0.00           C\n"
    "ATOM      7  C   GLY A   2       7.000   8.000   9.000  1.00  0.00           C\n"
    "ATOM      8  O   GLY A   2       8.000   9.000  10.000  1.00  0.00           O\n"
    "END\n"
)


class TestBindCraftNormalization:
    """Test argument normalization logic."""

    def test_normalize_basic_args(self):
        from ct.tools.bindcraft.implementation import normalize_args

        result = normalize_args({
            "target_pdb": SAMPLE_PDB,
            "chains": "A",
            "num_designs": 5,
        })
        assert result["chains"] == "A"
        assert result["num_designs"] == 5
        assert result["binder_length_min"] == 65
        assert result["binder_length_max"] == 150
        assert result["design_mode"] == "default"
        assert result["binder_name"] == "binder"

    def test_normalize_rejects_missing_pdb(self):
        from ct.tools.bindcraft.implementation import normalize_args

        with pytest.raises(ValueError, match="requires `target_pdb`"):
            normalize_args({"chains": "A"})

    def test_normalize_rejects_empty_pdb(self):
        from ct.tools.bindcraft.implementation import normalize_args

        with pytest.raises(ValueError, match="requires `target_pdb`"):
            normalize_args({"target_pdb": ""})

    def test_normalize_rejects_no_atom_records(self):
        from ct.tools.bindcraft.implementation import normalize_args

        with pytest.raises(ValueError, match="ATOM/HETATM"):
            normalize_args({"target_pdb": "HEADER only\nREMARK no atoms\n"})

    def test_normalize_rejects_invalid_chain(self):
        from ct.tools.bindcraft.implementation import normalize_args

        with pytest.raises(ValueError, match="Chain 'Z' not found"):
            normalize_args({"target_pdb": SAMPLE_PDB, "chains": "Z"})

    def test_normalize_clamps_binder_length(self):
        from ct.tools.bindcraft.implementation import normalize_args

        result = normalize_args({
            "target_pdb": SAMPLE_PDB,
            "binder_length_min": 1,
            "binder_length_max": 999,
        })
        assert result["binder_length_min"] == 4
        assert result["binder_length_max"] == 250

    def test_normalize_enforces_min_le_max(self):
        from ct.tools.bindcraft.implementation import normalize_args

        result = normalize_args({
            "target_pdb": SAMPLE_PDB,
            "binder_length_min": 100,
            "binder_length_max": 50,
        })
        assert result["binder_length_max"] >= result["binder_length_min"]

    def test_normalize_clamps_num_designs(self):
        from ct.tools.bindcraft.implementation import normalize_args

        result = normalize_args({
            "target_pdb": SAMPLE_PDB,
            "num_designs": 999,
        })
        assert result["num_designs"] == 100

    def test_normalize_rejects_invalid_design_mode(self):
        from ct.tools.bindcraft.implementation import normalize_args

        with pytest.raises(ValueError, match="design_mode"):
            normalize_args({"target_pdb": SAMPLE_PDB, "design_mode": "turbo"})

    def test_normalize_accepts_peptide_mode(self):
        from ct.tools.bindcraft.implementation import normalize_args

        result = normalize_args({
            "target_pdb": SAMPLE_PDB,
            "design_mode": "peptide",
            "binder_length_min": 8,
            "binder_length_max": 25,
        })
        assert result["design_mode"] == "peptide"

    def test_normalize_rejects_unexpected_args(self):
        from ct.tools.bindcraft.implementation import normalize_args

        with pytest.raises(ValueError, match="unsupported arguments"):
            normalize_args({"target_pdb": SAMPLE_PDB, "turbo_mode": True})

    def test_normalize_auto_detects_chain(self):
        from ct.tools.bindcraft.implementation import normalize_args

        result = normalize_args({"target_pdb": SAMPLE_PDB, "chains": ""})
        assert result["chains"] == "A"

    def test_normalize_pdb_text_alias(self):
        from ct.tools.bindcraft.implementation import normalize_args

        result = normalize_args({"pdb_text": SAMPLE_PDB})
        assert "ATOM" in result["target_pdb"]


class TestBindCraftHelpers:
    """Test helper functions."""

    def test_parse_chain_ids(self):
        from ct.tools.bindcraft.implementation import _parse_chain_ids

        chains = _parse_chain_ids(SAMPLE_PDB)
        assert "A" in chains

    def test_count_residues(self):
        from ct.tools.bindcraft.implementation import _count_residues

        count = _count_residues(SAMPLE_PDB)
        assert count == 2

    def test_build_settings_json(self):
        from ct.tools.bindcraft.implementation import _build_settings_json

        normalized = {
            "binder_name": "test_binder",
            "chains": "A",
            "hotspot_residues": "56,57",
            "binder_length_min": 65,
            "binder_length_max": 150,
            "num_designs": 5,
        }
        settings = _build_settings_json(normalized, "/tmp/target.pdb", "/tmp/output")
        assert settings["starting_pdb"] == "/tmp/target.pdb"
        assert settings["design_path"] == "/tmp/output"
        assert settings["binder_name"] == "test_binder"
        assert settings["chains"] == "A"
        assert settings["target_hotspot_residues"] == "56,57"
        assert settings["lengths"] == [65, 150]
        assert settings["number_of_final_designs"] == 5

    def test_build_settings_json_no_hotspots(self):
        from ct.tools.bindcraft.implementation import _build_settings_json

        normalized = {
            "binder_name": "binder",
            "chains": "A",
            "hotspot_residues": "",
            "binder_length_min": 65,
            "binder_length_max": 150,
            "num_designs": 5,
        }
        settings = _build_settings_json(normalized, "/tmp/target.pdb", "/tmp/output")
        assert settings["target_hotspot_residues"] is None


class TestBindCraftRun:
    """Test the main run() function with mocked subprocess."""

    def test_run_returns_error_for_empty_pdb(self):
        from ct.tools.bindcraft.implementation import run

        result = run(target_pdb="")
        assert "Error" in result["summary"]
        assert result["error"] == "invalid_args"

    def test_run_returns_error_for_too_large_complex(self):
        from ct.tools.bindcraft.implementation import run

        huge_pdb_lines = ["HEADER    TEST\n"]
        for i in range(1, 900):
            huge_pdb_lines.append(
                f"ATOM  {i:5d}  CA  ALA A{i:4d}       "
                f"{float(i):7.3f}{float(i):8.3f}{float(i):8.3f}"
                f"  1.00  0.00           C\n"
            )
        huge_pdb_lines.append("END\n")
        huge_pdb = "".join(huge_pdb_lines)

        result = run(target_pdb=huge_pdb, binder_length_max=150)
        assert "too large" in result["summary"]
        assert result["error"] == "complex_too_large"

    @patch("ct.tools.bindcraft.implementation.subprocess.run")
    @patch("ct.tools.bindcraft.implementation.os.path.isfile")
    def test_run_success_with_designs(self, mock_isfile, mock_subprocess, tmp_path):
        from ct.tools.bindcraft.implementation import run

        mock_isfile.return_value = True

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "BindCraft completed"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        accepted_dir = tmp_path / "output" / "Accepted"
        accepted_dir.mkdir(parents=True)
        (accepted_dir / "binder_001.pdb").write_text(SAMPLE_PDB)
        (accepted_dir / "binder_002.pdb").write_text(SAMPLE_PDB)

        with patch("ct.tools.bindcraft.implementation.tempfile.TemporaryDirectory") as mock_tmpdir:
            mock_tmpdir.return_value.__enter__ = lambda s: str(tmp_path)
            mock_tmpdir.return_value.__exit__ = MagicMock(return_value=False)

            with patch("ct.tools.bindcraft.implementation._get_gpu_vram_mb", return_value=0):
                result = run(target_pdb=SAMPLE_PDB, num_designs=2)

        assert "Error" not in result.get("summary", "") or "not installed" in result.get("summary", "").lower()

    @patch("ct.tools.bindcraft.implementation.subprocess.run")
    @patch("ct.tools.bindcraft.implementation.os.path.isfile", return_value=True)
    def test_run_handles_subprocess_failure(self, mock_isfile, mock_subprocess):
        from ct.tools.bindcraft.implementation import run

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "CUDA out of memory"
        mock_subprocess.return_value = mock_result

        with patch("ct.tools.bindcraft.implementation._get_gpu_vram_mb", return_value=0):
            result = run(target_pdb=SAMPLE_PDB)

        assert "Error" in result["summary"] or "failed" in result["summary"].lower()

    @patch("ct.tools.bindcraft.implementation.subprocess.run")
    @patch("ct.tools.bindcraft.implementation.os.path.isfile", return_value=True)
    def test_run_handles_timeout(self, mock_isfile, mock_subprocess):
        import subprocess as sp
        from ct.tools.bindcraft.implementation import run

        mock_subprocess.side_effect = sp.TimeoutExpired(cmd="bindcraft", timeout=7200)

        with patch("ct.tools.bindcraft.implementation._get_gpu_vram_mb", return_value=0):
            result = run(target_pdb=SAMPLE_PDB)

        assert "timeout" in result["summary"].lower() or result["error"] == "timeout"


class TestBindCraftCollectDesigns:
    """Test output collection logic."""

    def test_collect_from_accepted_dir(self, tmp_path):
        from ct.tools.bindcraft.implementation import _collect_designs

        accepted = tmp_path / "Accepted"
        accepted.mkdir()
        (accepted / "binder_001.pdb").write_text(SAMPLE_PDB)
        (accepted / "binder_002.pdb").write_text(SAMPLE_PDB)

        designs, pdbs = _collect_designs(str(tmp_path))
        assert len(designs) == 2
        assert len(pdbs) == 2
        assert designs[0]["name"] == "binder_001.pdb"

    def test_collect_from_ranked_subdir(self, tmp_path):
        from ct.tools.bindcraft.implementation import _collect_designs

        ranked = tmp_path / "Accepted" / "Ranked"
        ranked.mkdir(parents=True)
        (ranked / "ranked_001.pdb").write_text(SAMPLE_PDB)

        designs, pdbs = _collect_designs(str(tmp_path))
        assert len(designs) == 1
        assert designs[0]["name"] == "ranked_001.pdb"

    def test_collect_empty_dir(self, tmp_path):
        from ct.tools.bindcraft.implementation import _collect_designs

        designs, pdbs = _collect_designs(str(tmp_path))
        assert len(designs) == 0
        assert len(pdbs) == 0

    def test_collect_with_stats_csv(self, tmp_path):
        from ct.tools.bindcraft.implementation import _collect_designs

        accepted = tmp_path / "Accepted"
        accepted.mkdir()
        (accepted / "binder_001.pdb").write_text(SAMPLE_PDB)

        stats = "name,i_pTM,i_pAE,pLDDT,Binder_RMSD\nbinder_001,0.85,5.2,92.1,0.7\n"
        (tmp_path / "final_design_stats.csv").write_text(stats)

        designs, pdbs = _collect_designs(str(tmp_path))
        assert len(designs) == 1
        assert designs[0].get("i_pTM") == pytest.approx(0.85)
        assert designs[0].get("pLDDT") == pytest.approx(92.1)


class TestBindCraftRegistration:
    """Test that BindCraft registers correctly via the container tool loader."""

    def test_bindcraft_tool_yaml_loads(self):
        import yaml

        yaml_path = os.path.join(
            os.path.dirname(__file__),
            "..", "src", "ct", "tools", "bindcraft", "tool.yaml"
        )
        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        assert config["name"] == "design.bindcraft"
        assert config["category"] == "design"
        assert config["docker_image"] == "celltype/bindcraft:latest"
        assert config["compute"]["requires_gpu"] is True
        assert config["compute"]["min_vram_gb"] == 32
        assert "target_pdb" in config["parameters"]["properties"]

    def test_bindcraft_registered_in_registry(self):
        from ct.tools import registry, ensure_loaded

        ensure_loaded()

        tool = registry.get_tool("design.bindcraft")
        assert tool is not None
        assert tool.requires_gpu is True
        assert tool.docker_image == "celltype/bindcraft:latest"
        assert tool.category == "design"
        assert tool.min_vram_gb == 32
        assert tool.gpu_profile == "structure"


class TestBindCraftLocalRunner:
    """Test BindCraft integration with LocalRunner."""

    def test_local_runner_inlines_target_pdb(self, tmp_path):
        from dataclasses import dataclass
        from ct.cloud.local_runner import LocalRunner

        @dataclass
        class FakeTool:
            name: str = "design.bindcraft"
            requires_gpu: bool = True
            gpu_profile: str = "structure"
            estimated_cost: float = 0.25
            docker_image: str = "celltype/bindcraft:latest"
            min_vram_gb: int = 32
            min_ram_gb: int = 0
            cpu_only: bool = False
            num_gpus: int = 1

        runner = LocalRunner(workspace=tmp_path)
        tool = FakeTool()
        pdb_path = tmp_path / "target.pdb"
        pdb_path.write_text(SAMPLE_PDB, encoding="utf-8")

        with patch("ct.cloud.local_runner.subprocess.run") as mock_run:
            inspect_result = MagicMock(returncode=0)
            run_result = MagicMock(returncode=0, stdout="", stderr="")
            mock_run.side_effect = [inspect_result, run_result]

            runner.run(tool, target_pdb=str(pdb_path))

        input_payload = json.loads((runner._session_dir / "input.json").read_text())
        assert input_payload["target_pdb"] == SAMPLE_PDB

    def test_bindcraft_cache_mount_exists(self):
        from ct.cloud.local_runner import LocalRunner

        runner = LocalRunner()
        mounts = runner._get_cache_mounts()
        mount_str = " ".join(mounts)
        assert "bindcraft" in mount_str
