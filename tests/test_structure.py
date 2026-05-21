"""Tests for structure prediction tools."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def _rdkit_available():
    try:
        from rdkit import Chem
        return True
    except ImportError:
        return False


# ─── Helper: create a minimal PDB file ───────────────────────────
def _write_mini_pdb(path: Path) -> Path:
    """Write a small but valid PDB with a few atoms for testing."""
    pdb_text = """\
HEADER    TEST PROTEIN
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00  0.00           C
ATOM      4  O   ALA A   1       4.000   5.000   6.000  1.00  0.00           O
ATOM      5  N   VAL A   2       5.000   6.000   7.000  1.00  0.00           N
ATOM      6  CA  VAL A   2       6.000   7.000   8.000  1.00  0.00           C
ATOM      7  C   VAL A   2       7.000   8.000   9.000  1.00  0.00           C
ATOM      8  O   VAL A   2       8.000   9.000  10.000  1.00  0.00           O
ATOM      9  N   LEU A   3       9.000  10.000  11.000  1.00  0.00           N
ATOM     10  CA  LEU A   3      10.000  11.000  12.000  1.00  0.00           C
ATOM     11  C   LEU A   3      11.000  12.000  13.000  1.00  0.00           C
ATOM     12  O   LEU A   3      12.000  13.000  14.000  1.00  0.00           O
ATOM     13  N   PHE A   4      13.000  14.000  15.000  1.00  0.00           N
ATOM     14  CA  PHE A   4      14.000  15.000  16.000  1.00  0.00           C
ATOM     15  C   PHE A   4      15.000  16.000  17.000  1.00  0.00           C
ATOM     16  O   PHE A   4      16.000  17.000  18.000  1.00  0.00           O
ATOM     17  N   ILE A   5      17.000  18.000  19.000  1.00  0.00           N
ATOM     18  CA  ILE A   5      18.000  19.000  20.000  1.00  0.00           C
ATOM     19  C   ILE A   5      19.000  20.000  21.000  1.00  0.00           C
ATOM     20  O   ILE A   5      20.000  21.000  22.000  1.00  0.00           O
ATOM     21  N   TRP A   6      21.000  22.000  23.000  1.00  0.00           N
ATOM     22  CA  TRP A   6      22.000  23.000  24.000  1.00  0.00           C
ATOM     23  C   TRP A   6      23.000  24.000  25.000  1.00  0.00           C
ATOM     24  O   TRP A   6      24.000  25.000  26.000  1.00  0.00           O
ATOM     25  N   MET A   7       2.500   4.500   6.500  1.00  0.00           N
ATOM     26  CA  MET A   7       3.500   5.500   7.500  1.00  0.00           C
ATOM     27  N   PRO A   8       4.500   6.500   8.500  1.00  0.00           N
ATOM     28  CA  PRO A   8       5.500   7.500   9.500  1.00  0.00           C
ATOM     29  N   GLY A   9      10.500  11.500  12.500  1.00  0.00           N
ATOM     30  CA  GLY A   9      11.500  12.500  13.500  1.00  0.00           C
END
"""
    pdb_file = path / "test_protein.pdb"
    pdb_file.write_text(pdb_text)
    return pdb_file


# Mock compute providers for structure tools that submit cloud jobs
MOCK_PROVIDERS = {
    "providers": [
        {
            "id": "lambda",
            "name": "Lambda Labs",
            "website": "https://lambdalabs.com",
            "api_base_url": "https://cloud.lambdalabs.com/api/v1",
            "gpu_types": [
                {"id": "A100_80GB", "name": "NVIDIA A100 80GB", "vram_gb": 80, "price_per_hour": 1.29},
            ],
        },
    ],
    "job_templates": {
        "molecular_docking": {
            "description": "Molecular docking",
            "gpu_requirement_vram_gb": 24,
            "estimated_time_per_sample_minutes": 30,
            "recommended_gpu": "A100_80GB",
        },
        "molecular_dynamics": {
            "description": "Molecular dynamics simulation",
            "gpu_requirement_vram_gb": 24,
            "estimated_time_per_sample_minutes": 60,
            "recommended_gpu": "A100_80GB",
        },
    },
}


class TestAlphaFoldFetch:
    @patch("httpx.get")
    def test_downloads_structure(self, mock_get, tmp_path):
        from ct.tools.structure import alphafold_fetch

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "HEADER    SOME PDB DATA\nATOM   1  CA  ALA A   1       0.0  0.0  0.0\n"
        mock_get.return_value = mock_resp

        cache_dir = tmp_path / ".ct" / "cache" / "alphafold"
        with patch("ct.tools.structure.Path.home", return_value=tmp_path):
            result = alphafold_fetch("P04637")

        assert "summary" in result
        assert "P04637" in result["summary"]
        assert result["cached"] is False

    @patch("httpx.get")
    def test_returns_cached(self, mock_get, tmp_path):
        from ct.tools.structure import alphafold_fetch

        # Pre-create cached file
        cache_dir = tmp_path / ".ct" / "cache" / "alphafold"
        cache_dir.mkdir(parents=True)
        cached_file = cache_dir / "AF-P04637-F1-model_v4.pdb"
        cached_file.write_text("HEADER CACHED PDB\n")

        with patch("ct.tools.structure.Path.home", return_value=tmp_path):
            result = alphafold_fetch("P04637")

        assert result["cached"] is True
        mock_get.assert_not_called()

    @patch("httpx.get")
    def test_not_found(self, mock_get, tmp_path):
        from ct.tools.structure import alphafold_fetch

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        with patch("ct.tools.structure.Path.home", return_value=tmp_path):
            result = alphafold_fetch("INVALID_ID")

        assert "error" in result


class TestTernaryPredict:
    def test_missing_ternarypred(self, tmp_path):
        from ct.tools.structure import ternary_predict

        with patch("ct.tools.structure.TERNARYPRED_DIR", tmp_path):
            result = ternary_predict("CCO", "/tmp/target.pdb", "CRBN")

        assert "error" in result
        assert "not installed" in result["error"].lower()


class TestBatchScreen:
    def test_missing_ternarypred(self, tmp_path):
        from ct.tools.structure import batch_screen

        with patch("ct.tools.structure.TERNARYPRED_DIR", tmp_path):
            result = batch_screen("/tmp/compounds.csv", "/tmp/targets.csv")

        assert "error" in result


class TestCompound3d:
    @pytest.mark.skipif(
        not _rdkit_available(),
        reason="RDKit not installed",
    )
    def test_valid_smiles(self, tmp_path):
        from ct.tools.structure import compound_3d

        result = compound_3d("CCO", output_path=str(tmp_path / "ethanol.sdf"))

        assert "summary" in result
        assert result["n_atoms"] > 0
        assert (tmp_path / "ethanol.sdf").exists()

    @pytest.mark.skipif(
        not _rdkit_available(),
        reason="RDKit not installed",
    )
    def test_invalid_smiles(self):
        from ct.tools.structure import compound_3d

        result = compound_3d("NOT_A_SMILES_STRING_AT_ALL")
        assert "error" in result


# ─── Docking ──────────────────────────────────────────────────────


class TestDock:
    @pytest.mark.skipif(not _rdkit_available(), reason="RDKit not installed")
    def test_invalid_method(self, tmp_path):
        from ct.tools.structure import dock

        pdb = _write_mini_pdb(tmp_path)
        result = dock("CCO", str(pdb), method="invalid_method")
        assert "error" in result
        assert "invalid_method" in result["error"]

    @pytest.mark.skipif(not _rdkit_available(), reason="RDKit not installed")
    def test_invalid_smiles(self, tmp_path):
        from ct.tools.structure import dock

        pdb = _write_mini_pdb(tmp_path)
        # vina not installed => tries _prepare_ligand_pdbqt which fails on bad SMILES
        with patch("ct.tools.structure.subprocess") as mock_sub:
            mock_sub.run.side_effect = FileNotFoundError()
            result = dock("NOT_A_SMILES", str(pdb), method="vina")
        assert "error" in result

    @pytest.mark.skipif(not _rdkit_available(), reason="RDKit not installed")
    def test_target_pdb_not_found(self, tmp_path):
        """When target is neither a file nor a valid UniProt ID, should error."""
        from ct.tools.structure import dock

        with patch("ct.tools.structure.alphafold_fetch") as mock_af:
            mock_af.return_value = {"error": "not found"}
            result = dock("CCO", "/nonexistent/file.pdb", method="vina")
        assert "error" in result

    @pytest.mark.skipif(not _rdkit_available(), reason="RDKit not installed")
    @patch("ct.tools.compute._load_providers", return_value=MOCK_PROVIDERS)
    def test_diffdock_submits_cloud_job(self, mock_prov, tmp_path):
        """DiffDock should always submit as a cloud job."""
        from ct.tools.structure import dock
        import ct.tools.compute as compute_mod
        compute_mod._providers_data = None

        pdb = _write_mini_pdb(tmp_path)
        result = dock("CCO", str(pdb), method="diffdock", dry_run=True)
        assert "summary" in result
        assert result["method"] == "diffdock"
        assert "job" in result

    @pytest.mark.skipif(not _rdkit_available(), reason="RDKit not installed")
    @patch("ct.tools.compute._load_providers", return_value=MOCK_PROVIDERS)
    def test_vina_not_installed_submits_cloud(self, mock_prov, tmp_path):
        """When vina binary not found, should fall back to cloud submission."""
        from ct.tools.structure import dock
        import ct.tools.compute as compute_mod
        compute_mod._providers_data = None

        pdb = _write_mini_pdb(tmp_path)
        # Patch subprocess.run so vina --version check fails
        with patch("ct.tools.structure.subprocess") as mock_sub:
            mock_sub.run.side_effect = FileNotFoundError()
            mock_sub.TimeoutExpired = type("TimeoutExpired", (Exception,), {})
            result = dock("CCO", str(pdb), method="vina", dry_run=True)

        assert "summary" in result
        assert result.get("local") is False
        assert "job" in result

    @pytest.mark.skipif(not _rdkit_available(), reason="RDKit not installed")
    def test_vina_local_run(self, tmp_path):
        """When vina binary is found, should run locally and parse output."""
        from ct.tools.structure import dock

        pdb = _write_mini_pdb(tmp_path)

        vina_output = """\
mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b.| rmsd u.b.
-----+------------+----------+----------
   1       -8.3      0.000      0.000
   2       -7.9      1.234      2.345
   3       -7.5      2.000      3.456
"""
        with patch("ct.tools.structure.subprocess") as mock_sub:
            # First call: obabel (in _prepare_ligand_pdbqt) — fails so fallback writes PDBQT directly
            obabel_result = MagicMock()
            obabel_result.returncode = 1
            # Second call: vina --version succeeds
            version_result = MagicMock()
            version_result.returncode = 0
            # Third call: actual docking succeeds
            dock_result = MagicMock()
            dock_result.returncode = 0
            dock_result.stdout = vina_output
            dock_result.stderr = ""
            mock_sub.run.side_effect = [obabel_result, version_result, dock_result]
            mock_sub.TimeoutExpired = type("TimeoutExpired", (Exception,), {})

            result = dock("CCO", str(pdb), method="vina", n_poses=3)

        assert "summary" in result
        assert result["local"] is True
        assert len(result["poses"]) == 3
        assert result["poses"][0]["affinity_kcal_mol"] == -8.3


# ─── MD Simulation ──────────────────────────────────────────────

class TestMdSimulate:
    @patch("ct.tools.compute._load_providers", return_value=MOCK_PROVIDERS)
    def test_basic_submission(self, mock_prov, tmp_path):
        from ct.tools.structure import md_simulate
        import ct.tools.compute as compute_mod
        compute_mod._providers_data = None

        pdb = _write_mini_pdb(tmp_path)
        result = md_simulate(str(pdb), duration_ns=5.0, dry_run=True)

        assert "summary" in result
        assert "MD simulation" in result["summary"]
        assert result["config"]["duration_ns"] == 5.0
        assert result["config"]["temperature_k"] == 300.0
        assert result["config"]["forcefield"] == "amber14"
        assert "job" in result

    @patch("ct.tools.compute._load_providers", return_value=MOCK_PROVIDERS)
    def test_custom_forcefield(self, mock_prov, tmp_path):
        from ct.tools.structure import md_simulate
        import ct.tools.compute as compute_mod
        compute_mod._providers_data = None

        pdb = _write_mini_pdb(tmp_path)
        result = md_simulate(str(pdb), forcefield="charmm36", temperature_k=310.0, dry_run=True)

        assert result["config"]["forcefield"] == "charmm36"
        assert result["config"]["temperature_k"] == 310.0

    def test_invalid_forcefield(self, tmp_path):
        from ct.tools.structure import md_simulate

        pdb = _write_mini_pdb(tmp_path)
        result = md_simulate(str(pdb), forcefield="invalid_ff")
        assert "error" in result

    def test_negative_duration(self, tmp_path):
        from ct.tools.structure import md_simulate

        pdb = _write_mini_pdb(tmp_path)
        result = md_simulate(str(pdb), duration_ns=-1.0)
        assert "error" in result

    def test_pdb_not_found(self):
        from ct.tools.structure import md_simulate

        result = md_simulate("/nonexistent/protein.pdb")
        assert "error" in result


# ─── FEP ─────────────────────────────────────────────────────────

class TestFep:
    @pytest.mark.skipif(not _rdkit_available(), reason="RDKit not installed")
    @patch("ct.tools.compute._load_providers", return_value=MOCK_PROVIDERS)
    def test_basic_submission(self, mock_prov, tmp_path):
        from ct.tools.structure import fep
        import ct.tools.compute as compute_mod
        compute_mod._providers_data = None

        pdb = _write_mini_pdb(tmp_path)
        result = fep("CCO", "CCCO", str(pdb), dry_run=True)

        assert "summary" in result
        assert "FEP" in result["summary"]
        assert result["transformation"]["ligand_a"] == "CCO"
        assert result["transformation"]["ligand_b"] == "CCCO"
        assert "job" in result

    @pytest.mark.skipif(not _rdkit_available(), reason="RDKit not installed")
    def test_invalid_smiles_a(self, tmp_path):
        from ct.tools.structure import fep

        pdb = _write_mini_pdb(tmp_path)
        result = fep("INVALID", "CCO", str(pdb))
        assert "error" in result

    @pytest.mark.skipif(not _rdkit_available(), reason="RDKit not installed")
    def test_invalid_smiles_b(self, tmp_path):
        from ct.tools.structure import fep

        pdb = _write_mini_pdb(tmp_path)
        result = fep("CCO", "INVALID", str(pdb))
        assert "error" in result

    @pytest.mark.skipif(not _rdkit_available(), reason="RDKit not installed")
    def test_invalid_method(self, tmp_path):
        from ct.tools.structure import fep

        pdb = _write_mini_pdb(tmp_path)
        result = fep("CCO", "CCCO", str(pdb), method="invalid")
        assert "error" in result

    @pytest.mark.skipif(not _rdkit_available(), reason="RDKit not installed")
    def test_target_not_found(self):
        from ct.tools.structure import fep

        with patch("ct.tools.structure.alphafold_fetch") as mock_af:
            mock_af.return_value = {"error": "not found"}
            result = fep("CCO", "CCCO", "/nonexistent/target.pdb")
        assert "error" in result


# ─── Binding Site Detection ──────────────────────────────────────

class TestBindingSite:
    def test_geometric_detection(self, tmp_path):
        from ct.tools.structure import binding_site

        pdb = _write_mini_pdb(tmp_path)
        result = binding_site(str(pdb), method="geometric")

        assert "summary" in result
        assert result["method"] == "geometric"
        assert "pockets" in result

    def test_invalid_method(self, tmp_path):
        from ct.tools.structure import binding_site

        pdb = _write_mini_pdb(tmp_path)
        result = binding_site(str(pdb), method="invalid_method")
        assert "error" in result

    def test_target_resolution_from_uniprot(self, tmp_path):
        """When given a UniProt ID instead of file path, should resolve via AlphaFold."""
        from ct.tools.structure import binding_site

        pdb = _write_mini_pdb(tmp_path)
        with patch("ct.tools.structure.alphafold_fetch") as mock_af:
            mock_af.return_value = {"path": str(pdb)}
            result = binding_site("P04637", method="geometric")

        assert "summary" in result
        assert "pockets" in result

    def test_fpocket_fallback_to_geometric(self, tmp_path):
        """When fpocket is not installed, should fall back to geometric."""
        from ct.tools.structure import binding_site

        pdb = _write_mini_pdb(tmp_path)
        # fpocket won't be installed in test env
        result = binding_site(str(pdb), method="fpocket")

        assert "summary" in result
        assert result["method"] == "geometric"
        assert "pockets" in result

    def test_fpocket_runs_when_installed(self, tmp_path):
        """When fpocket is installed and succeeds, should parse its output."""
        from ct.tools.structure import binding_site

        pdb = _write_mini_pdb(tmp_path)
        protein_name = pdb.stem

        # Create mock fpocket output
        out_dir = tmp_path / f"{protein_name}_out"
        out_dir.mkdir()
        info_file = out_dir / f"{protein_name}_info.txt"
        info_file.write_text(
            "Pocket 1 :\n"
            "Score : 0.85\n"
            "Volume : 423.5\n"
            "Pocket 2 :\n"
            "Score : 0.62\n"
            "Volume : 215.0\n"
        )

        with patch("ct.tools.structure.subprocess") as mock_sub:
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_sub.run.return_value = mock_proc
            mock_sub.TimeoutExpired = type("TimeoutExpired", (Exception,), {})

            result = binding_site(str(pdb), method="fpocket")

        assert "summary" in result
        assert result["method"] == "fpocket"
        assert len(result["pockets"]) == 2


# ─── Helper function tests ───────────────────────────────────────

class TestResolvePdb:
    def test_existing_file(self, tmp_path):
        from ct.tools.structure import _resolve_pdb

        pdb = _write_mini_pdb(tmp_path)
        result = _resolve_pdb(str(pdb))
        assert "path" in result
        assert result["path"] == str(pdb)

    def test_uniprot_id_success(self, tmp_path):
        from ct.tools.structure import _resolve_pdb

        pdb = _write_mini_pdb(tmp_path)
        with patch("ct.tools.structure.alphafold_fetch") as mock_af:
            mock_af.return_value = {"path": str(pdb)}
            result = _resolve_pdb("P04637")
        assert "path" in result

    def test_uniprot_id_failure(self):
        from ct.tools.structure import _resolve_pdb

        with patch("ct.tools.structure.alphafold_fetch") as mock_af:
            mock_af.return_value = {"error": "not found"}
            result = _resolve_pdb("INVALID")
        assert "error" in result


class TestDetectSearchBox:
    def test_computes_box(self, tmp_path):
        from ct.tools.structure import _detect_search_box

        pdb = _write_mini_pdb(tmp_path)
        box = _detect_search_box(str(pdb))
        assert "center_x" in box
        assert "size_x" in box
        # Size should be positive
        assert box["size_x"] > 0
        assert box["size_y"] > 0
        assert box["size_z"] > 0

    def test_empty_pdb(self, tmp_path):
        from ct.tools.structure import _detect_search_box

        empty = tmp_path / "empty.pdb"
        empty.write_text("HEADER EMPTY\nEND\n")
        result = _detect_search_box(str(empty))
        assert "error" in result


class TestGeometricPocketDetection:
    def test_detects_pockets(self, tmp_path):
        from ct.tools.structure import _geometric_pocket_detection

        pdb = _write_mini_pdb(tmp_path)
        pockets = _geometric_pocket_detection(str(pdb), min_residues=2, distance_cutoff=15.0)
        assert isinstance(pockets, list)
        # With our test PDB, we should find at least one pocket
        if pockets:
            p = pockets[0]
            assert "pocket_id" in p
            assert "n_residues" in p
            assert "volume_approx_A3" in p
            assert "druggability_score" in p

    def test_empty_pdb(self, tmp_path):
        from ct.tools.structure import _geometric_pocket_detection

        empty = tmp_path / "empty.pdb"
        empty.write_text("HEADER EMPTY\nEND\n")
        pockets = _geometric_pocket_detection(str(empty))
        assert pockets == []


# ─── ImmuneBuilder ──────────────────────────────────────────────

class TestImmuneBuilderPredict:
    def test_missing_dependencies(self):
        from ct.tools.structure import immunebuilder_predict

        with patch.dict("sys.modules", {"ImmuneBuilder": None}):
            result = immunebuilder_predict({"H": "SEQ"})
        
        assert "error" in result
        assert "not installed" in result["error"].lower()

    def test_invalid_model_type(self):
        from ct.tools.structure import immunebuilder_predict

        mock_ib = MagicMock()
        with patch.dict("sys.modules", {"ImmuneBuilder": mock_ib}):
            result = immunebuilder_predict({"H": "SEQ"}, model_type="invalid_type")

        assert "error" in result
        assert "Invalid model_type" in result["error"]

    def test_successful_prediction(self):
        from ct.tools.structure import immunebuilder_predict

        mock_predictor = MagicMock()
        mock_out = MagicMock()
        mock_predictor.predict.return_value = mock_out

        mock_ib = MagicMock()
        mock_ib.ABodyBuilder2.return_value = mock_predictor

        with patch.dict("sys.modules", {"ImmuneBuilder": mock_ib}):
            result = immunebuilder_predict({"H": "SEQ", "L": "SEQ"}, model_type="antibody", output_path="/custom/path.pdb")

        assert "summary" in result
        assert result["output_path"] == "/custom/path.pdb"
        mock_out.save.assert_called_once_with("/custom/path.pdb")
