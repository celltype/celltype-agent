"""Tests for protein_design.antifold tool."""

import pytest
from unittest.mock import patch, MagicMock
import os
import tempfile


class TestAntiFold:
    def test_missing_pdb_path(self):
        from ct.tools.protein_design import antifold
        result = antifold(pdb_path="")
        assert "error" in result
        assert "requires" in result["summary"].lower()

    def test_none_pdb_path(self):
        from ct.tools.protein_design import antifold
        result = antifold(pdb_path=None)
        assert "error" in result

    def test_file_not_found(self):
        from ct.tools.protein_design import antifold
        result = antifold(pdb_path="/nonexistent/path/antibody.pdb")
        assert "error" in result
        assert "not found" in result["summary"].lower()

    def test_not_installed(self):
        """antifold package is not installed, should return install instructions."""
        # Create a temporary PDB file so we get past the file check
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as f:
            f.write("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
            pdb_path = f.name
        try:
            from ct.tools.protein_design import antifold
            result = antifold(pdb_path=pdb_path)
            assert "error" in result
            assert "not installed" in result["summary"].lower()
            assert "install_instructions" in result
            assert "dry_run" in result
            assert result["dry_run"]["pdb_path"] == pdb_path
        finally:
            os.unlink(pdb_path)

    def test_successful_prediction(self):
        """Mock antifold to test successful prediction path."""
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as f:
            f.write("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
            pdb_path = f.name

        try:
            mock_predict = MagicMock(return_value=[
                {"sequence": "CARGGYW", "score": 0.95},
                {"sequence": "CARGSYW", "score": 0.88},
            ])
            mock_antifold = MagicMock()
            mock_antifold_main = MagicMock()
            mock_antifold_main.predict = mock_predict

            with patch.dict("sys.modules", {
                "antifold": mock_antifold,
                "antifold.main": mock_antifold_main,
            }):
                from ct.tools.protein_design import antifold
                result = antifold(pdb_path=pdb_path, chains="H", num_sequences=5)

            assert "error" not in result
            assert result["n_sequences"] == 2
            assert "AntiFold" in result["summary"]
        finally:
            os.unlink(pdb_path)
