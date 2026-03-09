"""Tests for protein design tools: bindcraft, antibody_design.

Note: proteinmpnn and rfdiffusion are now container tools (design.proteinmpnn,
design.rfdiffusion) and tested separately.
"""

import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock


# ─── protein_design.bindcraft ─────────────────────────────────


class TestBindCraft:
    """Tests for protein_design.bindcraft."""

    def test_empty_target_returns_error(self):
        from ct.tools.protein_design import bindcraft
        result = bindcraft(target_pdb="")
        assert "error" in result

    def test_missing_target_returns_error(self):
        from ct.tools.protein_design import bindcraft
        result = bindcraft(target_pdb="/nonexistent/target.pdb")
        assert "error" in result

    def test_not_installed_returns_dry_run(self):
        """BindCraft not installed returns install instructions and dry run."""
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
            f.write("ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00  0.00           N\n")
            pdb_path = f.name

        try:
            with patch("ct.tools.protein_design._check_tool_installed", return_value=None):
                from ct.tools.protein_design import bindcraft
                result = bindcraft(target_pdb=pdb_path, hotspot_residues="A30", num_designs=2, binder_length=60)

            assert "install_instructions" in result
            assert "dry_run" in result
            assert result["dry_run"]["binder_length"] == 60
        finally:
            os.unlink(pdb_path)


# ─── protein_design.antibody_design ───────────────────────────


class TestAntibodyDesign:
    """Tests for protein_design.antibody_design."""

    def test_empty_heavy_chain_returns_error(self):
        from ct.tools.protein_design import antibody_design
        result = antibody_design(heavy_chain="")
        assert "error" in result

    def test_invalid_characters_returns_error(self):
        from ct.tools.protein_design import antibody_design
        result = antibody_design(heavy_chain="MKTL123!!!")
        assert "error" in result
        assert "invalid" in result["error"].lower() or "Invalid" in result["error"]

    def test_anarci_not_installed(self):
        """When ANARCI is not installed, returns basic analysis and instructions."""
        heavy = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR"

        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name in ("anarci",):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            from ct.tools.protein_design import antibody_design
            result = antibody_design(heavy_chain=heavy)

        assert "summary" in result
        assert result["computed_locally"] is False
        assert "install_instructions" in result
        assert result["heavy_chain_length"] == len(heavy)

    def test_anarci_success_mock(self):
        """Successful antibody analysis with mocked ANARCI."""
        import sys

        heavy = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR"

        # Create mock ANARCI module
        mock_anarci_mod = MagicMock()

        # ANARCI returns numbered results
        numbered_chain = []
        for i, aa in enumerate(heavy[:40]):
            numbered_chain.append(((i + 1, " "), aa))
        # Add CDR3-like region (positions 105-117 in IMGT)
        for i, aa in enumerate("ARDYVWGSYRP"):
            numbered_chain.append(((105 + i, " "), aa))

        mock_anarci_mod.anarci.return_value = ([[numbered_chain]], None, None)

        with patch.dict(sys.modules, {"anarci": mock_anarci_mod}):
            from ct.tools.protein_design import antibody_design
            result = antibody_design(heavy_chain=heavy, cdr_to_optimize="H3", num_designs=4)

        assert "summary" in result
        assert result["computed_locally"] is True
        assert "cdr_regions" in result
        assert "numbering_scheme" in result

    def test_with_light_chain(self):
        """Works when both heavy and light chains provided (without ANARCI)."""
        heavy = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMS"
        light = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLN"

        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name in ("anarci",):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            from ct.tools.protein_design import antibody_design
            result = antibody_design(heavy_chain=heavy, light_chain=light)

        assert result["heavy_chain_length"] == len(heavy)
        assert result["light_chain_length"] == len(light)
