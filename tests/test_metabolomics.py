"""Tests for metabolomics tools: hmdb_lookup, metabolite_search."""

import pytest
from unittest.mock import patch, MagicMock


# ─── metabolomics.hmdb_lookup ─────────────────────────────────


class TestHmdbLookup:
    """Tests for metabolomics.hmdb_lookup."""

    def test_empty_query_returns_error(self):
        from ct.tools.metabolomics import hmdb_lookup
        result = hmdb_lookup(query="")
        assert "error" in result
        assert "summary" in result

    def test_hmdb_id_lookup_success(self):
        """HMDB ID lookup returns metabolite data."""
        mock_response = {
            "name": "1-Methylhistidine",
            "chemical_formula": "C7H11N3O2",
            "average_molecular_weight": 169.18,
            "monisotopic_molecular_weight": 169.085,
            "smiles": "CN1C=NC(CC(N)C(=O)O)=C1",
            "inchikey": "BRMWTNUJHUMWMS-LURJTMIESA-N",
            "state": "Solid",
            "description": "1-Methylhistidine is a methylated amino acid.",
            "taxonomy": {
                "kingdom": "Organic compounds",
                "super_class": "Amino acids",
                "direct_parent": "Histidine and derivatives",
            },
            "biological_properties": {
                "pathways": [
                    {"name": "Histidine Metabolism", "smpdb_id": "SMP0000044", "kegg_map_id": "map00340"},
                ],
            },
            "biospecimen_locations": [
                {"biospecimen": "Blood"},
                {"biospecimen": "Urine"},
            ],
            "normal_concentrations": [
                {"biospecimen": "Blood", "concentration_value": "0.5-2.0", "concentration_units": "uM",
                 "subject_age": "Adult", "subject_sex": "Both", "subject_condition": "Normal"},
            ],
            "diseases": [
                {"name": "Histidinemia", "omim_id": "235800", "references": []},
            ],
            "protein_associations": [],
            "ontology": {"status": "Detected", "origins": ["Endogenous"], "biofunctions": []},
        }

        with patch("ct.tools.metabolomics.request_json", return_value=(mock_response, None)):
            from ct.tools.metabolomics import hmdb_lookup
            result = hmdb_lookup(query="HMDB0000001")

        assert "summary" in result
        assert result["found"] is True
        assert result["name"] == "1-Methylhistidine"
        assert result["chemical_formula"] == "C7H11N3O2"
        assert len(result["pathways"]) > 0

    def test_hmdb_id_normalization(self):
        """Short HMDB IDs are normalized to 7-digit format."""
        mock_response = {"name": "Test", "chemical_formula": "C1H1"}

        with patch("ct.tools.metabolomics.request_json", return_value=(mock_response, None)):
            from ct.tools.metabolomics import hmdb_lookup
            result = hmdb_lookup(query="HMDB1")

        assert result.get("hmdb_id") == "HMDB0000001"

    def test_name_search_fallback(self):
        """When query is a name, metabolite_search is called first."""
        with patch("ct.tools.metabolomics.metabolite_search") as mock_search:
            mock_search.return_value = {
                "n_results": 1,
                "results": [{"hmdb_id": "HMDB0000122", "name": "Glucose"}],
            }
            mock_response = {"name": "Glucose", "chemical_formula": "C6H12O6"}
            with patch("ct.tools.metabolomics.request_json", return_value=(mock_response, None)):
                from ct.tools.metabolomics import hmdb_lookup
                result = hmdb_lookup(query="glucose")

            mock_search.assert_called_once()
            assert result.get("hmdb_id") == "HMDB0000122"

    def test_api_error_returns_error(self):
        """API errors are handled gracefully."""
        with patch("ct.tools.metabolomics.request_json", return_value=(None, "Connection timeout")):
            from ct.tools.metabolomics import hmdb_lookup
            result = hmdb_lookup(query="HMDB0000001")

        assert "error" in result

    def test_not_found_returns_not_found(self):
        """Name search with no results returns found=False."""
        with patch("ct.tools.metabolomics.metabolite_search") as mock_search:
            mock_search.return_value = {"n_results": 0, "results": []}
            from ct.tools.metabolomics import hmdb_lookup
            result = hmdb_lookup(query="nonexistent_metabolite_xyz")

        assert result.get("found") is False


# ─── metabolomics.metabolite_search ───────────────────────────


class TestMetaboliteSearch:
    """Tests for metabolomics.metabolite_search."""

    def test_no_params_returns_error(self):
        from ct.tools.metabolomics import metabolite_search
        result = metabolite_search()
        assert "error" in result

    def test_name_search(self):
        """Name search returns results."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"hmdb_id": "HMDB0000122", "name": "D-Glucose", "chemical_formula": "C6H12O6", "average_molecular_weight": 180.16},
            {"hmdb_id": "HMDB0003345", "name": "L-Glucose", "chemical_formula": "C6H12O6", "average_molecular_weight": 180.16},
        ]

        with patch("ct.tools.metabolomics.request", return_value=(mock_resp, None)):
            from ct.tools.metabolomics import metabolite_search
            result = metabolite_search(name="glucose")

        assert "summary" in result
        assert result["n_results"] >= 1

    def test_mass_tolerance_clamped(self):
        """Mass tolerance is clamped to valid range."""
        with patch("ct.tools.metabolomics.request", return_value=(MagicMock(status_code=404), None)), \
             patch("ct.tools.metabolomics.request_json", return_value=([], None)):
            from ct.tools.metabolomics import metabolite_search
            result = metabolite_search(mass=180.06, mass_tolerance=100.0)

        # Should not crash; tolerance is clamped to 1.0
        assert "summary" in result

    def test_formula_search(self):
        """Formula search returns results."""
        with patch("ct.tools.metabolomics.request_json", return_value=(
            [{"accession": "HMDB0000122", "name": "Glucose", "chemical_formula": "C6H12O6"}], None
        )):
            from ct.tools.metabolomics import metabolite_search
            result = metabolite_search(formula="C6H12O6")

        assert "summary" in result
        assert result["n_results"] >= 1

    def test_deduplication(self):
        """Duplicate HMDB IDs are removed."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"hmdb_id": "HMDB0000122", "name": "Glucose", "chemical_formula": "C6H12O6"},
            {"hmdb_id": "HMDB0000122", "name": "Glucose", "chemical_formula": "C6H12O6"},
        ]

        with patch("ct.tools.metabolomics.request", return_value=(mock_resp, None)):
            from ct.tools.metabolomics import metabolite_search
            result = metabolite_search(name="glucose")

        assert result["n_results"] == 1
