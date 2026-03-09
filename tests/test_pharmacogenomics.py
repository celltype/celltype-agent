"""Tests for pharmacogenomics tools: pharmgkb_lookup, cpic_guidelines."""

import pytest
from unittest.mock import patch, MagicMock


# ─── pharmacogenomics.pharmgkb_lookup ─────────────────────────


class TestPharmgkbLookup:
    """Tests for pharmacogenomics.pharmgkb_lookup."""

    def test_empty_query_returns_error(self):
        from ct.tools.pharmacogenomics import pharmgkb_lookup
        result = pharmgkb_lookup(query="")
        assert "error" in result

    def test_gene_lookup_success(self):
        """Gene lookup returns PharmGKB annotations."""
        mock_entity = {
            "data": [{
                "id": "PA124",
                "name": "CYP2D6",
                "symbol": "CYP2D6",
                "crossReferences": [
                    {"resource": "NCBI Gene", "resourceId": "1565"},
                ],
            }],
        }
        mock_annotations = {
            "data": [{
                "id": "CA123",
                "level": {"term": "1A"},
                "phenotypeCategories": ["Dosage"],
                "relatedChemicals": [{"name": "codeine"}],
                "relatedGenes": [{"symbol": "CYP2D6"}],
                "text": "CYP2D6 poor metabolizers should avoid codeine.",
            }],
        }
        mock_labels = {"data": []}
        mock_guidelines = {"data": []}

        responses = []

        def mock_request_json(*args, **kwargs):
            # First call: entity search by name
            # If name search fails, second call is by symbol (fallback)
            # Then: clinical annotations, labels, guidelines
            url = args[1] if len(args) > 1 else kwargs.get("url", "")
            params = kwargs.get("params", {})

            if "/gene" in url or "/chemical" in url or "/variant" in url:
                if "clinicalAnnotation" not in url and "drugLabel" not in url and "guideline" not in url:
                    return mock_entity, None
            if "clinicalAnnotation" in url:
                return mock_annotations, None
            if "drugLabel" in url:
                return mock_labels, None
            if "guideline" in url:
                return mock_guidelines, None
            return mock_entity, None

        with patch("ct.tools.pharmacogenomics.request_json", side_effect=mock_request_json):
            from ct.tools.pharmacogenomics import pharmgkb_lookup
            result = pharmgkb_lookup(query="CYP2D6", entity_type="gene")

        assert "summary" in result
        assert result["found"] is True
        assert result["pharmgkb_id"] == "PA124"
        assert len(result["clinical_annotations"]) > 0

    def test_drug_lookup(self):
        """Drug entity lookup works."""
        mock_entity = {
            "data": [{
                "id": "PA449015",
                "name": "codeine",
                "symbol": "",
                "crossReferences": [],
            }],
        }
        mock_empty = {"data": []}

        call_count = [0]

        def mock_request_json(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 1:
                return mock_entity, None
            return mock_empty, None

        with patch("ct.tools.pharmacogenomics.request_json", side_effect=mock_request_json):
            from ct.tools.pharmacogenomics import pharmgkb_lookup
            result = pharmgkb_lookup(query="codeine", entity_type="drug")

        assert "summary" in result
        assert result["found"] is True

    def test_not_found(self):
        """Returns found=False when no results."""
        with patch("ct.tools.pharmacogenomics.request_json", return_value=({"data": []}, None)):
            from ct.tools.pharmacogenomics import pharmgkb_lookup
            result = pharmgkb_lookup(query="nonexistent_gene_xyz")

        assert result.get("found") is False

    def test_api_error(self):
        """API errors are handled gracefully."""
        with patch("ct.tools.pharmacogenomics.request_json", return_value=(None, "Timeout")):
            from ct.tools.pharmacogenomics import pharmgkb_lookup
            result = pharmgkb_lookup(query="CYP2D6")

        assert "error" in result

    def test_invalid_entity_type_defaults_to_gene(self):
        """Invalid entity_type falls back to 'gene'."""
        mock_entity = {"data": [{"id": "PA1", "name": "TEST", "symbol": "TEST", "crossReferences": []}]}
        mock_empty = {"data": []}

        call_count = [0]

        def mock_request_json(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 1:
                return mock_entity, None
            return mock_empty, None

        with patch("ct.tools.pharmacogenomics.request_json", side_effect=mock_request_json):
            from ct.tools.pharmacogenomics import pharmgkb_lookup
            result = pharmgkb_lookup(query="TEST", entity_type="invalid")

        assert "summary" in result


# ─── pharmacogenomics.cpic_guidelines ─────────────────────────


class TestCpicGuidelines:
    """Tests for pharmacogenomics.cpic_guidelines."""

    def test_no_params_returns_error(self):
        from ct.tools.pharmacogenomics import cpic_guidelines
        result = cpic_guidelines()
        assert "error" in result

    def test_gene_lookup_success(self):
        """Gene-based CPIC lookup returns guidelines."""
        mock_pairs = [
            {
                "genesymbol": "CYP2D6",
                "drugname": "codeine",
                "cpiclevel": "A",
                "pgkblevelofevidence": "1A",
                "pgxonfdalabel": "Yes",
                "url": "https://cpicpgx.org/guidelines/cpic-guideline-for-codeine/",
                "guidelineid": "cpic-123",
            },
        ]
        mock_recs = [
            {
                "lookupkey": {"CYP2D6": "Poor Metabolizer"},
                "drugrecommendation": "Avoid codeine. Use alternative analgesic.",
                "classification": "Strong",
                "strength": "Strong",
                "activityscore": "0",
                "phenotypes": {"CYP2D6": "Poor Metabolizer"},
            },
        ]

        call_count = [0]

        def mock_request_json(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_pairs, None
            return mock_recs, None

        with patch("ct.tools.pharmacogenomics.request_json", side_effect=mock_request_json):
            from ct.tools.pharmacogenomics import cpic_guidelines
            result = cpic_guidelines(gene="CYP2D6")

        assert "summary" in result
        assert result["found"] is True
        assert result["n_pairs"] == 1
        assert result["guidelines"][0]["gene"] == "CYP2D6"

    def test_drug_lookup(self):
        """Drug-based CPIC lookup works."""
        mock_pairs = [
            {
                "genesymbol": "CYP2C19",
                "drugname": "clopidogrel",
                "cpiclevel": "A",
                "pgkblevelofevidence": "1A",
                "pgxonfdalabel": "Yes",
                "url": "",
                "guidelineid": "",
            },
        ]

        with patch("ct.tools.pharmacogenomics.request_json", side_effect=[
            (mock_pairs, None),
            ([], None),
        ]):
            from ct.tools.pharmacogenomics import cpic_guidelines
            result = cpic_guidelines(drug="clopidogrel")

        assert result["found"] is True

    def test_not_found(self):
        """No CPIC guidelines returns found=False."""
        with patch("ct.tools.pharmacogenomics.request_json", return_value=([], None)):
            from ct.tools.pharmacogenomics import cpic_guidelines
            result = cpic_guidelines(gene="FAKEGENE")

        assert result.get("found") is False

    def test_api_error(self):
        """API errors handled gracefully."""
        with patch("ct.tools.pharmacogenomics.request_json", return_value=(None, "Connection error")):
            from ct.tools.pharmacogenomics import cpic_guidelines
            result = cpic_guidelines(gene="CYP2D6")

        assert "error" in result
