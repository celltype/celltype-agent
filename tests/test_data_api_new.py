"""Tests for data_api.ttd_search."""

from unittest.mock import MagicMock, patch

import pytest

from ct.tools.data_api import ttd_search


class TestTTDSearch:
    """Tests for the TTD search tool."""

    def test_empty_query(self):
        """Empty query should return an error."""
        result = ttd_search(query="")
        assert "error" in result
        assert "requires" in result["summary"].lower() or "required" in result["summary"].lower()

    def test_invalid_search_type(self):
        """Invalid search_type should return an error."""
        result = ttd_search(query="BRAF", search_type="protein")
        assert "error" in result
        assert "invalid" in result["summary"].lower()

    def test_successful_target_search(self):
        """Successful target search returns parsed results."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {
                "ttd_id": "TTD-T12345",
                "name": "BRAF",
                "type": "target",
                "uniprot_id": "P15056",
                "gene_name": "BRAF",
                "clinical_status": "Successful target",
                "drugs": ["vemurafenib", "dabrafenib"],
                "diseases": ["melanoma"],
                "description": "Serine/threonine-protein kinase B-raf",
            }
        ]

        with patch("ct.tools.data_api._http_get", return_value=mock_resp):
            result = ttd_search(query="BRAF", search_type="target")

        assert result["n_results"] == 1
        assert result["search_type"] == "target"
        assert result["results"][0]["name"] == "BRAF"
        assert result["results"][0]["id"] == "TTD-T12345"
        assert "BRAF" in result["summary"]

    def test_successful_drug_search(self):
        """Successful drug search returns results."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {
                    "ttd_id": "D12345",
                    "drug_name": "Imatinib",
                    "clinical_status": "Approved",
                    "description": "BCR-ABL inhibitor",
                }
            ]
        }

        with patch("ct.tools.data_api._http_get", return_value=mock_resp):
            result = ttd_search(query="imatinib", search_type="drug")

        assert result["n_results"] == 1
        assert result["search_type"] == "drug"
        assert result["results"][0]["name"] == "Imatinib"

    def test_successful_disease_search(self):
        """Successful disease search returns results."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {
                    "id": "DIS001",
                    "name": "Melanoma",
                    "type": "disease",
                    "description": "Skin cancer",
                }
            ]
        }

        with patch("ct.tools.data_api._http_get", return_value=mock_resp):
            result = ttd_search(query="melanoma", search_type="disease")

        assert result["n_results"] == 1
        assert result["results"][0]["name"] == "Melanoma"

    def test_api_http_error(self):
        """HTTP error from TTD should return error."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500

        with patch("ct.tools.data_api._http_get", return_value=mock_resp):
            result = ttd_search(query="BRAF", search_type="target")

        assert "error" in result
        assert "500" in result["summary"]

    def test_api_network_error_with_fallback_failure(self):
        """When both primary and fallback endpoints fail, should return error."""
        with patch("ct.tools.data_api._http_get", side_effect=Exception("Connection refused")):
            result = ttd_search(query="BRAF", search_type="target")

        assert "error" in result
        assert "failed" in result["summary"].lower()

    def test_no_results(self):
        """When API returns empty list, should report no results."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []

        with patch("ct.tools.data_api._http_get", return_value=mock_resp):
            result = ttd_search(query="XYZNONEXISTENT", search_type="target")

        assert result["n_results"] == 0
        assert "No TTD results" in result["summary"]

    def test_invalid_json_response(self):
        """When API returns invalid JSON, should return error."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("Invalid JSON")

        with patch("ct.tools.data_api._http_get", return_value=mock_resp):
            result = ttd_search(query="BRAF", search_type="target")

        assert "error" in result
        assert "invalid" in result["summary"].lower()

    def test_dict_response_with_targets_key(self):
        """When API returns dict with 'targets' key, should parse correctly."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "targets": [
                {"ttd_id": "T001", "name": "CDK4", "gene_name": "CDK4", "description": "Cyclin-dependent kinase 4"},
                {"ttd_id": "T002", "name": "CDK6", "gene_name": "CDK6", "description": "Cyclin-dependent kinase 6"},
            ]
        }

        with patch("ct.tools.data_api._http_get", return_value=mock_resp):
            result = ttd_search(query="CDK", search_type="target")

        assert result["n_results"] == 2
        assert result["results"][0]["name"] == "CDK4"
        assert result["results"][1]["name"] == "CDK6"
