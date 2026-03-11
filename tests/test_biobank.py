"""Tests for biobank tools: ukb_field_search."""

import pytest


# ─── biobank.ukb_field_search ──────────────────────────────────


class TestUkbFieldSearch:
    """Tests for biobank.ukb_field_search."""

    def test_empty_query(self):
        from ct.tools.biobank import ukb_field_search
        result = ukb_field_search(query="")
        assert "error" in result
        assert "summary" in result

    def test_search_blood_pressure(self):
        from ct.tools.biobank import ukb_field_search
        result = ukb_field_search(query="blood pressure")
        assert result["n_results"] > 0
        titles = [r["title"] for r in result["results"]]
        assert any("blood pressure" in t.lower() for t in titles)

    def test_search_cholesterol(self):
        from ct.tools.biobank import ukb_field_search
        result = ukb_field_search(query="cholesterol")
        assert result["n_results"] > 0

    def test_search_bmi(self):
        from ct.tools.biobank import ukb_field_search
        result = ukb_field_search(query="BMI")
        assert result["n_results"] > 0
        assert any(r["field_id"] == 21001 for r in result["results"])

    def test_category_filter(self):
        from ct.tools.biobank import ukb_field_search
        result = ukb_field_search(query="blood", category_filter="Blood biochemistry")
        for r in result["results"]:
            assert r["category"] == "Blood biochemistry"

    def test_no_matches(self):
        from ct.tools.biobank import ukb_field_search
        result = ukb_field_search(query="quantum entanglement xyz")
        assert result["n_results"] == 0

    def test_max_results(self):
        from ct.tools.biobank import ukb_field_search
        result = ukb_field_search(query="blood", max_results=3)
        assert len(result["results"]) <= 3

    def test_result_structure(self):
        from ct.tools.biobank import ukb_field_search
        result = ukb_field_search(query="haemoglobin")
        assert result["n_results"] > 0
        r = result["results"][0]
        assert "field_id" in r
        assert "title" in r
        assert "category" in r
        assert "value_type" in r
        assert "participants" in r
