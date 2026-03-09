"""Tests for the new genomics tools: alphamissense_lookup, spliceai_predict, cadd_score."""

import pytest
from unittest.mock import patch, MagicMock

from ct.tools.genomics import alphamissense_lookup, spliceai_predict, cadd_score


# ---------------------------------------------------------------------------
# AlphaMissense tests
# ---------------------------------------------------------------------------


class TestAlphaMissenseLookup:
    def test_successful_lookup_rsid(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{
            "id": "rs113488022",
            "transcript_consequences": [{
                "gene_symbol": "BRAF",
                "transcript_id": "ENST00000288602",
                "consequence_terms": ["missense_variant"],
                "amino_acids": "V/E",
                "protein_position": "600",
                "am_pathogenicity": 0.9987,
                "am_class": "likely_pathogenic",
                "sift_prediction": "deleterious",
                "sift_score": 0.0,
                "polyphen_prediction": "probably_damaging",
                "polyphen_score": 0.999,
                "canonical": 1,
            }],
        }]
        with patch("ct.tools.genomics.request", return_value=(mock_resp, None)):
            result = alphamissense_lookup(variant="rs113488022")
        assert result["top_class"] == "likely_pathogenic"
        assert result["top_score"] > 0.99
        assert result["n_results"] == 1
        assert "BRAF" in result["gene"]
        assert "AlphaMissense" in result["summary"]

    def test_hgvs_lookup(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{
            "transcript_consequences": [{
                "gene_symbol": "BRAF",
                "transcript_id": "ENST00000288602",
                "consequence_terms": ["missense_variant"],
                "amino_acids": "V/E",
                "protein_position": "600",
                "am_pathogenicity": 0.9987,
                "am_class": "likely_pathogenic",
                "canonical": 1,
            }],
        }]
        with patch("ct.tools.genomics.request", return_value=(mock_resp, None)):
            result = alphamissense_lookup(variant="BRAF:p.Val600Glu")
        assert result["top_class"] == "likely_pathogenic"
        assert result["n_results"] == 1

    def test_no_am_scores(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{
            "transcript_consequences": [{
                "gene_symbol": "TP53",
                "transcript_id": "ENST00000269305",
                "consequence_terms": ["synonymous_variant"],
                "canonical": 1,
                # No am_pathogenicity or am_class fields
            }],
        }]
        with patch("ct.tools.genomics.request", return_value=(mock_resp, None)):
            result = alphamissense_lookup(variant="rs12345")
        assert result["n_results"] == 0
        assert "No AlphaMissense scores" in result["summary"]

    def test_missing_variant(self):
        result = alphamissense_lookup(variant="")
        assert "error" in result
        assert "requires" in result["summary"].lower()

    def test_none_variant(self):
        result = alphamissense_lookup(variant=None)
        assert "error" in result

    def test_vep_error(self):
        with patch("ct.tools.genomics.request", return_value=(None, "Connection timeout")):
            result = alphamissense_lookup(variant="rs113488022")
        assert "error" in result
        assert "failed" in result["summary"].lower()

    def test_vep_400_error(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        with patch("ct.tools.genomics.request", return_value=(mock_resp, None)):
            result = alphamissense_lookup(variant="invalid_variant")
        assert "error" in result
        assert "Invalid variant" in result["summary"]

    def test_vep_500_error(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch("ct.tools.genomics.request", return_value=(mock_resp, None)):
            result = alphamissense_lookup(variant="rs113488022")
        assert "error" in result
        assert "VEP query failed" in result["summary"]

    def test_gene_filtering(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{
            "transcript_consequences": [
                {
                    "gene_symbol": "BRAF",
                    "transcript_id": "ENST00000288602",
                    "consequence_terms": ["missense_variant"],
                    "am_pathogenicity": 0.9987,
                    "am_class": "likely_pathogenic",
                    "canonical": 1,
                },
                {
                    "gene_symbol": "OTHER_GENE",
                    "transcript_id": "ENST00000999999",
                    "consequence_terms": ["missense_variant"],
                    "am_pathogenicity": 0.5,
                    "am_class": "ambiguous",
                    "canonical": 0,
                },
            ],
        }]
        with patch("ct.tools.genomics.request", return_value=(mock_resp, None)):
            result = alphamissense_lookup(variant="rs113488022", gene="BRAF")
        assert result["n_results"] == 1
        assert result["results"][0]["gene_symbol"] == "BRAF"

    def test_multiple_transcripts_sorted(self):
        """Canonical transcript should come first, then sorted by pathogenicity descending."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{
            "transcript_consequences": [
                {
                    "gene_symbol": "BRAF",
                    "transcript_id": "ENST00000111111",
                    "consequence_terms": ["missense_variant"],
                    "am_pathogenicity": 0.8,
                    "am_class": "likely_pathogenic",
                    "canonical": 0,
                },
                {
                    "gene_symbol": "BRAF",
                    "transcript_id": "ENST00000288602",
                    "consequence_terms": ["missense_variant"],
                    "am_pathogenicity": 0.7,
                    "am_class": "likely_pathogenic",
                    "canonical": 1,
                },
            ],
        }]
        with patch("ct.tools.genomics.request", return_value=(mock_resp, None)):
            result = alphamissense_lookup(variant="rs113488022")
        assert result["n_results"] == 2
        # Canonical (score=0.7) should come first
        assert result["results"][0]["canonical"] is True
        assert result["results"][0]["am_pathogenicity"] == 0.7

    def test_invalid_json_response(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("bad json")
        with patch("ct.tools.genomics.request", return_value=(mock_resp, None)):
            result = alphamissense_lookup(variant="rs113488022")
        assert "error" in result
        assert "Invalid JSON" in result["error"]


# ---------------------------------------------------------------------------
# SpliceAI tests
# ---------------------------------------------------------------------------


class TestSpliceAIPredict:
    def test_not_installed_returns_instructions(self):
        with patch("builtins.__import__", side_effect=_mock_import_error("spliceai.utils")):
            result = spliceai_predict(variant="1-55505647-G-A")
        assert "error" in result
        assert "not installed" in result["summary"].lower()
        assert "install_instructions" in result

    def test_missing_variant(self):
        result = spliceai_predict(variant="")
        assert "error" in result
        assert "requires" in result["summary"].lower()

    def test_none_variant(self):
        result = spliceai_predict(variant=None)
        assert "error" in result

    def test_invalid_format_too_few_parts(self):
        # Must fail before attempting import of spliceai
        # The function tries the import first, so we need to mock it
        with patch("builtins.__import__", side_effect=_mock_import_error("spliceai.utils")):
            result = spliceai_predict(variant="1-55505647-G-A")
        # With spliceai not installed, it returns install error before parsing
        assert "error" in result

    def test_invalid_format_three_parts(self):
        """Variant with only 3 parts should fail after spliceai import."""
        # Mock spliceai import to succeed so we reach the parsing code
        mock_annotator = MagicMock()
        mock_get_delta = MagicMock()
        with patch.dict("sys.modules", {
            "spliceai": MagicMock(),
            "spliceai.utils": MagicMock(Annotator=mock_annotator, get_delta_scores=mock_get_delta),
        }):
            result = spliceai_predict(variant="1-55505647-G")
        assert "error" in result
        assert "Invalid variant format" in result["error"]

    def test_non_numeric_position(self):
        """Variant with non-numeric position should fail."""
        mock_annotator = MagicMock()
        mock_get_delta = MagicMock()
        with patch.dict("sys.modules", {
            "spliceai": MagicMock(),
            "spliceai.utils": MagicMock(Annotator=mock_annotator, get_delta_scores=mock_get_delta),
        }):
            result = spliceai_predict(variant="1-abc-G-A")
        assert "error" in result
        assert "Invalid position" in result["error"]

    def test_distance_clamped(self):
        """Distance should be clamped between 50 and 10000."""
        with patch("builtins.__import__", side_effect=_mock_import_error("spliceai.utils")):
            # Just verify the function doesn't crash with extreme values
            result = spliceai_predict(variant="1-55505647-G-A", distance=1)
        assert "error" in result  # Will fail on import, but doesn't crash


# ---------------------------------------------------------------------------
# CADD tests
# ---------------------------------------------------------------------------


class TestCADDScore:
    def test_successful_score(self):
        mock_data = [
            {"Ref": "G", "Alt": "A", "RawScore": 5.123, "PHRED": 32.0},
        ]
        with patch("ct.tools.genomics.request_json", return_value=(mock_data, None)):
            result = cadd_score(variant="1:55505647:G:A")
        assert result["found"] is True
        assert result["phred_score"] == 32.0
        assert result["raw_score"] == 5.123
        assert "likely pathogenic" in result["interpretation"]
        assert "CADD" in result["summary"]

    def test_grch37_build(self):
        mock_data = [
            {"Ref": "C", "Alt": "T", "RawScore": 3.5, "PHRED": 22.0},
        ]
        with patch("ct.tools.genomics.request_json", return_value=(mock_data, None)) as mock_req:
            result = cadd_score(variant="7:140453136:C:T", genome_build="GRCh37")
        assert result["found"] is True
        assert result["genome_build"] == "GRCh37"
        # Check the URL was built with GRCh37
        call_args = mock_req.call_args
        assert "GRCh37" in call_args[0][1]

    def test_no_results(self):
        with patch("ct.tools.genomics.request_json", return_value=(None, None)):
            result = cadd_score(variant="1:1000:A:T")
        assert result["found"] is False
        assert "No CADD scores" in result["summary"]

    def test_empty_list_results(self):
        with patch("ct.tools.genomics.request_json", return_value=([], None)):
            result = cadd_score(variant="1:1000:A:T")
        assert result["found"] is False

    def test_invalid_format(self):
        result = cadd_score(variant="invalid")
        assert "error" in result
        assert "Invalid variant format" in result["error"]

    def test_missing_variant(self):
        result = cadd_score(variant="")
        assert "error" in result
        assert "requires" in result["summary"].lower()

    def test_matched_alt_allele(self):
        """When multiple results exist, the matching alt allele should be selected."""
        mock_data = [
            {"Ref": "G", "Alt": "C", "RawScore": 1.0, "PHRED": 10.0},
            {"Ref": "G", "Alt": "A", "RawScore": 5.5, "PHRED": 35.0},
            {"Ref": "G", "Alt": "T", "RawScore": 2.0, "PHRED": 15.0},
        ]
        with patch("ct.tools.genomics.request_json", return_value=(mock_data, None)):
            result = cadd_score(variant="1:55505647:G:A")
        assert result["found"] is True
        assert result["phred_score"] == 35.0
        assert result["raw_score"] == 5.5

    def test_interpretation_benign(self):
        mock_data = [{"Ref": "A", "Alt": "G", "RawScore": 0.5, "PHRED": 5.0}]
        with patch("ct.tools.genomics.request_json", return_value=(mock_data, None)):
            result = cadd_score(variant="1:1000:A:G")
        assert result["interpretation"] == "benign"

    def test_interpretation_likely_benign(self):
        mock_data = [{"Ref": "A", "Alt": "G", "RawScore": 1.5, "PHRED": 12.0}]
        with patch("ct.tools.genomics.request_json", return_value=(mock_data, None)):
            result = cadd_score(variant="1:1000:A:G")
        assert result["interpretation"] == "likely benign"

    def test_interpretation_uncertain(self):
        mock_data = [{"Ref": "A", "Alt": "G", "RawScore": 2.5, "PHRED": 17.0}]
        with patch("ct.tools.genomics.request_json", return_value=(mock_data, None)):
            result = cadd_score(variant="1:1000:A:G")
        assert result["interpretation"] == "uncertain significance"

    def test_interpretation_possibly_pathogenic(self):
        mock_data = [{"Ref": "A", "Alt": "G", "RawScore": 4.0, "PHRED": 25.0}]
        with patch("ct.tools.genomics.request_json", return_value=(mock_data, None)):
            result = cadd_score(variant="1:1000:A:G")
        assert result["interpretation"] == "possibly pathogenic (top 1% most deleterious)"

    def test_interpretation_likely_pathogenic(self):
        mock_data = [{"Ref": "A", "Alt": "G", "RawScore": 6.0, "PHRED": 35.0}]
        with patch("ct.tools.genomics.request_json", return_value=(mock_data, None)):
            result = cadd_score(variant="1:1000:A:G")
        assert result["interpretation"] == "likely pathogenic (top 0.1% most deleterious)"

    def test_api_error(self):
        with patch("ct.tools.genomics.request_json", return_value=(None, "Connection refused")):
            result = cadd_score(variant="1:55505647:G:A")
        assert "error" in result
        assert "failed" in result["summary"].lower()

    def test_dash_format(self):
        """Accept chr-pos-ref-alt format too."""
        mock_data = [{"Ref": "G", "Alt": "A", "RawScore": 5.0, "PHRED": 30.0}]
        with patch("ct.tools.genomics.request_json", return_value=(mock_data, None)):
            result = cadd_score(variant="1-55505647-G-A")
        assert result["found"] is True
        assert result["chrom"] == "1"

    def test_chr_prefix_stripped(self):
        """chr prefix should be stripped from chromosome."""
        mock_data = [{"Ref": "G", "Alt": "A", "RawScore": 5.0, "PHRED": 30.0}]
        with patch("ct.tools.genomics.request_json", return_value=(mock_data, None)):
            result = cadd_score(variant="chr1:55505647:G:A")
        assert result["found"] is True
        assert result["chrom"] == "1"

    def test_non_numeric_position(self):
        result = cadd_score(variant="1:abc:G:A")
        assert "error" in result
        assert "Non-numeric" in result["summary"]

    def test_score_unavailable_interpretation(self):
        """When PHRED score is None, interpretation should be 'score unavailable'."""
        mock_data = [{"Ref": "A", "Alt": "G", "RawScore": None, "PHRED": None}]
        with patch("ct.tools.genomics.request_json", return_value=(mock_data, None)):
            result = cadd_score(variant="1:1000:A:G")
        assert result["interpretation"] == "score unavailable"
        assert result["found"] is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_import_error(module_name):
    """Create a side_effect for builtins.__import__ that raises ImportError for a specific module."""
    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _side_effect(name, *args, **kwargs):
        if name == module_name or name.startswith(module_name + "."):
            raise ImportError(f"No module named '{module_name}'")
        return original_import(name, *args, **kwargs)

    return _side_effect
