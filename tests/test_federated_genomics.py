"""Tests for new genomics tools: federated variant lookup, PheWAS, eQTL Catalogue, PGS Catalog."""

import pytest
from unittest.mock import patch, MagicMock


# ─── Mock data ─────────────────────────────────────────────────

MOCK_ENSEMBL_VARIATION = {
    "mappings": [
        {
            "assembly_name": "GRCh38",
            "seq_region_name": "19",
            "start": 44908684,
            "allele_string": "C/T",
        },
        {
            "assembly_name": "GRCh37",
            "seq_region_name": "19",
            "start": 45412079,
            "allele_string": "C/T",
        },
    ],
}

MOCK_FINNGEN_RESPONSE = {
    "results": [
        {"phenocode": "I9_CHD", "phenostring": "Coronary heart disease", "pval": 1.2e-10, "beta": 0.15, "sebeta": 0.02, "maf": 0.08, "maf_cases": 0.09, "maf_controls": 0.07},
        {"phenocode": "E4_HYPERLIPNAS", "phenostring": "Hyperlipidaemia", "pval": 3.5e-8, "beta": 0.12, "sebeta": 0.02, "maf": 0.08, "maf_cases": 0.10, "maf_controls": 0.07},
        {"phenocode": "G6_HEADACHE", "phenostring": "Headache", "pval": 0.5, "beta": 0.01, "sebeta": 0.02, "maf": 0.08, "maf_cases": 0.08, "maf_controls": 0.08},
    ],
}

MOCK_UKB_RESPONSE = {
    "phenos": [
        {"phenocode": "I25", "phenostring": "Chronic ischaemic heart disease", "pval": 5.0e-12, "beta": 0.18, "sebeta": 0.02, "maf": 0.08, "num_cases": 25000, "num_controls": 450000},
        {"phenocode": "E78", "phenostring": "Disorders of lipoprotein metabolism", "pval": 1.1e-9, "beta": 0.14, "sebeta": 0.02, "maf": 0.08, "num_cases": 80000, "num_controls": 400000},
        {"phenocode": "R51", "phenostring": "Headache", "pval": 0.8, "beta": 0.0, "sebeta": 0.02, "maf": 0.08, "num_cases": 5000, "num_controls": 470000},
    ],
}

MOCK_BBJ_RESPONSE = {
    "phenos": [
        {"phenocode": "CAD", "phenostring": "Coronary artery disease", "pval": 2.0e-6, "beta": 0.10, "sebeta": 0.02, "maf": 0.12, "num_cases": 15000, "num_controls": 165000},
    ],
}

MOCK_EQTL_RESPONSE = [
    {"gene_id": "ENSG00000130203", "molecular_trait_id": "APOE", "pvalue": 1e-20, "beta": 0.8, "se": 0.05, "neg_log10_pvalue": 20.0, "rsid": "rs7412", "variant": "19_44908684_C_T", "tissue": "whole_blood"},
    {"gene_id": "ENSG00000175535", "molecular_trait_id": "PVRL2", "pvalue": 5e-4, "beta": -0.1, "se": 0.03, "neg_log10_pvalue": 3.3, "rsid": "rs7412", "variant": "19_44908684_C_T", "tissue": "whole_blood"},
]

MOCK_PGS_TRAIT_RESPONSE = {
    "results": [
        {"id": "EFO_0001360", "label": "type 2 diabetes mellitus", "description": "A type of diabetes mellitus.", "associated_pgs_ids": ["PGS000014", "PGS000036"]},
    ],
}

MOCK_PGS_SCORE_SEARCH = {
    "results": [
        {"id": "PGS000014", "name": "T2D PRS", "trait_reported": "Type 2 diabetes", "variants_number": 1289, "samples_variants": [{"sample_number": 898130}]},
        {"id": "PGS000036", "name": "Diabetes Risk Score", "trait_reported": "Type 2 diabetes", "variants_number": 65, "samples_variants": [{"sample_number": 120000}]},
    ],
}

MOCK_PGS_SCORE_INFO = {
    "id": "PGS000014",
    "name": "T2D PRS",
    "trait_reported": "Type 2 diabetes",
    "trait_efo": [{"label": "type 2 diabetes mellitus"}],
    "variants_number": 1289,
    "samples_variants": [{"sample_number": 898130}],
    "samples_training": [{"sample_number": 455000}],
    "method_name": "LDpred",
    "method_params": "rho=0.01",
    "date_release": "2019-07-10",
    "publication": {
        "title": "A PRS for T2D",
        "doi": "10.1234/test",
        "journal": "Nature Genetics",
        "firstauthor": "Smith A",
        "date_publication": "2019-06-01",
    },
}


# ─── _resolve_rsid ─────────────────────────────────────────────


class TestResolveRsid:
    """Tests for _resolve_rsid helper."""

    @patch("ct.tools.genomics.request_json")
    def test_successful_resolution(self, mock_rj):
        mock_rj.return_value = (MOCK_ENSEMBL_VARIATION, None)
        from ct.tools.genomics import _resolve_rsid
        result = _resolve_rsid("rs7412")
        assert result["chr"] == "19"
        assert result["pos_grch38"] == 44908684
        assert result["pos_grch37"] == 45412079
        assert result["ref"] == "C"
        assert result["alt"] == "T"

    def test_invalid_rsid_format(self):
        from ct.tools.genomics import _resolve_rsid
        result = _resolve_rsid("BRCA1")
        assert "error" in result

    def test_empty_rsid(self):
        from ct.tools.genomics import _resolve_rsid
        result = _resolve_rsid("")
        assert "error" in result

    @patch("ct.tools.genomics.request_json")
    def test_api_error(self, mock_rj):
        mock_rj.return_value = (None, "Connection timeout")
        from ct.tools.genomics import _resolve_rsid
        result = _resolve_rsid("rs7412")
        assert "error" in result

    @patch("ct.tools.genomics.request_json")
    def test_no_mappings(self, mock_rj):
        mock_rj.return_value = ({"mappings": []}, None)
        from ct.tools.genomics import _resolve_rsid
        result = _resolve_rsid("rs9999999999")
        assert "error" in result


# ─── genomics.finngen_phewas ────────────────────────────────────


class TestFinngenPhewas:
    """Tests for genomics.finngen_phewas."""

    def test_empty_rsid_returns_error(self):
        from ct.tools.genomics import finngen_phewas
        result = finngen_phewas(rsid="")
        assert "error" in result
        assert "summary" in result

    @patch("ct.tools.genomics.request_json")
    def test_successful_phewas(self, mock_rj):
        mock_rj.side_effect = [
            (MOCK_ENSEMBL_VARIATION, None),
            (MOCK_FINNGEN_RESPONSE, None),
        ]
        from ct.tools.genomics import finngen_phewas
        result = finngen_phewas(rsid="rs7412")
        assert "summary" in result
        assert result["n_significant"] == 2  # Only p < 0.05
        assert result["associations"][0]["pval"] < result["associations"][1]["pval"]

    @patch("ct.tools.genomics.request_json")
    def test_resolve_error(self, mock_rj):
        mock_rj.return_value = (None, "timeout")
        from ct.tools.genomics import finngen_phewas
        result = finngen_phewas(rsid="rs7412")
        assert "error" in result

    @patch("ct.tools.genomics.request_json")
    def test_api_error(self, mock_rj):
        mock_rj.side_effect = [
            (MOCK_ENSEMBL_VARIATION, None),
            (None, "FinnGen timeout"),
        ]
        from ct.tools.genomics import finngen_phewas
        result = finngen_phewas(rsid="rs7412")
        assert "error" in result

    @patch("ct.tools.genomics.request_json")
    def test_no_results(self, mock_rj):
        mock_rj.side_effect = [
            (MOCK_ENSEMBL_VARIATION, None),
            ({"results": []}, None),
        ]
        from ct.tools.genomics import finngen_phewas
        result = finngen_phewas(rsid="rs7412")
        assert result["n_significant"] == 0

    @patch("ct.tools.genomics.request_json")
    def test_max_results_cap(self, mock_rj):
        many_results = [{"phenocode": f"P{i}", "phenostring": f"Pheno {i}", "pval": 0.001 * (i+1), "beta": 0.1, "sebeta": 0.01, "maf": 0.05, "maf_cases": 0.06, "maf_controls": 0.04} for i in range(100)]
        mock_rj.side_effect = [
            (MOCK_ENSEMBL_VARIATION, None),
            ({"results": many_results}, None),
        ]
        from ct.tools.genomics import finngen_phewas
        result = finngen_phewas(rsid="rs7412", max_results=5)
        assert len(result["associations"]) == 5


# ─── genomics.ukb_phewas ───────────────────────────────────────


class TestUkbPhewas:
    """Tests for genomics.ukb_phewas."""

    def test_empty_rsid(self):
        from ct.tools.genomics import ukb_phewas
        result = ukb_phewas(rsid="")
        assert "error" in result

    @patch("ct.tools.genomics.request_json")
    def test_successful_phewas(self, mock_rj):
        mock_rj.side_effect = [
            (MOCK_ENSEMBL_VARIATION, None),
            (MOCK_UKB_RESPONSE, None),
        ]
        from ct.tools.genomics import ukb_phewas
        result = ukb_phewas(rsid="rs7412")
        assert result["n_significant"] == 2
        assert "UKB" in result["summary"]

    @patch("ct.tools.genomics.request_json")
    def test_api_error(self, mock_rj):
        mock_rj.side_effect = [
            (MOCK_ENSEMBL_VARIATION, None),
            (None, "UKB API error"),
        ]
        from ct.tools.genomics import ukb_phewas
        result = ukb_phewas(rsid="rs7412")
        assert "error" in result

    @patch("ct.tools.genomics.request_json")
    def test_empty_phenos(self, mock_rj):
        mock_rj.side_effect = [
            (MOCK_ENSEMBL_VARIATION, None),
            ({"phenos": []}, None),
        ]
        from ct.tools.genomics import ukb_phewas
        result = ukb_phewas(rsid="rs7412")
        assert result["n_significant"] == 0


# ─── genomics.bbj_phewas ───────────────────────────────────────


class TestBbjPhewas:
    """Tests for genomics.bbj_phewas."""

    def test_empty_rsid(self):
        from ct.tools.genomics import bbj_phewas
        result = bbj_phewas(rsid="")
        assert "error" in result

    @patch("ct.tools.genomics.request_json")
    def test_successful_phewas(self, mock_rj):
        mock_rj.side_effect = [
            (MOCK_ENSEMBL_VARIATION, None),
            (MOCK_BBJ_RESPONSE, None),
        ]
        from ct.tools.genomics import bbj_phewas
        result = bbj_phewas(rsid="rs7412")
        assert result["n_significant"] == 1
        assert result["genome_build"] == "GRCh37"
        # BBJ uses GRCh37 coordinates
        assert "45412079" in result["variant"]

    @patch("ct.tools.genomics.request_json")
    def test_no_grch37_coordinates(self, mock_rj):
        no_grch37 = {
            "mappings": [
                {"assembly_name": "GRCh38", "seq_region_name": "19", "start": 44908684, "allele_string": "C/T"},
            ],
        }
        mock_rj.return_value = (no_grch37, None)
        from ct.tools.genomics import bbj_phewas
        result = bbj_phewas(rsid="rs7412")
        assert "error" in result
        assert "GRCh37" in result["error"]

    @patch("ct.tools.genomics.request_json")
    def test_api_error(self, mock_rj):
        mock_rj.side_effect = [
            (MOCK_ENSEMBL_VARIATION, None),
            (None, "BBJ timeout"),
        ]
        from ct.tools.genomics import bbj_phewas
        result = bbj_phewas(rsid="rs7412")
        assert "error" in result


# ─── genomics.eqtl_catalogue_lookup ────────────────────────────


class TestEqtlCatalogueLookup:
    """Tests for genomics.eqtl_catalogue_lookup."""

    def test_empty_rsid(self):
        from ct.tools.genomics import eqtl_catalogue_lookup
        result = eqtl_catalogue_lookup(rsid="")
        assert "error" in result

    @patch("ct.tools.genomics.request_json")
    def test_successful_lookup(self, mock_rj):
        mock_rj.side_effect = [
            (MOCK_ENSEMBL_VARIATION, None),
            (MOCK_EQTL_RESPONSE, None),
        ]
        from ct.tools.genomics import eqtl_catalogue_lookup
        result = eqtl_catalogue_lookup(rsid="rs7412")
        assert result["n_associations"] == 2
        assert result["associations"][0]["pvalue"] < result["associations"][1]["pvalue"]

    @patch("ct.tools.genomics.request_json")
    def test_gene_filter(self, mock_rj):
        mock_rj.side_effect = [
            (MOCK_ENSEMBL_VARIATION, None),
            (MOCK_EQTL_RESPONSE, None),
        ]
        from ct.tools.genomics import eqtl_catalogue_lookup
        result = eqtl_catalogue_lookup(rsid="rs7412", gene="APOE")
        assert result["n_associations"] == 1
        assert result["associations"][0]["molecular_trait_id"] == "APOE"

    @patch("ct.tools.genomics.request_json")
    def test_api_error(self, mock_rj):
        mock_rj.side_effect = [
            (MOCK_ENSEMBL_VARIATION, None),
            (None, "eQTL API timeout"),
        ]
        from ct.tools.genomics import eqtl_catalogue_lookup
        result = eqtl_catalogue_lookup(rsid="rs7412")
        assert "error" in result

    @patch("ct.tools.genomics.request_json")
    def test_empty_results(self, mock_rj):
        mock_rj.side_effect = [
            (MOCK_ENSEMBL_VARIATION, None),
            ([], None),
        ]
        from ct.tools.genomics import eqtl_catalogue_lookup
        result = eqtl_catalogue_lookup(rsid="rs7412")
        assert result["n_associations"] == 0


# ─── genomics.pgs_trait_search ──────────────────────────────────


class TestPgsTraitSearch:
    """Tests for genomics.pgs_trait_search."""

    def test_empty_trait(self):
        from ct.tools.genomics import pgs_trait_search
        result = pgs_trait_search(trait="")
        assert "error" in result

    @patch("ct.tools.genomics.request_json")
    def test_successful_search(self, mock_rj):
        mock_rj.side_effect = [
            (MOCK_PGS_TRAIT_RESPONSE, None),
            (MOCK_PGS_SCORE_SEARCH, None),
        ]
        from ct.tools.genomics import pgs_trait_search
        result = pgs_trait_search(trait="type 2 diabetes")
        assert result["n_scores"] == 2
        assert "EFO_0001360" in result["matched_trait"]["id"]

    @patch("ct.tools.genomics.request_json")
    def test_no_traits_found(self, mock_rj):
        mock_rj.return_value = ({"results": []}, None)
        from ct.tools.genomics import pgs_trait_search
        result = pgs_trait_search(trait="nonexistent disease xyz")
        assert result["scores"] == []

    @patch("ct.tools.genomics.request_json")
    def test_trait_api_error(self, mock_rj):
        mock_rj.return_value = (None, "PGS API down")
        from ct.tools.genomics import pgs_trait_search
        result = pgs_trait_search(trait="diabetes")
        assert "error" in result

    @patch("ct.tools.genomics.request_json")
    def test_score_api_error(self, mock_rj):
        mock_rj.side_effect = [
            (MOCK_PGS_TRAIT_RESPONSE, None),
            (None, "Score API down"),
        ]
        from ct.tools.genomics import pgs_trait_search
        result = pgs_trait_search(trait="diabetes")
        assert "error" in result


# ─── genomics.pgs_score_info ────────────────────────────────────


class TestPgsScoreInfo:
    """Tests for genomics.pgs_score_info."""

    def test_empty_pgs_id(self):
        from ct.tools.genomics import pgs_score_info
        result = pgs_score_info(pgs_id="")
        assert "error" in result

    @patch("ct.tools.genomics.request_json")
    def test_successful_lookup(self, mock_rj):
        mock_rj.return_value = (MOCK_PGS_SCORE_INFO, None)
        from ct.tools.genomics import pgs_score_info
        result = pgs_score_info(pgs_id="PGS000014")
        assert result["id"] == "PGS000014"
        assert result["variants_number"] == 1289
        assert "Smith A" in result["summary"]

    @patch("ct.tools.genomics.request_json")
    def test_not_found(self, mock_rj):
        mock_rj.return_value = (None, "HTTP 404")
        from ct.tools.genomics import pgs_score_info
        result = pgs_score_info(pgs_id="PGS999999")
        assert "error" in result

    @patch("ct.tools.genomics.request_json")
    def test_empty_response(self, mock_rj):
        mock_rj.return_value = ({}, None)
        from ct.tools.genomics import pgs_score_info
        result = pgs_score_info(pgs_id="PGS000001")
        # Empty dict but valid — should still return
        assert "summary" in result


# ─── genomics.variant_federated_lookup ──────────────────────────


class TestVariantFederatedLookup:
    """Tests for genomics.variant_federated_lookup."""

    def test_empty_rsid(self):
        from ct.tools.genomics import variant_federated_lookup
        result = variant_federated_lookup(rsid="")
        assert "error" in result

    @patch("ct.tools.genomics.request_json")
    def test_successful_federated_lookup(self, mock_rj):
        # _resolve_rsid in federated + 4 individual tools each call _resolve_rsid + their API
        # Total: 1 (federated resolve) + 4 * 2 (each phewas: resolve + api) = 9 calls
        # But actually, the individual tools call _resolve_rsid themselves, so:
        # federated calls _resolve_rsid (1) then dispatches 4 tools, each of which calls _resolve_rsid again
        mock_rj.side_effect = [
            # federated _resolve_rsid
            (MOCK_ENSEMBL_VARIATION, None),
            # finngen: _resolve_rsid + api
            (MOCK_ENSEMBL_VARIATION, None),
            (MOCK_FINNGEN_RESPONSE, None),
            # ukb: _resolve_rsid + api
            (MOCK_ENSEMBL_VARIATION, None),
            (MOCK_UKB_RESPONSE, None),
            # bbj: _resolve_rsid + api
            (MOCK_ENSEMBL_VARIATION, None),
            (MOCK_BBJ_RESPONSE, None),
            # eqtl: _resolve_rsid + api
            (MOCK_ENSEMBL_VARIATION, None),
            (MOCK_EQTL_RESPONSE, None),
        ]
        from ct.tools.genomics import variant_federated_lookup
        result = variant_federated_lookup(rsid="rs7412")
        assert "summary" in result
        assert "results" in result
        assert "finngen" in result["results"]
        assert "ukb_topmed" in result["results"]
        assert "biobank_japan" in result["results"]
        assert "eqtl_catalogue" in result["results"]

    @patch("ct.tools.genomics.request_json")
    def test_resolve_error(self, mock_rj):
        mock_rj.return_value = (None, "timeout")
        from ct.tools.genomics import variant_federated_lookup
        result = variant_federated_lookup(rsid="rs7412")
        assert "error" in result

    @patch("ct.tools.genomics.request_json")
    def test_partial_failures(self, mock_rj):
        # federated resolve succeeds, then some sub-tools fail
        mock_rj.side_effect = [
            # federated _resolve_rsid
            (MOCK_ENSEMBL_VARIATION, None),
            # finngen: resolve + api success
            (MOCK_ENSEMBL_VARIATION, None),
            (MOCK_FINNGEN_RESPONSE, None),
            # ukb: resolve + api fails
            (MOCK_ENSEMBL_VARIATION, None),
            (None, "UKB timeout"),
            # bbj: resolve + api success
            (MOCK_ENSEMBL_VARIATION, None),
            (MOCK_BBJ_RESPONSE, None),
            # eqtl: resolve + api fails
            (MOCK_ENSEMBL_VARIATION, None),
            (None, "eQTL timeout"),
        ]
        from ct.tools.genomics import variant_federated_lookup
        result = variant_federated_lookup(rsid="rs7412")
        assert "summary" in result
        # Should still have results from working databases
        assert "results" in result
        # FinnGen and BBJ should have succeeded
        finngen = result["results"]["finngen"]
        assert "n_significant" in finngen
