"""Tests for metagenomics tools: who_arg_classify."""

import pytest


# ─── metagenomics.who_arg_classify ─────────────────────────────


class TestWhoArgClassify:
    """Tests for metagenomics.who_arg_classify."""

    def test_empty_genes(self):
        from ct.tools.metagenomics import who_arg_classify
        result = who_arg_classify(genes="")
        assert "error" in result
        assert "summary" in result

    def test_single_critical_gene(self):
        from ct.tools.metagenomics import who_arg_classify
        result = who_arg_classify(genes="blaNDM-1")
        assert result["counts"]["critical"] == 1
        assert result["classified_genes"][0]["tier"] == "critical"
        assert result["classified_genes"][0]["pathogen"] == "Enterobacterales"

    def test_mixed_tiers(self):
        from ct.tools.metagenomics import who_arg_classify
        result = who_arg_classify(genes="mecA,blaNDM-1,tetM")
        assert result["total_genes"] == 3
        assert result["counts"]["critical"] == 1  # blaNDM-1
        assert result["counts"]["high"] == 1      # mecA
        assert result["counts"]["medium"] == 1    # tetM

    def test_unclassified_gene(self):
        from ct.tools.metagenomics import who_arg_classify
        result = who_arg_classify(genes="unknownGene123")
        assert result["counts"]["unclassified"] == 1
        assert result["classified_genes"][0]["tier"] == "unclassified"

    def test_exclude_unclassified(self):
        from ct.tools.metagenomics import who_arg_classify
        result = who_arg_classify(genes="unknownGene123", include_unclassified=False)
        assert len(result["classified_genes"]) == 0
        assert result["counts"]["unclassified"] == 1  # Still counted

    def test_whitespace_handling(self):
        from ct.tools.metagenomics import who_arg_classify
        result = who_arg_classify(genes=" mecA , blaNDM-1 , tetM ")
        assert result["total_genes"] == 3

    def test_summary_format(self):
        from ct.tools.metagenomics import who_arg_classify
        result = who_arg_classify(genes="blaKPC-2,vanA,sul1")
        assert "Classified 3 ARGs" in result["summary"]
        assert "critical" in result["summary"]

    def test_drug_class_info(self):
        from ct.tools.metagenomics import who_arg_classify
        result = who_arg_classify(genes="mecA")
        gene_info = result["classified_genes"][0]
        assert gene_info["drug_class"] == "methicillin/oxacillin"
        assert gene_info["mechanism"] == "target alteration"

    def test_by_tier_structure(self):
        from ct.tools.metagenomics import who_arg_classify
        result = who_arg_classify(genes="blaOXA-23,vanA,tetB,unknownXYZ")
        assert "blaOXA-23" in result["by_tier"]["critical"]
        assert "vanA" in result["by_tier"]["high"]
        assert "tetB" in result["by_tier"]["medium"]
        assert "unknownXYZ" in result["by_tier"]["unclassified"]
