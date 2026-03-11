"""Tests for Galaxy tool discovery tools: tool_search, tool_details."""

import pytest
from unittest.mock import patch, MagicMock


MOCK_CATALOG = [
    {
        "id": "toolshed.g2.bx.psu.edu/repos/devteam/fastqc/fastqc/0.74+galaxy0",
        "name": "FastQC",
        "description": "Read Quality reports",
        "section": "FASTQ Quality Control",
        "edam_topics": ["Sequencing", "Data quality management"],
        "edam_operations": ["Sequencing quality control"],
        "inputs": [{"name": "input_file", "type": "fastqsanger"}],
        "outputs": [{"name": "html_file", "type": "html"}, {"name": "text_file", "type": "txt"}],
        "version": "0.74+galaxy0",
        "tool_shed_url": "https://toolshed.g2.bx.psu.edu",
    },
    {
        "id": "toolshed.g2.bx.psu.edu/repos/iuc/star/star/2.7.10b+galaxy4",
        "name": "STAR",
        "description": "Spliced Transcripts Alignment to a Reference",
        "section": "Mapping",
        "edam_topics": ["RNA-seq", "Transcriptomics"],
        "edam_operations": ["Sequence alignment", "Splice site prediction"],
        "inputs": [{"name": "input_fastq", "type": "fastqsanger"}],
        "outputs": [{"name": "aligned_reads", "type": "bam"}],
        "version": "2.7.10b+galaxy4",
        "tool_shed_url": "https://toolshed.g2.bx.psu.edu",
    },
    {
        "id": "toolshed.g2.bx.psu.edu/repos/iuc/freebayes/freebayes/1.3.6+galaxy0",
        "name": "FreeBayes",
        "description": "Bayesian haplotype-based polymorphism discovery and genotyping",
        "section": "Variant Calling",
        "edam_topics": ["Variant calling", "Genomics"],
        "edam_operations": ["Variant calling"],
        "inputs": [{"name": "bam_input", "type": "bam"}],
        "outputs": [{"name": "output_vcf", "type": "vcf"}],
        "version": "1.3.6+galaxy0",
        "tool_shed_url": "https://toolshed.g2.bx.psu.edu",
    },
]


# ─── galaxy.tool_search ────────────────────────────────────────


class TestGalaxyToolSearch:
    """Tests for galaxy.tool_search."""

    def test_empty_query(self):
        from ct.tools.galaxy import tool_search
        result = tool_search(query="")
        assert "error" in result

    @patch("ct.tools.galaxy._load_catalog", return_value=MOCK_CATALOG)
    def test_search_by_name(self, mock_cat):
        from ct.tools.galaxy import tool_search
        result = tool_search(query="FastQC")
        assert result["n_results"] >= 1
        assert result["results"][0]["name"] == "FastQC"

    @patch("ct.tools.galaxy._load_catalog", return_value=MOCK_CATALOG)
    def test_search_by_topic(self, mock_cat):
        from ct.tools.galaxy import tool_search
        result = tool_search(query="variant calling")
        assert result["n_results"] >= 1
        names = [r["name"] for r in result["results"]]
        assert "FreeBayes" in names

    @patch("ct.tools.galaxy._load_catalog", return_value=MOCK_CATALOG)
    def test_no_matches(self, mock_cat):
        from ct.tools.galaxy import tool_search
        result = tool_search(query="quantum computing")
        assert result["n_results"] == 0

    @patch("ct.tools.galaxy._load_catalog", return_value=MOCK_CATALOG)
    def test_max_results(self, mock_cat):
        from ct.tools.galaxy import tool_search
        result = tool_search(query="quality sequencing", max_results=1)
        assert len(result["results"]) <= 1

    @patch("ct.tools.galaxy._load_catalog", return_value=MOCK_CATALOG)
    def test_result_has_relevance_score(self, mock_cat):
        from ct.tools.galaxy import tool_search
        result = tool_search(query="STAR alignment")
        if result["n_results"] > 0:
            assert "relevance_score" in result["results"][0]


# ─── galaxy.tool_details ───────────────────────────────────────


class TestGalaxyToolDetails:
    """Tests for galaxy.tool_details."""

    def test_no_identifier(self):
        from ct.tools.galaxy import tool_details
        result = tool_details()
        assert "error" in result

    @patch("ct.tools.galaxy._load_catalog", return_value=MOCK_CATALOG)
    def test_lookup_by_id(self, mock_cat):
        from ct.tools.galaxy import tool_details
        result = tool_details(tool_id="toolshed.g2.bx.psu.edu/repos/devteam/fastqc/fastqc/0.74+galaxy0")
        assert result["found"] is True
        assert result["name"] == "FastQC"

    @patch("ct.tools.galaxy._load_catalog", return_value=MOCK_CATALOG)
    def test_lookup_by_name(self, mock_cat):
        from ct.tools.galaxy import tool_details
        result = tool_details(tool_name="STAR")
        assert result["found"] is True
        assert result["name"] == "STAR"

    @patch("ct.tools.galaxy._load_catalog", return_value=MOCK_CATALOG)
    def test_not_found(self, mock_cat):
        from ct.tools.galaxy import tool_details
        result = tool_details(tool_name="NonExistentTool123")
        assert result["found"] is False

    @patch("ct.tools.galaxy.request_json")
    def test_live_api_success(self, mock_rj):
        mock_rj.return_value = ({
            "name": "FastQC",
            "description": "Read Quality reports",
            "version": "0.74",
            "edam_topics": ["Sequencing"],
            "edam_operations": ["QC"],
            "inputs": [{"name": "input_file", "label": "Input FASTQ", "type": "data", "optional": False}],
            "outputs": [{"name": "html_file", "format": "html", "label": "HTML report"}],
        }, None)
        from ct.tools.galaxy import tool_details
        result = tool_details(tool_id="toolshed.g2.bx.psu.edu/repos/devteam/fastqc/fastqc/0.74+galaxy0", live=True)
        assert result["found"] is True
        assert result["source"] == "usegalaxy.org API"

    @patch("ct.tools.galaxy.request_json")
    def test_live_api_error(self, mock_rj):
        mock_rj.return_value = (None, "Connection refused")
        from ct.tools.galaxy import tool_details
        result = tool_details(tool_id="some/tool/id", live=True)
        assert "error" in result

    @patch("ct.tools.galaxy._load_catalog", return_value=MOCK_CATALOG)
    def test_partial_name_match(self, mock_cat):
        from ct.tools.galaxy import tool_details
        result = tool_details(tool_name="Free")
        assert result["found"] is True
        assert result["name"] == "FreeBayes"
