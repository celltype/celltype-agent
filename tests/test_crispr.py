"""Tests for CRISPR screen analysis tools: screen_qc, mageck_analyze."""

import os
import pytest
import tempfile
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


# ─── crispr.screen_qc ────────────────────────────────────────


class TestScreenQc:
    """Tests for crispr.screen_qc."""

    def test_empty_path_returns_error(self):
        from ct.tools.crispr import screen_qc
        result = screen_qc(count_file="")
        assert "error" in result

    def test_missing_file_returns_error(self):
        from ct.tools.crispr import screen_qc
        result = screen_qc(count_file="/nonexistent/file.tsv")
        assert "error" in result

    def test_basic_qc_tsv(self):
        """QC on a well-formatted TSV count matrix."""
        n_sgrnas = 200
        n_samples = 3

        df = pd.DataFrame({
            "gene": [f"Gene_{i // 4}" for i in range(n_sgrnas)],
            "sample1": np.random.poisson(500, n_sgrnas),
            "sample2": np.random.poisson(500, n_sgrnas),
            "sample3": np.random.poisson(500, n_sgrnas),
        }, index=[f"sgRNA_{i}" for i in range(n_sgrnas)])

        with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
            df.to_csv(f, sep="\t")
            tsv_path = f.name

        try:
            from ct.tools.crispr import screen_qc
            result = screen_qc(count_file=tsv_path)

            assert "summary" in result
            assert result["n_sgrnas"] == n_sgrnas
            assert result["n_samples"] == n_samples
            assert "sample_metrics" in result
            assert len(result["sample_metrics"]) == n_samples

            for sample, metrics in result["sample_metrics"].items():
                assert "gini_index" in metrics
                assert 0 <= metrics["gini_index"] <= 1
                assert "zero_count_fraction" in metrics
                assert "total_reads" in metrics

            assert "verdict" in result
            assert result["verdict"] in ("PASS", "ACCEPTABLE", "FAIL")
        finally:
            os.unlink(tsv_path)

    def test_csv_format(self):
        """QC works with CSV format."""
        n_sgrnas = 50
        df = pd.DataFrame({
            "gene": [f"Gene_{i // 2}" for i in range(n_sgrnas)],
            "ctrl": np.random.poisson(300, n_sgrnas),
            "treat": np.random.poisson(300, n_sgrnas),
        }, index=[f"sg_{i}" for i in range(n_sgrnas)])

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            df.to_csv(f)
            csv_path = f.name

        try:
            from ct.tools.crispr import screen_qc
            result = screen_qc(count_file=csv_path)
            assert result["n_sgrnas"] == n_sgrnas
            assert "sample_correlations" in result
        finally:
            os.unlink(csv_path)

    def test_high_zero_fraction_flagged(self):
        """Screens with many zero counts get flagged."""
        n_sgrnas = 100
        counts = np.zeros(n_sgrnas, dtype=int)
        counts[:20] = np.random.poisson(500, 20)  # only 20% have reads

        df = pd.DataFrame({
            "gene": [f"Gene_{i}" for i in range(n_sgrnas)],
            "sample1": counts,
        }, index=[f"sg_{i}" for i in range(n_sgrnas)])

        with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
            df.to_csv(f, sep="\t")
            tsv_path = f.name

        try:
            from ct.tools.crispr import screen_qc
            result = screen_qc(count_file=tsv_path)
            assert result["sample_metrics"]["sample1"]["zero_count_fraction"] > 0.5
            assert len(result["qc_flags"]) > 0
        finally:
            os.unlink(tsv_path)

    def test_library_mapping_rate(self):
        """Mapping rate computed when library file provided."""
        n_sgrnas = 50
        df = pd.DataFrame({
            "gene": [f"Gene_{i}" for i in range(n_sgrnas)],
            "sample1": np.random.poisson(500, n_sgrnas),
        }, index=[f"sg_{i}" for i in range(n_sgrnas)])

        lib_df = pd.DataFrame({
            "sgRNA": [f"sg_{i}" for i in range(100)],  # library has 100 sgRNAs, data has 50
            "gene": [f"Gene_{i}" for i in range(100)],
        })

        with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
            df.to_csv(f, sep="\t")
            tsv_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
            lib_df.to_csv(f, sep="\t", index=False)
            lib_path = f.name

        try:
            from ct.tools.crispr import screen_qc
            result = screen_qc(count_file=tsv_path, library_file=lib_path)
            assert result["mapping_rate"] is not None
            assert result["mapping_rate"]["mapping_rate"] == 0.5  # 50/100
        finally:
            os.unlink(tsv_path)
            os.unlink(lib_path)


# ─── crispr.mageck_analyze ────────────────────────────────────


class TestMageckAnalyze:
    """Tests for crispr.mageck_analyze."""

    def test_empty_params_return_error(self):
        from ct.tools.crispr import mageck_analyze
        result = mageck_analyze(count_file="")
        assert "error" in result

    def test_missing_treatment_control(self):
        from ct.tools.crispr import mageck_analyze
        result = mageck_analyze(count_file="counts.tsv", treatment="", control="")
        assert "error" in result
        assert "treatment" in result["error"].lower() or "control" in result["error"].lower()

    def test_mageck_not_installed(self):
        """When MAGeCK is not in PATH, return install instructions."""
        with patch("ct.tools.crispr.shutil.which", return_value=None):
            from ct.tools.crispr import mageck_analyze
            result = mageck_analyze(count_file="counts.tsv", treatment="treat", control="ctrl")

        assert "error" in result
        assert "install" in result["error"].lower() or "not installed" in result["error"].lower()
        assert "install_instructions" in result

    def test_missing_count_file(self):
        """Non-existent count file returns error."""
        with patch("ct.tools.crispr.shutil.which", return_value="/usr/bin/mageck"):
            from ct.tools.crispr import mageck_analyze
            result = mageck_analyze(count_file="/nonexistent/counts.tsv", treatment="treat", control="ctrl")

        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_mageck_success(self):
        """Successful MAGeCK run with mocked subprocess."""
        # Create a fake count file
        with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
            f.write("sgRNA\tgene\tctrl\ttreat\n")
            f.write("sg1\tGeneA\t100\t10\n")
            f.write("sg2\tGeneA\t200\t20\n")
            count_path = f.name

        # Create fake gene summary output
        gene_summary_content = "id\tneg|score\tneg|fdr\tneg|lfc\tpos|score\tpos|fdr\tpos|lfc\n"
        gene_summary_content += "GeneA\t0.001\t0.01\t-2.5\t0.5\t0.8\t0.5\n"
        gene_summary_content += "GeneB\t0.5\t0.9\t-0.1\t0.001\t0.02\t3.0\n"

        def mock_subprocess_run(cmd, **kw):
            # Write fake output file
            output_prefix = None
            for i, arg in enumerate(cmd):
                if arg == "-n" and i + 1 < len(cmd):
                    output_prefix = cmd[i + 1]
            if output_prefix:
                with open(f"{output_prefix}.gene_summary.txt", "w") as f:
                    f.write(gene_summary_content)
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            return result

        try:
            with patch("ct.tools.crispr.shutil.which", return_value="/usr/bin/mageck"), \
                 patch("subprocess.run", side_effect=mock_subprocess_run):
                from ct.tools.crispr import mageck_analyze
                result = mageck_analyze(count_file=count_path, treatment="treat", control="ctrl", fdr_threshold=0.05)

            assert "summary" in result
            assert result["n_depleted_hits"] == 1  # GeneA
            assert result["n_enriched_hits"] == 1  # GeneB
            assert result["depleted_hits"][0]["gene"] == "GeneA"
            assert result["enriched_hits"][0]["gene"] == "GeneB"
        finally:
            os.unlink(count_path)
