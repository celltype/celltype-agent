"""
CRISPR screen analysis tools: MAGeCK statistical analysis and screen QC.

Provides tools for analyzing pooled CRISPR knockout/activation screens,
including quality control metrics and hit identification.
"""

import shutil

from ct.tools import registry


@registry.register(
    name="crispr.screen_qc",
    description="Compute quality control metrics for a CRISPR screen: Gini index, zero-count fraction, mapping rate, sample correlations",
    category="crispr",
    parameters={
        "count_file": "Path to sgRNA count matrix (TSV/CSV: sgRNA rows × sample columns, first col = sgRNA ID, second col = gene)",
        "library_file": "Path to library file for mapping rate calculation (optional, TSV: sgRNA ID, gene)",
    },
    requires_data=[],
    usage_guide="You have raw CRISPR screen count data and need to assess quality before analysis. Checks read distribution evenness (Gini), dropout (zero-count sgRNAs), coverage, and sample-to-sample correlations. Run this before MAGeCK or any hit calling.",
)
def screen_qc(count_file: str, library_file: str = None, **kwargs) -> dict:
    """Compute CRISPR screen QC metrics from a count matrix."""
    import numpy as np
    import pandas as pd

    count_file = (count_file or "").strip()
    if not count_file:
        return {"error": "count_file path is required", "summary": "No count file provided for CRISPR QC"}

    # Load count matrix
    try:
        sep = "\t" if count_file.endswith(".tsv") else ","
        counts = pd.read_csv(count_file, sep=sep, index_col=0)
    except Exception as e:
        return {"error": f"Failed to read count file: {e}", "summary": f"Could not read {count_file}"}

    # Expect: first column after index = gene name, rest = sample counts
    # Detect if second column is gene names (non-numeric)
    gene_col = None
    sample_cols = []
    for col in counts.columns:
        try:
            pd.to_numeric(counts[col])
            sample_cols.append(col)
        except (ValueError, TypeError):
            if gene_col is None:
                gene_col = col
            else:
                sample_cols.append(col)

    if gene_col:
        genes = counts[gene_col]
        count_matrix = counts[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    else:
        genes = pd.Series(["unknown"] * len(counts), index=counts.index)
        count_matrix = counts.apply(pd.to_numeric, errors="coerce").fillna(0)

    n_sgrnas = len(count_matrix)
    n_samples = len(count_matrix.columns)
    n_genes = genes.nunique()

    if n_sgrnas == 0 or n_samples == 0:
        return {"error": "Count matrix is empty", "summary": "Empty count matrix"}

    # Per-sample QC metrics
    sample_metrics = {}
    for sample in count_matrix.columns:
        values = count_matrix[sample].values.astype(float)
        total_reads = float(np.sum(values))
        mean_reads = float(np.mean(values))
        median_reads = float(np.median(values))

        # Gini index — measures evenness of read distribution
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        cumsum = np.cumsum(sorted_vals)
        if total_reads > 0 and n > 0:
            gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_vals) - (n + 1) * total_reads) / (n * total_reads)
            gini = max(0.0, min(1.0, float(gini)))
        else:
            gini = 1.0

        # Zero-count fraction
        zero_fraction = float(np.sum(values == 0) / n) if n > 0 else 1.0

        # Low-count fraction (< 30 reads)
        low_count_fraction = float(np.sum(values < 30) / n) if n > 0 else 1.0

        sample_metrics[sample] = {
            "total_reads": int(total_reads),
            "mean_reads_per_sgrna": round(mean_reads, 1),
            "median_reads_per_sgrna": round(median_reads, 1),
            "gini_index": round(gini, 4),
            "zero_count_fraction": round(zero_fraction, 4),
            "low_count_fraction": round(low_count_fraction, 4),
        }

    # Mapping rate (if library file provided)
    mapping_rate = None
    if library_file:
        try:
            lib_sep = "\t" if library_file.endswith(".tsv") else ","
            library = pd.read_csv(library_file, sep=lib_sep)
            library_sgrnas = set(library.iloc[:, 0].astype(str))
            detected_sgrnas = set(count_matrix.index.astype(str))
            n_in_library = len(library_sgrnas)
            n_mapped = len(detected_sgrnas & library_sgrnas)
            mapping_rate = {
                "n_library_sgrnas": n_in_library,
                "n_detected_sgrnas": len(detected_sgrnas),
                "n_mapped": n_mapped,
                "mapping_rate": round(n_mapped / n_in_library, 4) if n_in_library > 0 else 0.0,
            }
        except Exception:
            pass

    # Sample correlations (Pearson on log-transformed counts)
    correlations = {}
    if n_samples >= 2:
        log_counts = np.log2(count_matrix + 1)
        corr_matrix = log_counts.corr(method="pearson")
        for i, s1 in enumerate(count_matrix.columns):
            for j, s2 in enumerate(count_matrix.columns):
                if i < j:
                    correlations[f"{s1}_vs_{s2}"] = round(float(corr_matrix.iloc[i, j]), 4)

    # Overall QC verdict
    gini_values = [m["gini_index"] for m in sample_metrics.values()]
    zero_values = [m["zero_count_fraction"] for m in sample_metrics.values()]
    corr_values = list(correlations.values()) if correlations else [1.0]

    avg_gini = np.mean(gini_values)
    avg_zero = np.mean(zero_values)
    min_corr = min(corr_values) if corr_values else 1.0

    flags = []
    if avg_gini > 0.3:
        flags.append(f"High Gini index ({avg_gini:.3f}) — uneven read distribution")
    if avg_zero > 0.1:
        flags.append(f"High zero-count fraction ({avg_zero:.3f}) — significant sgRNA dropout")
    if min_corr < 0.7:
        flags.append(f"Low sample correlation ({min_corr:.3f}) — check replicate consistency")
    if mapping_rate and mapping_rate["mapping_rate"] < 0.8:
        flags.append(f"Low mapping rate ({mapping_rate['mapping_rate']:.3f})")

    if not flags:
        verdict = "PASS"
    elif len(flags) <= 1:
        verdict = "ACCEPTABLE"
    else:
        verdict = "FAIL"

    flag_str = "; ".join(flags) if flags else "All metrics within normal range"

    return {
        "summary": (
            f"CRISPR screen QC: {n_sgrnas} sgRNAs, {n_genes} genes, {n_samples} samples. "
            f"Verdict: {verdict}. {flag_str}"
        ),
        "verdict": verdict,
        "n_sgrnas": n_sgrnas,
        "n_genes": n_genes,
        "n_samples": n_samples,
        "sample_metrics": sample_metrics,
        "mapping_rate": mapping_rate,
        "sample_correlations": correlations,
        "qc_flags": flags,
    }


@registry.register(
    name="crispr.mageck_analyze",
    description="Run MAGeCK statistical analysis on CRISPR screen data to identify enriched/depleted genes",
    category="crispr",
    parameters={
        "count_file": "Path to sgRNA count matrix (MAGeCK format: sgRNA, gene, sample1, sample2, ...)",
        "treatment": "Treatment sample name(s), comma-separated",
        "control": "Control sample name(s), comma-separated",
        "fdr_threshold": "FDR threshold for significance (default 0.05)",
    },
    requires_data=[],
    usage_guide="You have CRISPR screen count data and want to identify gene hits — genes whose knockout causes enrichment (resistance) or depletion (sensitivity). MAGeCK is the gold-standard statistical method. Run screen_qc first to verify data quality.",
)
def mageck_analyze(count_file: str, treatment: str = "", control: str = "", fdr_threshold: float = 0.05, **kwargs) -> dict:
    """Run MAGeCK test for CRISPR screen hit identification."""
    import os
    import tempfile

    count_file = (count_file or "").strip()
    treatment = (treatment or "").strip()
    control = (control or "").strip()

    if not count_file:
        return {"error": "count_file is required", "summary": "No count file provided"}
    if not treatment or not control:
        return {"error": "Both treatment and control sample names are required", "summary": "Specify treatment and control sample names"}

    fdr_threshold = max(0.001, min(float(fdr_threshold or 0.05), 1.0))

    # Check if MAGeCK is installed
    mageck_path = shutil.which("mageck")
    if mageck_path is None:
        return {
            "error": (
                "MAGeCK is not installed. Install with:\n"
                "  pip install mageck\n"
                "Or from source: https://sourceforge.net/projects/mageck/"
            ),
            "summary": "MAGeCK not installed. Install with: pip install mageck",
            "install_instructions": {
                "pip": "pip install mageck",
                "conda": "conda install -c bioconda mageck",
                "source": "https://sourceforge.net/projects/mageck/",
            },
        }

    # Check count file exists
    if not os.path.isfile(count_file):
        return {"error": f"Count file not found: {count_file}", "summary": f"File not found: {count_file}"}

    # Run MAGeCK test
    import subprocess

    with tempfile.TemporaryDirectory(prefix="mageck_") as tmpdir:
        output_prefix = os.path.join(tmpdir, "mageck_output")

        cmd = [
            "mageck", "test",
            "-k", count_file,
            "-t", treatment,
            "-c", control,
            "-n", output_prefix,
            "--normcounts-to-file",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
            )
        except subprocess.TimeoutExpired:
            return {"error": "MAGeCK timed out after 5 minutes", "summary": "MAGeCK analysis timed out"}
        except Exception as e:
            return {"error": f"MAGeCK execution failed: {e}", "summary": f"MAGeCK failed: {e}"}

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()[:500]
            return {"error": f"MAGeCK failed (exit code {result.returncode}): {stderr}", "summary": f"MAGeCK error: {stderr[:100]}"}

        # Parse gene summary results
        import pandas as pd
        gene_summary_file = f"{output_prefix}.gene_summary.txt"
        if not os.path.isfile(gene_summary_file):
            return {"error": "MAGeCK output not found", "summary": "MAGeCK ran but output file missing"}

        try:
            gene_summary = pd.read_csv(gene_summary_file, sep="\t")
        except Exception as e:
            return {"error": f"Failed to parse MAGeCK output: {e}", "summary": "Could not parse MAGeCK gene summary"}

        # Extract significant hits
        neg_hits = []
        pos_hits = []

        neg_fdr_col = [c for c in gene_summary.columns if "neg" in c.lower() and "fdr" in c.lower()]
        pos_fdr_col = [c for c in gene_summary.columns if "pos" in c.lower() and "fdr" in c.lower()]
        neg_lfc_col = [c for c in gene_summary.columns if "neg" in c.lower() and "lfc" in c.lower()]
        pos_lfc_col = [c for c in gene_summary.columns if "pos" in c.lower() and "lfc" in c.lower()]
        neg_score_col = [c for c in gene_summary.columns if "neg" in c.lower() and "score" in c.lower()]
        pos_score_col = [c for c in gene_summary.columns if "pos" in c.lower() and "score" in c.lower()]

        for _, row in gene_summary.iterrows():
            gene_name = row.get("id", row.get("Gene", ""))

            # Negative selection (depleted = essential/sensitizing)
            if neg_fdr_col:
                neg_fdr = float(row.get(neg_fdr_col[0], 1.0))
                neg_lfc = float(row.get(neg_lfc_col[0], 0)) if neg_lfc_col else 0
                neg_score = float(row.get(neg_score_col[0], 0)) if neg_score_col else 0
                if neg_fdr <= fdr_threshold:
                    neg_hits.append({
                        "gene": gene_name,
                        "fdr": round(neg_fdr, 6),
                        "lfc": round(neg_lfc, 4),
                        "score": round(neg_score, 6),
                        "direction": "depleted",
                    })

            # Positive selection (enriched = resistance)
            if pos_fdr_col:
                pos_fdr = float(row.get(pos_fdr_col[0], 1.0))
                pos_lfc = float(row.get(pos_lfc_col[0], 0)) if pos_lfc_col else 0
                pos_score = float(row.get(pos_score_col[0], 0)) if pos_score_col else 0
                if pos_fdr <= fdr_threshold:
                    pos_hits.append({
                        "gene": gene_name,
                        "fdr": round(pos_fdr, 6),
                        "lfc": round(pos_lfc, 4),
                        "score": round(pos_score, 6),
                        "direction": "enriched",
                    })

        # Sort by FDR
        neg_hits.sort(key=lambda x: x["fdr"])
        pos_hits.sort(key=lambda x: x["fdr"])

        # Parse sgRNA-level summary
        sgrna_summary_file = f"{output_prefix}.sgrna_summary.txt"
        n_sgrnas_tested = 0
        if os.path.isfile(sgrna_summary_file):
            try:
                sgrna_df = pd.read_csv(sgrna_summary_file, sep="\t")
                n_sgrnas_tested = len(sgrna_df)
            except Exception:
                pass

        n_genes_tested = len(gene_summary)
        top_neg = ", ".join(h["gene"] for h in neg_hits[:5]) if neg_hits else "none"
        top_pos = ", ".join(h["gene"] for h in pos_hits[:5]) if pos_hits else "none"

        return {
            "summary": (
                f"MAGeCK analysis ({treatment} vs {control}): "
                f"{n_genes_tested} genes tested, "
                f"{len(neg_hits)} depleted hits, {len(pos_hits)} enriched hits (FDR<{fdr_threshold}). "
                f"Top depleted: {top_neg}. Top enriched: {top_pos}"
            ),
            "treatment": treatment,
            "control": control,
            "fdr_threshold": fdr_threshold,
            "n_genes_tested": n_genes_tested,
            "n_sgrnas_tested": n_sgrnas_tested,
            "n_depleted_hits": len(neg_hits),
            "n_enriched_hits": len(pos_hits),
            "depleted_hits": neg_hits[:50],
            "enriched_hits": pos_hits[:50],
        }
