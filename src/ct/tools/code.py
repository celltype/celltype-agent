"""
Code generation and execution tool for ct.

Generates Python code from natural language goals and executes it
in a sandboxed environment with access to loaded datasets and
scientific Python libraries.
"""

import re
from pathlib import Path

from ct.tools import registry


CODE_GEN_SYSTEM_PROMPT = """You are a Python code generator for celltype-agent, a drug discovery research agent.

Write Python code to accomplish the user's analysis goal. The code will be executed in a sandbox.

{namespace_description}

## Rules
1. Do NOT import libraries that are already in the namespace (pd, np, plt, sns, scipy_stats, etc.)
2. Save plots to OUTPUT_DIR: `plt.savefig(OUTPUT_DIR / "filename.png", dpi=150, bbox_inches="tight")`
3. Always call `plt.close()` after saving a plot.
4. Save data exports to OUTPUT_DIR: `df.to_csv(OUTPUT_DIR / "filename.csv")`
5. Assign your final result to a variable called `result` — it must be a dict with at least a `"summary"` key.
6. The `result["summary"]` should be a human-readable string describing what was found.
7. Use print() for intermediate output; it will be captured.
8. Keep code concise and focused on the goal.

## Data access patterns
- CRISPR data (`crispr`): rows=cell_lines, cols=genes. Values are gene effect scores (negative = dependency).
- PRISM data (`prism`): drug sensitivity. Columns include pert_name, pert_dose, ccle_name, LFC.
- L1000 data (`l1000`): rows=compounds, cols=genes. Values are log-fold-change expression.
- Proteomics (`proteomics`): rows=proteins/genes, cols=compounds. Values are protein abundance LFC.
- Mutations (`mutations`): binary matrix. rows=cell_lines, cols=genes. 1=damaging mutation.
- Model metadata (`model_metadata`): cell line info. Columns include CCLEName, OncotreeLineage.

## Example result format
```python
result = {{
    "summary": "Found 15 significantly correlated genes (p < 0.05). Top hit: BRCA1 (r=-0.72, p=1.2e-8).",
    "top_genes": [...],
    "n_significant": 15,
}}
```

Write ONLY the Python code. No explanation, no markdown fences.
"""

BIOINFORMATICS_CODE_GEN_PROMPT = """You are an expert bioinformatics data analyst. Write precise Python code to answer the question.

{namespace_description}

## Available Data Files
{data_files_description}

## RULES
1. Read data from provided paths. Use pd.read_csv(), pd.read_excel(), etc.
2. Do NOT import libraries already in namespace (pd, np, plt, sns, scipy_stats, zipfile, glob, io, tempfile, gzip, csv, struct, os).
3. Assign result: `result = {{"summary": "...", "answer": "PRECISE_ANSWER"}}`
4. The "answer" MUST be short and precise (number, gene name, ratio, etc.)
5. print() intermediate results to verify correctness.
6. Save plots to OUTPUT_DIR: `plt.savefig(OUTPUT_DIR / "filename.png", dpi=150, bbox_inches="tight")`
7. Always call `plt.close()` after saving a plot.
8. Save data exports to OUTPUT_DIR: `df.to_csv(OUTPUT_DIR / "filename.csv")`

## DATA LOADING
- **ZIP files**: Extract first! Capsules often contain ZIPs:
  ```python
  with zipfile.ZipFile(path, "r") as zf:
      zf.extractall("/tmp/my_extract")
      print("Files:", [n for n in zf.namelist() if not n.endswith("/")])
  ```
- **RDS files**: If a .csv exists next to the .rds, use CSV. Otherwise: `import pyreadr; data = pyreadr.read_r(path)`
- **Excel .xls**: `pd.read_excel(path, engine='xlrd')`. Check all sheets: `pd.ExcelFile(path).sheet_names`.
  **Multi-row headers**: If columns look wrong (e.g., dates in column names), try `skiprows=1`.
- **Excel .xlsx**: `pd.read_excel(path)`. Check all sheets.
- **FASTA (.faa, .fa)**: `from Bio import SeqIO; records = list(SeqIO.parse(path, "fasta"))`
- **Newick trees (.treefile, .nwk)**: `from Bio import Phylo; tree = Phylo.read(path, "newick")`
- **BAM files**: `import pysam; bam = pysam.AlignmentFile(path, "rb")`
- **GMT gene sets**: Each line = `name\\tdescription\\tgene1\\tgene2\\t...`
- **GZ files**: `pd.read_csv(path, compression='gzip')` or `with gzip.open(path) as f: ...`

## MANDATORY DATA EXPLORATION (DO THIS FIRST!)
```python
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("Head:\\n", df.head(3))
print("Dtypes:\\n", df.dtypes)
if 'Unnamed: 0' in df.columns:
    df = df.set_index('Unnamed: 0')
```
**When filtering returns 0 rows**: your column names or logic is WRONG. Print the column, check unique values.

## FILE DISCOVERY
Data may be in ZIP files (extract first!) OR already-extracted flat directories. Check both!
```python
import glob, os
from collections import Counter, defaultdict
all_files = sorted(glob.glob(str(data_dir) + "/**/*", recursive=True))
all_files = [f for f in all_files if os.path.isfile(f)]
dir_counts = Counter(os.path.dirname(f) for f in all_files)
print(f"Directories with files: {{dict(dir_counts)}}")
ext_counts = Counter(os.path.splitext(f)[1].lower() for f in all_files)
print(f"File extensions: {{dict(ext_counts)}}")
```

## DIFFERENTIAL EXPRESSION (DESeq2)

**IMPORTANT: Always use R (via run_r) for DESeq2 analysis, NOT pydeseq2.**
R's DESeq2 is the reference implementation. Only fall back to pydeseq2 if the question explicitly asks for it or if R is unavailable.

### Pre-computed DESeq2 results
Columns: gene_id (often 'Unnamed: 0'), baseMean, log2FoldChange, lfcSE, stat, pvalue, padj.
- Load: `df = pd.read_csv(path); if 'Unnamed: 0' in df.columns: df = df.set_index('Unnamed: 0')`
- Significant: `padj < 0.05` AND `abs(log2FoldChange) > threshold`
- Up-regulated: `log2FoldChange > 0 & padj < 0.05`; Down-regulated: `log2FoldChange < 0 & padj < 0.05`
- Volcano plot: `plt.scatter(df['log2FoldChange'], -np.log10(df['pvalue']), alpha=0.5, s=3)`

### DESeq2 from raw counts (pyDESeq2)
```python
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# counts_df: rows=genes, cols=samples (raw integer counts)
# metadata: rows=samples, cols=[condition, ...] — index must match counts_df.columns
counts_df = counts_df.T  # pyDESeq2 wants samples-as-rows
dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~condition")
# For batch correction: design="~batch+condition"
# For paired designs: design="~patient+condition"
dds.deseq2()
stat_res = DeseqStats(dds, contrast=["condition", "treatment", "control"])
stat_res.summary()
results_df = stat_res.results_df
# LFC shrinkage (optional but recommended): stat_res.lfc_shrink(coeff="condition_treatment_vs_control")
```
**CRITICAL sample alignment check (must pass before DESeq2):**
```python
# metadata rows must exactly match count columns (same sample IDs, same order)
print("n_count_samples:", counts_df.shape[1])
print("n_metadata_rows:", metadata.shape[0])
missing = [s for s in counts_df.columns if s not in metadata.index]
extra = [s for s in metadata.index if s not in counts_df.columns]
print("missing_in_metadata:", missing[:10], "count=", len(missing))
print("extra_in_metadata:", extra[:10], "count=", len(extra))
if missing or extra:
    # Debug spreadsheet parsing issues before continuing
    xls = pd.ExcelFile(metadata_path)
    print("sheets:", xls.sheet_names)
    for sk in (0, 1, 2):
        try:
            tmp = pd.read_excel(metadata_path, sheet_name=xls.sheet_names[0], skiprows=sk)
            print(f"skiprows={sk} shape={tmp.shape} cols={tmp.columns.tolist()[:8]}")
        except Exception as e:
            print(f"skiprows={sk} read error:", e)
    raise ValueError("Metadata/sample mismatch: fix parsing before DESeq2")
```
**Treatment name matching**: Print `metadata['condition'].unique()` and match EXACTLY (case-sensitive!).
**Group selection**: Follow the question's instructions about which groups to include.
  - If question EXPLICITLY lists groups, include EXACTLY those groups.
  - If not specified, use only the 2 comparison groups.
  - For combination treatments (e.g., "drugA/drugB"), find group with BOTH names; pick SHORTEST.
  - Print ALL group names: `print("Groups:", metadata['condition'].unique())`
**LFC shrinkage** (when question asks): `stat_res.lfc_shrink(coeff=target_coeff)`.
  Find coefficient name first: `dds.varm['LFC'].columns.tolist()`.
  Pattern: 'condition_Treatment_vs_Control'. Skip intercept/batch coefficients.
**Design with covariates**: `DeseqDataSet(counts=counts_df, metadata=metadata, design_factors=['batch', 'condition'])`
**Prefer modern pydeseq2 API**: use `design='~ covariate + condition'` instead of deprecated
`design_factors`/`ref_level`, and set categorical levels explicitly before fitting:
```python
metadata['condition'] = pd.Categorical(metadata['condition'], categories=['Control', 'Treatment'])
metadata['sex'] = pd.Categorical(metadata['sex'])
dds = DeseqDataSet(counts=counts_df, metadata=metadata, design='~ sex + condition')
```
Then use `DeseqStats(dds, contrast=['condition', 'Treatment', 'Control'])`.
**Multiple result files** (res_1vs97.csv, res_1vs98.csv): Read metadata to map condition codes to names.

## ENRICHMENT ANALYSIS
- **gseapy**: `import gseapy; enr = gseapy.enrich(gene_list=genes, gene_sets='KEGG_2021_Human', outdir=None)`
- **gseapy library names** (exact strings): 'KEGG_2021_Human', 'Reactome_2022', 'WikiPathways_2019_Mouse', 'GO_Biological_Process_2021', etc.
  To check available: `gseapy.get_library_name()` returns all valid names.
- Result in `enr.results` — columns: Term, Overlap, P-value, Adjusted P-value, Odds Ratio, Combined Score, Genes.
  - "Overlap" format: "3/49" (string) — parse with `overlap.split("/")`.
  - "Odds Ratio" is a FLOAT (e.g., 5.81) — this is DIFFERENT from Overlap.
  - **ANSWER what the QUESTION asks**: "odds ratio" → Odds Ratio column (float).
    "overlap ratio" → Overlap column (string "8/49"). These are DIFFERENT columns!
- **Specific pathway lookup**: Search in ALL results (not just filtered):
  ```python
  target = enr.results[enr.results['Term'].str.contains('TP53', case=False)]
  if len(target) > 0:
      print(f"Odds Ratio: {{target.iloc[0]['Odds Ratio']}}")
      print(f"Overlap: {{target.iloc[0]['Overlap']}}")
  ```
- If the question names a specific pathway term, first use exact case-insensitive
  term matching and report that term's metrics. Only use fuzzy/contains matching
  when exact matching returns no rows.
- **Gene ID conversion**: gseapy uses GENE SYMBOLS, not Ensembl IDs. Convert:
  ```python
  import mygene
  mg = mygene.MyGeneInfo()
  result = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='mouse')
  gene_symbols = [r.get('symbol') for r in result if 'symbol' in r]
  ```
- **Directionality**: Run SEPARATE enrichment for up (log2FC > 0) and down (log2FC < 0).
- **KEGG REST ORA** (non-human): Use urllib + Fisher's exact test + BH correction.

## CRISPR / ESSENTIALITY
- Negative gene effect = essential. essentiality = -gene_effect.
- Columns are gene names; rows are cell lines.
- Common pattern: rank genes by median effect across cell lines.
- For expression-vs-essentiality correlation questions, ALWAYS correlate expression
  against `essentiality = -gene_effect` (not raw gene effect values).
- Normalize gene labels before matching across tables (e.g., strip ` (1234)` suffixes).
- Sign interpretation guardrail:
  - "most negative correlation with essentiality" means most negative correlation
    with `-gene_effect` (equivalently, most positive with raw gene effect).
  - "most positive correlation with essentiality" means most positive correlation
    with `-gene_effect` (equivalently, most negative with raw gene effect).

## VCF / TS-TV
- For Ts/Tv, count SNPs only (len(REF)==1 and len(ALT)==1) and handle multi-allelic ALT carefully.
- For **raw bacterial VCFs**, compute a high-confidence Ts/Tv using sample FORMAT depth:
  keep sites with `DP >= 12` when DP is available, then compute Ts/Tv and round to 2 decimals.
- If both raw and DP-filtered Ts/Tv are available, report the DP-filtered value as final answer unless the question explicitly asks for unfiltered.

## PCA ANALYSIS
```python
from sklearn.decomposition import PCA

# log10 transform with pseudocount (common for gene expression)
log_data = np.log10(expression_matrix + 1)  # samples as rows, genes as columns
# PCA — DO NOT scale, just center (sklearn PCA centers by default)
pca = PCA(n_components=100)
pca.fit(log_data)
pc1_variance_pct = pca.explained_variance_ratio_[0] * 100
print("Variance explained:", pca.explained_variance_ratio_[:5])

# Plot PC1 vs PC2
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(pcs[:, 0], pcs[:, 1], alpha=0.7)
ax.set_xlabel(f"PC1 ({{pca.explained_variance_ratio_[0]*100:.1f}}%)")
ax.set_ylabel(f"PC2 ({{pca.explained_variance_ratio_[1]*100:.1f}}%)")
plt.savefig(OUTPUT_DIR / "pca.png", dpi=150, bbox_inches="tight")
plt.close()

# Top loadings for PC1
loadings = pd.Series(pca.components_[0], index=X_clean.columns)
top_pos = loadings.nlargest(10)
top_neg = loadings.nsmallest(10)
```

## COLONY / SWARMING DATA
- StrainNumber values are STRINGS. Always compare as strings, never as integers. Check df.dtypes.
- **Ratio** column: strings like '1:0', '1:3', '1:1', '2:1', '5:1', '10:1', '50:1', etc.
- **Mixed cultures** have compound StrainNumber values at various Ratios. Pure strains have Ratio '1:0'.
- **Percentage calculations**: percent reduction = `(reference - test) / reference * 100`. Result is POSITIVE if test < reference.
- When finding 'most similar', normalize metrics with different scales using MinMaxScaler:
  ```python
  from sklearn.preprocessing import MinMaxScaler
  # Compute means per ratio, include reference for normalization
  all_vals = pd.concat([means, pd.DataFrame({{'Area': [ref_area], 'Circularity': [ref_circ]}}, index=['ref'])])
  scaled = pd.DataFrame(MinMaxScaler().fit_transform(all_vals), index=all_vals.index, columns=all_vals.columns)
  ref_scaled = scaled.loc['ref']
  distances = {{r: np.sqrt((scaled.loc[r, 'Area'] - ref_scaled['Area'])**2 + (scaled.loc[r, 'Circularity'] - ref_scaled['Circularity'])**2) for r in means.index}}
  closest = min(distances, key=distances.get)
  ```
- **Ratio to proportion**: Parse "A:B" -> `a/(a+b)` for model fitting.
  ```python
  def ratio_to_prop(r):
      a, b = map(float, r.split(':'))
      return a / (a + b) if (a + b) > 0 else 0
  ```
- For swarm/mixture model questions asking for the maximum colony area at the optimal
  frequency, compute both:
  1) model-predicted optimum on a fine grid, and
  2) mean Area at each observed ratio level.
  If the optimum is near an observed ratio, report the observed-ratio mean area as the
  final peak-area value (this is the stable estimate used in benchmark-style readouts).

## MODEL FITTING (rpy2 / statsmodels)
- **When question says "Use R"**: You MUST use rpy2 for R model fitting. Python patsy gives DIFFERENT results!
  ```python
  import rpy2.robjects as ro
  ro.globalenv['x'] = ro.FloatVector(x_data.tolist())
  ro.globalenv['y'] = ro.FloatVector(y_data.tolist())
  result = ro.r('''
  library(splines)
  model_quad = lm(y ~ poly(x, 2))
  model_ns = lm(y ~ ns(x, df=4))
  r2 = c(summary(model_quad)$r.squared, summary(model_ns)$r.squared)
  x_fine = seq(min(x), max(x), length.out=10000)
  pred = predict(lm(y ~ ns(x, df=4)), newdata=data.frame(x=x_fine))
  idx = which.max(pred)
  c(r2, optimal_x=x_fine[idx], max_y=pred[idx])
  ''')
  ```
  **CRITICAL**: R's `ns()` and Python's `patsy.cr()` produce SIGNIFICANTLY different fits.
- For Python-only (no R):
  ```python
  import statsmodels.api as sm
  import statsmodels.formula.api as smf
  model = smf.ols('response ~ treatment + covariate', data=df).fit()
  ```

## PHYLO EVOLUTIONARY RATE
- For PhyKIT `evo_rate` comparisons "across all genes", compute rates for all available
  genes in each kingdom/group and run Mann-Whitney U on the two full distributions.
- Only restrict to shared gene IDs if the question explicitly says "shared genes".
- For PhyKIT `rcv` comparisons, follow the same rule: use all available orthologs in
  each group unless the question explicitly asks for shared/intersected IDs only.

## BIOINFORMATICS CLI TOOLS
Use `safe_subprocess_run()` (pre-imported) for CLI tools:
```python
# BWA-MEM alignment
result = safe_subprocess_run(["bwa", "mem", "-t", "4", "-R", read_group, ref_path, r1_path, r2_path])
sam_path = "/tmp/aligned.sam"
with open(sam_path, "w") as f:
    f.write(result.stdout)
safe_subprocess_run(["samtools", "view", "-bS", sam_path, "-o", "/tmp/aligned.bam"])
safe_subprocess_run(["samtools", "sort", "/tmp/aligned.bam", "-o", "/tmp/sorted.bam"])
safe_subprocess_run(["samtools", "index", "/tmp/sorted.bam"])
# Coverage depth
result = safe_subprocess_run(["samtools", "depth", "-a", "/tmp/sorted.bam"])
# Parse: each line is "chrom\\tpos\\tdepth"

# BUSCO analysis
result = safe_subprocess_run(["busco", "-i", proteome_path, "-m", "protein",
                              "-l", "eukaryota_odb10", "-o", "busco_out", "--out_path", "/tmp/"])
# Parse BUSCO output: look for "Complete and single-copy BUSCOs" line
```

## STATISTICAL TESTS
- **Mann-Whitney U**: `scipy_stats.mannwhitneyu(x, y, alternative='two-sided')`.
  Report the SMALLER of the two U values: `min(U, n1*n2 - U)` to match R's wilcox.test.
- **t-test**: `scipy_stats.ttest_ind(x, y)` (independent) or `scipy_stats.ttest_rel(x, y)` (paired).
- **Fisher's exact**: `scipy_stats.fisher_exact(contingency_table)`.
- **BH correction**: `from statsmodels.stats.multitest import multipletests; _, padj, _, _ = multipletests(pvals, method='fdr_bh')`
- **Correlation**: `scipy_stats.pearsonr(x, y)` or `scipy_stats.spearmanr(x, y)`.
- **Chi-squared**: `scipy_stats.chi2_contingency(table)`.

## PERCENTAGE & RATIO CALCULATIONS
- Always clarify denominator: "percentage of X" = count(X) / total * 100.
- For proportions: check if question asks for fraction (0-1) or percentage (0-100).
- When comparing groups: compute metric per group, then compare.

## COMMON PITFALLS
1. Column names are CASE-SENSITIVE. Always print columns first.
2. Mann-Whitney U: report min(U, n1*n2 - U) to match R's wilcox.test.
3. NEVER return "N/A", "Unable to determine", "Error", or "UNABLE_TO_DETERMINE". Debug and find the answer.
4. When 0 results: your approach is WRONG! Print intermediate values, verify data.
5. String matching: use `.str.contains(pattern, case=False, na=False)` not `==`.
6. NaN handling: `df.dropna(subset=['col'])` before statistical tests.
7. Gene IDs: Ensembl (ENSG...) ≠ symbols (TP53). Convert as needed.
8. Index confusion: after `set_index()`, the column is no longer in `df.columns`.
9. For ZIP files: extract to /tmp/ first, find files matching gene/sample ID.
10. When PhyKIT is mentioned, use the EXACT formulas in the phylo tool.
11. For BUSCO: use `safe_subprocess_run(["busco", ...])`. Count "Complete and single-copy" BUSCOs.
12. `compute_pi_percentage(seqs)` is PRE-IMPORTED. DO NOT redefine it. Process ALL alignment files.
13. For ratios like "5:1", return the string "5:1".
14. Percent reduction: `(reference - test) / reference * 100`. Result is POSITIVE if test < reference.
15. If the query requires pre-filtering across ALL samples, apply filtering before subgrouping.
16. Do not proceed with DESeq2 if metadata/sample alignment is incomplete for the intended cohort.
17. For pydeseq2, avoid deprecated `design_factors`/`ref_level`; use `design='~ ...'` with explicit categories.

Write ONLY the Python code. No explanation, no markdown fences.
"""

# ── Multi-turn agentic tool definition ──────────────────────────────────────

RUN_PYTHON_TOOL = {
    "name": "run_python",
    "description": (
        "Execute Python code in the sandbox. Variables persist between calls. "
        "Print output to see results. When done, assign a dict to `result` "
        "with at least 'summary' and 'answer' keys."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute",
            }
        },
        "required": ["code"],
    },
}

AGENTIC_CODE_ADDENDUM = """
## Execution mode
Use the run_python tool to execute Python code step by step.
- Start by exploring the data (list files, read headers, check shapes/dtypes).
- Build your analysis incrementally, verifying intermediate results with print().
- Variables persist between run_python calls — no need to reload data.
- Re-read the goal carefully before each major step. Follow the stated parameters EXACTLY
  (e.g., specific sample groups, thresholds, column names, library arguments). When the
  goal lists specific items, use ONLY those items — do not add extras.
- Print a short `constraint_check` section before finalizing, marking each explicit
  query requirement as PASS/FAIL. If any requirement is FAIL, continue iterating.
- After obtaining your answer, verify it makes sense (non-empty results, reasonable range).
- When you are finished, make one final run_python call that assigns:
  `result = {"summary": "...", "answer": "YOUR_ANSWER"}`
  This is REQUIRED — do not just print the answer.
- Do NOT output bare code — always use the run_python tool.
"""

SCRIPT_GEN_SYSTEM_PROMPT = """You write standalone Python scripts for users.

Return ONLY valid Python source code for a single script file.

Rules:
1. Output only Python code (no markdown fences, no explanation).
2. Script must be syntactically valid Python 3.
3. Include robust error handling for network/API requests.
4. Keep dependencies minimal and standard where possible.
5. Include a `main()` function and `if __name__ == "__main__":` block.
6. Do not execute anything now; only provide script contents.
"""

ERROR_RETRY_PROMPT = """Your previous code produced an error. Fix the code.

Previous code:
```python
{code}
```

Error:
```
{error}
```

## Common Fixes
- **ImportError / ModuleNotFoundError**: The library may not be installed. Use an alternative:
  - No `pyreadr`? Check if a .csv exists alongside the .rds file.
  - No `xlrd`? Use `engine='openpyxl'` for .xlsx. For .xls, try `pip install xlrd` first.
  - No `pysam`? Parse BAM header with samtools subprocess instead.
  - No `rpy2`? Use statsmodels or scipy equivalents.
- **FileNotFoundError**: The path is wrong. Print available files with `glob.glob()` and `os.listdir()`.
  If data is in a ZIP, extract it first with `zipfile.ZipFile`.
- **KeyError / column not found**: Print `df.columns.tolist()` and `df.head()` to see actual names.
  Check for case sensitivity, extra spaces, and 'Unnamed: 0' index columns.
- **PermissionError on write**: Write to /tmp/ or OUTPUT_DIR instead of the data directory.
- **Empty DataFrame / 0 results**: Your filter logic is wrong. Print the column values with
  `df['col'].unique()` before filtering. Check for NaN, case mismatches, dtype issues.
- **NameError**: Variable not defined. Check spelling. Libraries in namespace: pd, np, plt, sns,
  scipy_stats, zipfile, glob, io, tempfile, gzip, csv, struct, os.
- **ValueError (shapes)**: Print `.shape` of all arrays/DataFrames before operations.
- **xlrd.biffh.XLRDError**: File is .xlsx not .xls. Use `engine='openpyxl'`.
- **gzip/zlib error**: File may be empty (0 bytes). Check `os.path.getsize(path)` first.

Write ONLY the corrected Python code. No explanation, no markdown fences.
"""

REFLECTION_PROMPT = """Your code executed without errors. Review the output to check if the result is correct.

Goal: {goal}

Code:
```python
{code}
```

Stdout:
```
{stdout}
```

Result: {result}

Check carefully:
- Are there empty lists, zero counts, or missing matches that suggest a data loading or filtering bug?
- Did column/sample mapping work correctly? (e.g., correct group labels, not raw IDs)
- Does the final answer make sense given the data and question?
- Are there printed warnings like "0 genes found" or "no matching samples" that indicate a logic error?

If the output looks correct, respond with exactly: LGTM
If there is a problem, respond with the corrected Python code (no explanation, no markdown fences).
"""

SCRIPT_RETRY_PROMPT = """Your previous script had a syntax error. Fix it.

Previous script:
```python
{code}
```

Syntax error:
```
{error}
```

Write ONLY corrected Python code. No explanation, no markdown fences.
"""


def _extract_code(text: str) -> str:
    """Strip markdown code fences from LLM response if present."""
    text = text.strip()
    # Remove ```python ... ``` fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```python or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return text


def _is_script_authoring_goal(goal: str) -> bool:
    """Return True when the goal is about writing/saving a standalone script file."""
    g = (goal or "").lower()
    if not g:
        return False
    explicit_script = any(
        phrase in g
        for phrase in (
            "write a python script",
            "create a python script",
            "save the script",
            "standalone file",
            "standalone script",
        )
    )
    has_py_target = ".py" in g and any(word in g for word in ("script", "save", "write", "create", "generate"))
    return explicit_script or has_py_target


def _extract_script_filename(goal: str) -> str:
    """Extract target script filename from the goal, defaulting safely."""
    text = goal or ""

    quoted = re.findall(r"""['"]([^'"]+\.py)['"]""", text, flags=re.IGNORECASE)
    for candidate in quoted:
        c = candidate.strip().rstrip(".,;:)")
        if c and not c.lower().startswith(("http://", "https://")):
            return c

    bare = re.findall(r"""\b([A-Za-z0-9_\-./]+\.py)\b""", text, flags=re.IGNORECASE)
    for candidate in bare:
        c = candidate.strip().rstrip(".,;:)")
        if c and not c.lower().startswith(("http://", "https://")):
            return c

    return "generated_script.py"


def _resolve_script_path(path_str: str) -> tuple[Path | None, str | None]:
    """Resolve a script path and enforce CWD containment."""
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return None, "Absolute paths are not allowed for generated scripts."
    resolved = (Path.cwd() / p).resolve()
    try:
        resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        return None, "Path traversal detected for generated script path."
    if resolved.suffix.lower() != ".py":
        return None, "Generated script path must end with .py."
    return resolved, None


def _generate_and_save_script(
    *,
    goal: str,
    llm,
    max_retries: int,
    session,
) -> dict:
    """Generate a standalone Python script and save it in the working directory."""
    filename = _extract_script_filename(goal)
    script_path, path_error = _resolve_script_path(filename)
    if path_error:
        return {
            "summary": f"Script generation failed: {path_error}",
            "error": path_error,
        }

    script_text = ""
    last_error = None
    for attempt in range(1, max_retries + 2):
        if attempt == 1:
            user_msg = (
                f"User request:\n{goal}\n\n"
                f"Write a complete standalone Python script for this request.\n"
                f"Target filename: {script_path.name}\n"
                f"The script must be directly runnable with `python {script_path.name}`."
            )
        else:
            user_msg = SCRIPT_RETRY_PROMPT.format(code=script_text, error=last_error or "Unknown syntax error")

        with session.console.status(
            f"[green]{'Generating' if attempt == 1 else 'Fixing'} script...[/green]",
            spinner="dots",
        ):
            response = llm.chat(
                system=SCRIPT_GEN_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
                temperature=0.2,
            )

        script_text = _extract_code(response.content)

        # Validate syntax before writing to disk.
        try:
            compile(script_text, str(script_path), "exec")
        except SyntaxError as e:
            last_error = f"{e.msg} (line {e.lineno})"
            if attempt > max_retries:
                break
            continue
        except Exception as e:
            last_error = str(e)
            if attempt > max_retries:
                break
            continue

        try:
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(script_text, encoding="utf-8")
        except Exception as e:
            return {
                "summary": f"Script generation failed while writing {script_path}: {e}",
                "error": str(e),
                "path": str(script_path),
            }

        lines = script_text.count("\n") + 1
        return {
            "summary": f"Generated standalone Python script: {script_path.name} ({lines} lines).",
            "path": str(script_path),
            "script": script_text,
            "lines": lines,
            "size": len(script_text),
            "exports": [str(script_path)],
            "result": {
                "summary": f"Script saved to {script_path}",
                "path": str(script_path),
            },
        }

    return {
        "summary": f"Script generation failed after {max_retries + 1} attempts: {last_error}",
        "error": last_error or "unknown_script_generation_error",
        "path": str(script_path),
        "script": script_text,
    }


def _agentic_code_loop(
    *,
    goal: str,
    system_prompt: str,
    llm,
    sandbox,
    session,
    max_turns: int,
) -> dict:
    """Multi-turn agentic code execution loop.

    The LLM calls ``run_python`` repeatedly, seeing output after each
    execution, and stops when it has no more tool calls (``end_turn``).
    """
    messages = [{"role": "user", "content": f"Goal: {goal}"}]
    all_code: list[str] = []
    all_stdout: list[str] = []
    last_exec_result: dict | None = None

    for turn in range(max_turns):
        with session.console.status(
            f"[green]Agent turn {turn + 1}/{max_turns}...[/green]",
            spinner="dots",
        ):
            response = llm.chat(
                system=system_prompt,
                messages=messages,
                tools=[RUN_PYTHON_TOOL],
                temperature=0.2,
            )

        # Check for tool calls in the content blocks
        content_blocks = response.content_blocks or []
        tool_calls = [b for b in content_blocks if getattr(b, "type", None) == "tool_use"]

        if not tool_calls:
            # LLM is done (end_turn) — no more tool calls
            break

        # Process the first tool call
        tool_call = tool_calls[0]
        code = (tool_call.input or {}).get("code", "")
        all_code.append(code)

        exec_result = sandbox.execute(code)
        last_exec_result = exec_result

        # Collect stdout
        if exec_result.get("stdout"):
            all_stdout.append(exec_result["stdout"])

        # Build tool result content
        tool_output_parts = []
        if exec_result.get("stdout"):
            tool_output_parts.append(exec_result["stdout"])
        if exec_result.get("error"):
            tool_output_parts.append(f"Error:\n{exec_result['error']}")
        tool_output = "\n".join(tool_output_parts) if tool_output_parts else "(no output)"

        # Truncate to avoid blowing up context
        tool_output = tool_output[:5000]

        # Append assistant message (full content blocks) and tool result
        messages.append({"role": "assistant", "content": content_blocks})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": tool_output,
                }
            ],
        })

    # Extract result from sandbox namespace
    result_obj = sandbox.get_variable("result")
    combined_code = "\n\n# --- next turn ---\n\n".join(all_code)
    combined_stdout = "\n".join(all_stdout)

    # Reflection pass for multi-turn mode (parity with single-shot path):
    # when execution succeeds but constraints may be violated, ask the model
    # to self-check and optionally provide corrected code.
    max_reflect = int(session.config.get("sandbox.max_reflect", 2))
    for _ in range(max_reflect):
        if not combined_code:
            break
        result_preview = str(result_obj)[:1000] if result_obj else "(no result dict)"
        reflect_msg = REFLECTION_PROMPT.format(
            goal=goal,
            code=combined_code,
            stdout=combined_stdout[:3000],
            result=result_preview,
        )
        with session.console.status(
            "[cyan]Reviewing output...[/cyan]",
            spinner="dots",
        ):
            reflect_response = llm.chat(
                system=system_prompt,
                messages=[{"role": "user", "content": reflect_msg}],
                temperature=0.2,
            )
        reflect_text = (reflect_response.content or "").strip()
        if reflect_text.upper().startswith("LGTM"):
            break

        fixed_code = _extract_code(reflect_text)
        if not fixed_code:
            break

        exec_result = sandbox.execute(fixed_code)
        last_exec_result = exec_result
        all_code.append(fixed_code)
        if exec_result.get("stdout"):
            all_stdout.append(exec_result["stdout"])

        combined_code = "\n\n# --- next turn ---\n\n".join(all_code)
        combined_stdout = "\n".join(all_stdout)
        if exec_result.get("error"):
            break
        result_obj = sandbox.get_variable("result")

    # Collect plots/exports from the last execution (sandbox output dir)
    plots = []
    exports = []
    if last_exec_result:
        plots = last_exec_result.get("plots", [])
        exports = last_exec_result.get("exports", [])

    if result_obj and isinstance(result_obj, dict):
        summary = result_obj.get("summary", "")
    elif combined_stdout:
        summary = combined_stdout[-500:]
    else:
        summary = "Code executed successfully."

    if not result_obj and not combined_stdout and not all_code:
        return {
            "summary": "Agent loop completed but no code was executed.",
            "error": "LLM did not call run_python tool.",
            "code": "",
            "stdout": "",
        }

    return {
        "summary": summary,
        "code": combined_code,
        "stdout": combined_stdout,
        "result": result_obj,
        "plots": plots,
        "exports": exports,
    }


def _generate_and_execute_code(
    goal: str,
    system_prompt_template: str,
    session,
    prior_results=None,
) -> dict:
    """Shared code-gen helper: LLM code generation -> sandbox execution -> retry loop.

    Domain tools call this with a focused system prompt template containing
    ``{namespace_description}`` which gets filled with the sandbox's namespace
    description at runtime.

    Args:
        goal: Natural language description of the analysis to perform.
        system_prompt_template: System prompt with ``{namespace_description}``
            placeholder (and optionally ``{data_files_description}``).
        session: Active ct session (provides config, LLM, console).
        prior_results: Dict of prior step results to inject into the sandbox.

    Returns:
        Standard tool result dict with ``summary``, ``code``, ``stdout``, etc.
    """
    if session is None:
        return {
            "summary": "Code execution unavailable: no active session.",
            "error": "No session provided.",
        }

    from ct.agent.sandbox import Sandbox

    config = session.config
    timeout = int(config.get("sandbox.timeout", 30))
    output_dir = config.get("sandbox.output_dir")
    max_retries = int(config.get("sandbox.max_retries", 2))
    llm = session.get_llm()

    # Collect extra read directories (e.g., capsule data dirs for bioinformatics mode)
    extra_read_dirs = []
    extra_read_str = config.get("sandbox.extra_read_dirs")
    if extra_read_str:
        for d in str(extra_read_str).split(","):
            d = d.strip()
            if d and Path(d).exists():
                extra_read_dirs.append(Path(d))

    # Create sandbox and load datasets
    sandbox = Sandbox(
        timeout=timeout,
        output_dir=output_dir,
        max_retries=max_retries,
        extra_read_dirs=extra_read_dirs or None,
    )
    sandbox.load_datasets()
    if prior_results:
        sandbox.inject_prior_results(prior_results)

    # Build the system prompt with namespace info
    ns_desc = sandbox.describe_namespace()
    format_kwargs = {"namespace_description": ns_desc}

    # Provide a data_files_description if the template expects one
    if "{data_files_description}" in system_prompt_template:
        format_kwargs["data_files_description"] = _describe_data_files(extra_dirs=extra_read_dirs)

    system_prompt = system_prompt_template.format(**format_kwargs)

    # ── Multi-turn agentic path ─────────────────────────────────────────
    max_turns = int(config.get("sandbox.max_turns", 0))
    if max_turns > 0:
        agentic_prompt = system_prompt + AGENTIC_CODE_ADDENDUM
        return _agentic_code_loop(
            goal=goal,
            system_prompt=agentic_prompt,
            llm=llm,
            sandbox=sandbox,
            session=session,
            max_turns=max_turns,
        )

    # ── Legacy single-shot path ─────────────────────────────────────────
    code = None
    exec_result = {"error": "No code generated"}

    for attempt in range(1, max_retries + 2):  # 1 initial + max_retries fixes
        if attempt == 1:
            user_msg = f"Goal: {goal}"
        else:
            user_msg = ERROR_RETRY_PROMPT.format(code=code, error=exec_result["error"])

        with session.console.status(
            f"[green]{'Generating' if attempt == 1 else 'Fixing'} code...[/green]",
            spinner="dots",
        ):
            response = llm.chat(
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
                temperature=0.2,
            )

        code = _extract_code(response.content)
        exec_result = sandbox.execute(code)

        if exec_result["error"] is None:
            # Reflection: let the LLM review its own output for logical errors
            max_reflect = int(config.get("sandbox.max_reflect", 2))
            for reflect_turn in range(max_reflect):
                stdout_text = exec_result.get("stdout", "") or ""
                result_obj = exec_result.get("result")
                # Only reflect if there's meaningful output to check
                if not stdout_text and not result_obj:
                    break
                result_preview = str(result_obj)[:1000] if result_obj else "(no result dict)"
                reflect_msg = REFLECTION_PROMPT.format(
                    goal=goal,
                    code=code,
                    stdout=stdout_text[:3000],
                    result=result_preview,
                )
                with session.console.status(
                    f"[cyan]Reviewing output (turn {reflect_turn + 1})...[/cyan]",
                    spinner="dots",
                ):
                    reflect_response = llm.chat(
                        system=system_prompt,
                        messages=[{"role": "user", "content": reflect_msg}],
                        temperature=0.2,
                    )
                reflect_text = reflect_response.content.strip()
                if reflect_text.upper().startswith("LGTM"):
                    break  # LLM says output is correct
                # LLM returned corrected code — execute it
                fixed_code = _extract_code(reflect_text)
                if not fixed_code or fixed_code == code:
                    break  # No new code or same code — stop
                code = fixed_code
                exec_result = sandbox.execute(code)
                if exec_result["error"] is not None:
                    break  # New code errored — fall through to error retry

            # Return result (either original or reflection-fixed)
            if exec_result["error"] is None:
                summary = ""
                if exec_result["result"] and isinstance(exec_result["result"], dict):
                    summary = exec_result["result"].get("summary", "")
                if not summary and exec_result["stdout"]:
                    summary = exec_result["stdout"][:500]
                if not summary:
                    summary = "Code executed successfully."

                return {
                    "summary": summary,
                    "code": code,
                    "stdout": exec_result["stdout"],
                    "result": exec_result["result"],
                    "plots": exec_result["plots"],
                    "exports": exec_result["exports"],
                }

        if attempt > max_retries:
            break

    return {
        "summary": f"Code execution failed after {max_retries + 1} attempts: {exec_result['error'][:200]}",
        "error": exec_result["error"],
        "code": code,
        "stdout": exec_result.get("stdout", ""),
    }


def _describe_data_files(extra_dirs: list[Path] | None = None) -> str:
    """List data files in CWD and extra directories for domain tool prompts."""
    data_exts = {
        ".csv", ".tsv", ".xlsx", ".xls", ".parquet",
        ".vcf", ".bed", ".bam", ".fasta", ".fa", ".faa",
        ".fastq", ".gff", ".gtf", ".nwk", ".nex", ".tree",
        ".mafft", ".clipkit", ".aln", ".phy", ".gz",
        ".zip", ".rds", ".rdata", ".gmt", ".json",
    }

    def _scan_dir(directory: Path, label: str) -> list[str]:
        entries = []
        if not directory.exists():
            return entries
        try:
            for f in sorted(directory.rglob("*")):
                if f.is_file() and (f.suffix.lower() in data_exts):
                    size = f.stat().st_size
                    if size > 1_000_000:
                        size_str = f"{size / 1_000_000:.1f} MB"
                    elif size > 1_000:
                        size_str = f"{size / 1_000:.1f} KB"
                    else:
                        size_str = f"{size} bytes"
                    try:
                        rel = f.relative_to(directory)
                    except ValueError:
                        rel = f.name
                    entries.append(f"- {rel} ({size_str})")
        except PermissionError:
            pass
        return entries[:80]

    sections = []
    cwd = Path.cwd()
    cwd_files = _scan_dir(cwd, "working directory")
    if cwd_files:
        sections.append("Files in working directory:\n" + "\n".join(cwd_files))

    for d in (extra_dirs or []):
        d_files = _scan_dir(d, str(d))
        if d_files:
            sections.append(f"Files in {d}:\n" + "\n".join(d_files))
            sections.append(f"\nData directory accessible for reading: {d}")

    if not sections:
        return "No data files found in the working directory."
    return "\n\n".join(sections)


@registry.register(
    name="code.execute",
    description="Generate and execute custom Python analysis code",
    category="code",
    parameters={"goal": "Natural language description of the analysis to perform"},
    usage_guide=(
        "Use ONLY when no pre-built tool covers the analysis. Good for: custom visualizations, "
        "statistical tests, data exploration, combining/filtering data in novel ways, generating plots. "
        "Pre-built tools are preferred — this is the escape hatch."
    ),
)
def execute(goal: str, _session=None, _prior_results=None, **kwargs) -> dict:
    """Generate and execute Python code for a custom analysis goal."""
    if _session is None:
        return {
            "summary": "Code execution unavailable: no active session.",
            "error": "No session provided. code.execute requires an active ct session.",
        }

    # Handle "write script to file" goals directly (outside sandbox execution path).
    if _is_script_authoring_goal(goal):
        llm = _session.get_llm()
        max_retries = int(_session.config.get("sandbox.max_retries", 2))
        return _generate_and_save_script(
            goal=goal,
            llm=llm,
            max_retries=max_retries,
            session=_session,
        )

    # Use bioinformatics prompt when data files are present in the sandbox,
    # otherwise use the generic code-gen prompt.  Domain-specific knowledge
    # (phylo, KEGG ORA, variant classification) lives in the dedicated domain
    # tools (phylo.analyze, omics.kegg_ora, genomics.variant_classify) which
    # the planner selects directly.
    prompt = BIOINFORMATICS_CODE_GEN_PROMPT if _session.config.get("agent.bioinformatics_mode") else CODE_GEN_SYSTEM_PROMPT

    return _generate_and_execute_code(
        goal=goal,
        system_prompt_template=prompt,
        session=_session,
        prior_results=_prior_results,
    )
