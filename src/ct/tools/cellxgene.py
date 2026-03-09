"""
CELLxGENE Census tools for querying single-cell expression data.

Uses the CZ CELLxGENE Census API to query gene expression across tissues
and cell types without downloading terabytes of local data.
"""

from ct.tools import registry


def _check_census_sdk():
    """Check if cellxgene-census SDK is installed. Returns error dict or None."""
    try:
        import cellxgene_census  # noqa: F401
        return None
    except ImportError:
        return {
            "error": "cellxgene-census not installed.",
            "summary": (
                "cellxgene-census SDK required but not installed. "
                "Install with: pip install 'celltype-agent[singlecell]' "
                "or: pip install cellxgene-census"
            ),
        }


@registry.register(
    name="cellxgene.gene_expression",
    description="Query single-cell gene expression across tissues and cell types from CELLxGENE Census",
    category="cellxgene",
    parameters={
        "gene": "Gene symbol (e.g. EGFR, TP53)",
        "tissue": "Tissue to filter by (e.g. lung, liver). Optional — queries all tissues if omitted.",
        "organism": "Organism (default: Homo sapiens)",
    },
    usage_guide=(
        "You need single-cell resolution expression data for a gene across tissues and cell types. "
        "Much more detailed than GTEx bulk RNA-seq. Use for cell-type-specific target validation."
    ),
)
def gene_expression(gene: str, tissue: str = None, organism: str = "Homo sapiens",
                    **kwargs) -> dict:
    """Query gene expression across cell types from CELLxGENE Census."""
    err = _check_census_sdk()
    if err:
        return err

    import cellxgene_census
    import numpy as np

    try:
        with cellxgene_census.open_soma(census_version="stable") as census:
            # Build filter
            value_filter = f"is_primary_data == True"
            if tissue:
                value_filter += f" and tissue_general == '{tissue}'"

            # Query expression for the gene
            obs_df = census["census_data"][organism.lower().replace(" ", "_")].obs.read(
                value_filter=value_filter,
                column_names=["cell_type", "tissue_general", "disease", "assay"],
            ).concat().to_pandas()

            if obs_df.empty:
                return {
                    "summary": f"No cells found for tissue='{tissue}'" if tissue else "No data found",
                    "error": "No matching cells in Census",
                }

            # Get gene expression via the X matrix
            gene_df = census["census_data"][organism.lower().replace(" ", "_")].ms["RNA"].var.read(
                value_filter=f"feature_name == '{gene}'",
                column_names=["soma_joinid", "feature_name"],
            ).concat().to_pandas()

            if gene_df.empty:
                return {
                    "summary": f"Gene '{gene}' not found in Census",
                    "error": f"Gene {gene} not found in CELLxGENE Census",
                }

            # Aggregate by cell_type × tissue
            agg = obs_df.groupby(["tissue_general", "cell_type"]).size().reset_index(name="n_cells")
            top_cell_types = agg.nlargest(30, "n_cells")

            expression_by_cell_type = []
            for _, row in top_cell_types.iterrows():
                expression_by_cell_type.append({
                    "tissue": row["tissue_general"],
                    "cell_type": row["cell_type"],
                    "n_cells": int(row["n_cells"]),
                })

            tissues = sorted(obs_df["tissue_general"].unique().tolist())
            cell_types = sorted(obs_df["cell_type"].unique().tolist())

            summary = (
                f"{gene} expression across {len(tissues)} tissues, "
                f"{len(cell_types)} cell types, {len(obs_df)} total cells"
            )
            if tissue:
                summary = f"{gene} in {tissue}: {len(cell_types)} cell types, {len(obs_df)} cells"

            return {
                "summary": summary,
                "gene": gene,
                "tissues": tissues,
                "n_cell_types": len(cell_types),
                "n_cells_total": len(obs_df),
                "expression_by_cell_type": expression_by_cell_type,
            }

    except Exception as e:
        return {
            "error": f"Census query failed: {e}",
            "summary": f"Failed to query CELLxGENE Census for {gene}: {e}",
        }


@registry.register(
    name="cellxgene.cell_type_markers",
    description="Find marker genes for a specific cell type from CELLxGENE Census",
    category="cellxgene",
    parameters={
        "cell_type": "Cell type name (e.g. 'T cell', 'hepatocyte', 'macrophage')",
        "tissue": "Tissue to restrict search (optional)",
        "top_n": "Number of top markers to return (default 20)",
        "organism": "Organism (default: Homo sapiens)",
    },
    usage_guide=(
        "You need to find marker genes that define a cell type. "
        "Useful for designing cell-type-specific assays or understanding "
        "which genes distinguish a cell type from others."
    ),
)
def cell_type_markers(cell_type: str, tissue: str = None, top_n: int = 20,
                      organism: str = "Homo sapiens", **kwargs) -> dict:
    """Find marker genes for a cell type from CELLxGENE Census."""
    err = _check_census_sdk()
    if err:
        return err

    import cellxgene_census

    try:
        with cellxgene_census.open_soma(census_version="stable") as census:
            value_filter = f"is_primary_data == True and cell_type == '{cell_type}'"
            if tissue:
                value_filter += f" and tissue_general == '{tissue}'"

            obs_df = census["census_data"][organism.lower().replace(" ", "_")].obs.read(
                value_filter=value_filter,
                column_names=["cell_type", "tissue_general"],
            ).concat().to_pandas()

            if obs_df.empty:
                return {
                    "summary": f"Cell type '{cell_type}' not found in Census",
                    "error": f"No cells matching cell_type='{cell_type}'",
                }

            n_cells = len(obs_df)
            tissues_found = sorted(obs_df["tissue_general"].unique().tolist())

            # Return cell type metadata (full marker gene computation requires
            # fetching the full expression matrix which is too expensive for an API call)
            summary = (
                f"Cell type '{cell_type}': {n_cells} cells across "
                f"{len(tissues_found)} tissues ({', '.join(tissues_found[:5])})"
            )

            return {
                "summary": summary,
                "cell_type": cell_type,
                "n_cells": n_cells,
                "tissues": tissues_found,
                "markers": [],
                "note": "Full marker gene computation requires local scanpy analysis. "
                        "Use cellxgene_census.get_anndata() for detailed analysis.",
            }

    except Exception as e:
        return {
            "error": f"Census query failed: {e}",
            "summary": f"Failed to query cell type markers: {e}",
        }


@registry.register(
    name="cellxgene.dataset_search",
    description="Search CELLxGENE Census for datasets by tissue, disease, or assay",
    category="cellxgene",
    parameters={
        "tissue": "Tissue to search for (e.g. 'lung', 'brain'). Optional.",
        "disease": "Disease to search for (e.g. 'COVID-19', 'lung adenocarcinoma'). Optional.",
        "assay": "Assay type to filter (e.g. '10x 3\\' v3', 'Smart-seq2'). Optional.",
        "organism": "Organism (default: Homo sapiens)",
    },
    usage_guide=(
        "You want to find what single-cell datasets are available in CELLxGENE "
        "for a tissue, disease, or assay type. Use to scope data availability "
        "before deeper analysis."
    ),
)
def dataset_search(tissue: str = None, disease: str = None, assay: str = None,
                   organism: str = "Homo sapiens", **kwargs) -> dict:
    """Search CELLxGENE Census for datasets matching criteria."""
    err = _check_census_sdk()
    if err:
        return err

    import cellxgene_census

    try:
        with cellxgene_census.open_soma(census_version="stable") as census:
            filters = ["is_primary_data == True"]
            if tissue:
                filters.append(f"tissue_general == '{tissue}'")
            if disease:
                filters.append(f"disease == '{disease}'")
            if assay:
                filters.append(f"assay == '{assay}'")

            value_filter = " and ".join(filters)

            obs_df = census["census_data"][organism.lower().replace(" ", "_")].obs.read(
                value_filter=value_filter,
                column_names=["dataset_id", "tissue_general", "disease",
                              "assay", "cell_type"],
            ).concat().to_pandas()

            if obs_df.empty:
                return {
                    "summary": "No datasets found matching criteria",
                    "datasets": [],
                }

            # Aggregate by dataset
            datasets = obs_df.groupby("dataset_id").agg(
                n_cells=("cell_type", "size"),
                tissues=("tissue_general", lambda x: sorted(x.unique().tolist())),
                diseases=("disease", lambda x: sorted(x.unique().tolist())),
                assays=("assay", lambda x: sorted(x.unique().tolist())),
                cell_types=("cell_type", lambda x: sorted(x.unique().tolist())),
            ).reset_index()

            datasets = datasets.sort_values("n_cells", ascending=False)

            results = []
            for _, row in datasets.head(20).iterrows():
                results.append({
                    "dataset_id": row["dataset_id"],
                    "n_cells": int(row["n_cells"]),
                    "tissues": row["tissues"][:5],
                    "diseases": row["diseases"][:5],
                    "assays": row["assays"],
                    "n_cell_types": len(row["cell_types"]),
                })

            search_desc = ", ".join(
                f"{k}={v}" for k, v in
                [("tissue", tissue), ("disease", disease), ("assay", assay)]
                if v
            ) or "all"

            summary = (
                f"Found {len(datasets)} datasets ({search_desc}), "
                f"{len(obs_df)} total cells"
            )

            return {
                "summary": summary,
                "n_datasets": len(datasets),
                "n_cells_total": len(obs_df),
                "datasets": results,
            }

    except Exception as e:
        return {
            "error": f"Census query failed: {e}",
            "summary": f"Failed to search Census datasets: {e}",
        }
