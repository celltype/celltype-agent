"""
Genomics tools: GWAS lookup, eQTL analysis, variant annotation, Mendelian randomization.

These are REST/GraphQL API wrappers -- no local data required.
"""

import math

from ct.tools import registry
from ct.tools.http_client import request, request_json


@registry.register(
    name="genomics.gwas_lookup",
    description="Query the GWAS Catalog for genetic associations for a gene, optionally filtered by trait",
    category="genomics",
    parameters={
        "gene": "Gene symbol (e.g. 'BRCA1', 'TP53')",
        "trait": "Trait or disease name to filter (optional)",
        "p_threshold": "P-value threshold for significance (default 5e-8)",
    },
    requires_data=[],
    usage_guide="You want to find genome-wide significant genetic associations for a specific gene. Optionally add a trait filter to focus disease context.",
)
def gwas_lookup(gene: str = None, trait: str = None, p_threshold: float = 5e-8, **kwargs) -> dict:
    """Query the NHGRI-EBI GWAS Catalog REST API for genetic associations."""
    try:
        import httpx
    except ImportError:
        return {"error": "httpx required (pip install httpx)", "summary": "httpx required (pip install httpx)"}
    gene = str(gene or "").strip()
    trait = str(trait or "").strip() or None
    if not gene:
        detail = f" (trait='{trait}')" if trait else ""
        return {
            "error": f"Missing required parameter: gene{detail}",
            "summary": "GWAS lookup requires a non-empty gene symbol (e.g., SNCA, APOE).",
            "gene": gene,
            "trait_filter": trait,
            "suggestion": (
                "First identify candidate genes (e.g., with data_api.opentargets_search), "
                "then run genomics.gwas_lookup with one gene at a time."
            ),
        }

    base = "https://www.ebi.ac.uk/gwas/rest/api"

    # Step 1: Find SNPs associated with the gene
    snp_url = f"{base}/singleNucleotidePolymorphisms/search/findByGene"
    params = {"geneName": gene, "size": 100}

    data, error = request_json(
        "GET",
        snp_url,
        params=params,
        timeout=30,
        retries=2,
    )
    if error:
        return {"error": f"GWAS Catalog query failed: {error}", "summary": f"GWAS Catalog query failed: {error}"}
    embedded = data.get("_embedded", {})
    snps = embedded.get("singleNucleotidePolymorphisms", [])

    if not snps:
        return {
            "summary": f"No GWAS associations found for gene {gene}",
            "gene": gene,
            "associations": [],
            "n_associations": 0,
        }

    # Step 2: For each SNP, fetch associations using the summary projection
    # which embeds EFO traits inline (avoids extra per-trait API calls)
    associations = []
    seen = set()

    for snp_entry in snps[:30]:  # Cap at 30 SNPs to limit API calls
        rsid = snp_entry.get("rsId", "")
        if not rsid:
            continue

        # Use the associationBySnp projection which embeds traits inline
        assoc_url = f"{base}/singleNucleotidePolymorphisms/{rsid}/associations"
        assoc_data, assoc_error = request_json(
            "GET",
            assoc_url,
            params={"projection": "associationBySnp"},
            timeout=10,
            retries=2,
        )
        if assoc_error:
            continue

        assoc_list = assoc_data.get("_embedded", {}).get("associations", [])

        for assoc in assoc_list:
            pval_mantissa = assoc.get("pvalueMantissa")
            pval_exponent = assoc.get("pvalueExponent")
            if pval_mantissa is not None and pval_exponent is not None:
                try:
                    pval = float(pval_mantissa) * (10 ** int(pval_exponent))
                except (ValueError, TypeError):
                    pval = None
            else:
                pval = None

            # Filter by p-value threshold
            if pval is not None and pval > p_threshold:
                continue

            # Extract risk allele info from loci
            loci = assoc.get("loci", [])
            risk_allele_name = ""
            if loci:
                risk_alleles = loci[0].get("strongestRiskAlleles", [])
                if risk_alleles:
                    risk_allele_name = risk_alleles[0].get("riskAlleleName", "")

            # Extract traits from embedded efoTraits (no extra API call needed)
            efo_traits = assoc.get("efoTraits", [])
            trait_names = [t.get("trait", "") for t in efo_traits if t.get("trait")]
            trait_name = "; ".join(trait_names)

            # Filter by trait if specified
            if trait and trait_name:
                if trait.lower() not in trait_name.lower():
                    continue

            or_value = assoc.get("orPerCopyNum")
            beta = assoc.get("betaNum")
            beta_unit = assoc.get("betaUnit", "")
            beta_direction = assoc.get("betaDirection", "")

            assoc_id = f"{rsid}_{pval}_{trait_name}"
            if assoc_id in seen:
                continue
            seen.add(assoc_id)

            associations.append({
                "rsid": rsid,
                "risk_allele": risk_allele_name,
                "p_value": pval,
                "p_value_str": f"{pval_mantissa}e{pval_exponent}" if pval_mantissa else None,
                "trait": trait_name,
                "or_per_copy": or_value,
                "beta": beta,
                "beta_unit": beta_unit,
                "beta_direction": beta_direction,
                "mapped_gene": gene,
            })

        # Stop early if we have enough
        if len(associations) >= 50:
            break

    # Sort by p-value (most significant first)
    associations.sort(key=lambda x: x["p_value"] if x["p_value"] is not None else 1.0)

    trait_str = f" for trait '{trait}'" if trait else ""
    return {
        "summary": (
            f"GWAS associations for {gene}{trait_str}: "
            f"{len(associations)} genome-wide significant hits (p < {p_threshold})"
        ),
        "gene": gene,
        "trait_filter": trait,
        "p_threshold": p_threshold,
        "n_associations": len(associations),
        "associations": associations[:30],  # Return top 30
    }


@registry.register(
    name="genomics.eqtl_lookup",
    description="Query GTEx for expression quantitative trait loci (eQTLs) for a gene across tissues",
    category="genomics",
    parameters={
        "gene": "Gene symbol (e.g. 'BRCA1', 'TP53')",
        "tissue": "GTEx tissue name to filter (optional, e.g. 'Liver', 'Brain_Cortex')",
    },
    requires_data=[],
    usage_guide="You want to find genetic variants that regulate gene expression in specific tissues. Use to understand tissue-specific regulation, identify regulatory variants, and connect GWAS signals to gene function.",
)
def eqtl_lookup(gene: str, tissue: str = None, **kwargs) -> dict:
    """Query the GTEx API for significant eQTLs for a gene."""
    try:
        import httpx
    except ImportError:
        return {"error": "httpx required (pip install httpx)", "summary": "httpx required (pip install httpx)"}
    gtex_base = "https://gtexportal.org/api/v2"

    # Step 1: Resolve gene symbol to GENCODE ID
    gene_url = f"{gtex_base}/reference/gene"
    gene_params = {"geneId": gene}

    gene_data, error = request_json(
        "GET",
        gene_url,
        params=gene_params,
        timeout=10,
        retries=2,
    )
    if error:
        return {"error": f"GTEx gene lookup failed: {error}", "summary": f"GTEx gene lookup failed: {error}"}
    genes_list = gene_data.get("data", [])
    if not genes_list:
        return {
            "error": f"Gene '{gene}' not found in GTEx GENCODE v26 reference",
            "suggestion": "Try using the official HGNC gene symbol",
        }

    # Use the first matching gene entry
    gene_info = genes_list[0]
    gencode_id = gene_info.get("gencodeId", "")
    gene_symbol = gene_info.get("geneSymbol", gene)
    description = gene_info.get("description", "")

    if not gencode_id:
        return {"error": f"Could not resolve GENCODE ID for {gene}", "summary": f"Could not resolve GENCODE ID for {gene}"}
    # Step 2: Query significant single-tissue eQTLs
    eqtl_url = f"{gtex_base}/association/singleTissueEqtl"
    eqtl_params = {
        "gencodeId": gencode_id,
        "datasetId": "gtex_v8",
    }
    if tissue:
        eqtl_params["tissueSiteDetailId"] = tissue

    eqtl_data, error = request_json(
        "GET",
        eqtl_url,
        params=eqtl_params,
        timeout=10,
        retries=2,
    )
    if error:
        return {"error": f"GTEx eQTL query failed: {error}", "summary": f"GTEx eQTL query failed: {error}"}
    eqtls_raw = eqtl_data.get("data", [])

    if not eqtls_raw:
        tissue_str = f" in {tissue}" if tissue else ""
        return {
            "summary": f"No significant eQTLs found for {gene_symbol}{tissue_str} in GTEx v8",
            "gene": gene_symbol,
            "gencode_id": gencode_id,
            "eqtls": [],
            "n_eqtls": 0,
        }

    # Parse eQTL results
    eqtls = []
    tissues_found = set()

    for eqtl in eqtls_raw:
        tissue_id = eqtl.get("tissueSiteDetailId", "")
        tissues_found.add(tissue_id)

        eqtls.append({
            "variant_id": eqtl.get("variantId", ""),
            "snp_id": eqtl.get("snpId", ""),
            "tissue": tissue_id,
            "p_value": eqtl.get("pValue"),
            "nes": eqtl.get("nes"),  # Normalized effect size
            "chromosome": eqtl.get("chromosome", ""),
            "pos": eqtl.get("pos"),
            "gene_symbol": eqtl.get("geneSymbol", gene_symbol),
        })

    # Sort by absolute NES (largest effect first)
    eqtls.sort(key=lambda x: abs(x["nes"]) if x["nes"] is not None else 0, reverse=True)

    tissue_str = f" in {tissue}" if tissue else f" across {len(tissues_found)} tissues"
    return {
        "summary": (
            f"GTEx eQTLs for {gene_symbol} ({gencode_id}){tissue_str}: "
            f"{len(eqtls)} significant eQTLs found"
        ),
        "gene": gene_symbol,
        "gencode_id": gencode_id,
        "gene_description": description,
        "n_eqtls": len(eqtls),
        "n_tissues": len(tissues_found),
        "tissues": sorted(tissues_found),
        "eqtls": eqtls[:50],  # Return top 50 by effect size
    }


@registry.register(
    name="genomics.variant_annotate",
    description="Annotate a genetic variant using Ensembl VEP (Variant Effect Predictor)",
    category="genomics",
    parameters={
        "variant": "Variant identifier: rsID (e.g. 'rs1234') or HGVS notation (e.g. '17:g.41245466G>A')",
    },
    requires_data=[],
    usage_guide="You want to understand the functional consequence of a specific genetic variant. Use to get consequence type (missense, synonymous, etc.), impact prediction, amino acid changes, allele frequencies, and clinical significance.",
)
def variant_annotate(variant: str, **kwargs) -> dict:
    """Annotate a variant using the Ensembl VEP REST API."""
    try:
        import httpx
    except ImportError:
        return {"error": "httpx required (pip install httpx)", "summary": "httpx required (pip install httpx)"}
    ensembl_base = "https://rest.ensembl.org"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # Determine if this is an rsID or HGVS notation
    variant_clean = variant.strip()
    if variant_clean.lower().startswith("rs"):
        url = f"{ensembl_base}/vep/human/id/{variant_clean}"
    else:
        url = f"{ensembl_base}/vep/human/hgvs/{variant_clean}"

    resp, error = request(
        "GET",
        url,
        headers=headers,
        timeout=30,
        retries=2,
        raise_for_status=False,
    )
    if error:
        return {"error": f"Ensembl VEP query failed: {error}", "summary": f"Ensembl VEP query failed: {error}"}
    if resp.status_code == 400:
        return {"error": f"Invalid variant format: '{variant}'. Use rsID (e.g. rs1234) or HGVS (e.g. 17:g.41245466G>A)", "summary": f"Invalid variant format: '{variant}'. Use rsID (e.g. rs1234) or HGVS (e.g. 17:g.41245466G>A)"}
    if resp.status_code >= 400:
        return {"error": f"Ensembl VEP query failed: HTTP {resp.status_code}", "summary": f"Ensembl VEP query failed: HTTP {resp.status_code}"}
    try:
        data = resp.json()
    except Exception:
        return {"error": f"Ensembl VEP query failed: invalid JSON response", "summary": f"Ensembl VEP query failed: invalid JSON response"}
    if not data or not isinstance(data, list):
        return {"error": f"No VEP results for variant {variant}", "summary": f"No VEP results for variant {variant}"}
    vep_result = data[0]

    # Extract variant identifiers
    variant_id = vep_result.get("id", variant)
    input_str = vep_result.get("input", variant)
    most_severe = vep_result.get("most_severe_consequence", "")
    allele_string = vep_result.get("allele_string", "")
    strand = vep_result.get("strand")
    assembly = vep_result.get("assembly_name", "")
    seq_region = vep_result.get("seq_region_name", "")
    start = vep_result.get("start")
    end = vep_result.get("end")

    # Extract colocated variants (for allele frequencies, clinical significance)
    colocated = vep_result.get("colocated_variants", [])
    allele_frequencies = {}
    clinical_significance = []
    existing_ids = []

    for cv in colocated:
        cv_id = cv.get("id", "")
        if cv_id:
            existing_ids.append(cv_id)

        # Allele frequencies from different populations
        freqs = cv.get("frequencies", {})
        for allele, pop_freqs in freqs.items():
            for pop, freq in pop_freqs.items():
                key = f"{allele}_{pop}"
                allele_frequencies[key] = freq

        # Minor allele frequency
        maf = cv.get("minor_allele_freq")
        minor_allele = cv.get("minor_allele", "")
        if maf is not None:
            allele_frequencies["minor_allele"] = minor_allele
            allele_frequencies["minor_allele_freq"] = maf

        # Clinical significance
        clin_sig = cv.get("clin_sig", [])
        if clin_sig:
            clinical_significance.extend(clin_sig)

    # Extract transcript consequences
    transcript_consequences = []
    for tc in vep_result.get("transcript_consequences", []):
        consequence_terms = tc.get("consequence_terms", [])
        transcript_consequences.append({
            "gene_id": tc.get("gene_id", ""),
            "gene_symbol": tc.get("gene_symbol", ""),
            "transcript_id": tc.get("transcript_id", ""),
            "biotype": tc.get("biotype", ""),
            "consequence_terms": consequence_terms,
            "impact": tc.get("impact", ""),
            "amino_acids": tc.get("amino_acids", ""),
            "codons": tc.get("codons", ""),
            "protein_position": tc.get("protein_position", ""),
            "sift_prediction": tc.get("sift_prediction", ""),
            "sift_score": tc.get("sift_score"),
            "polyphen_prediction": tc.get("polyphen_prediction", ""),
            "polyphen_score": tc.get("polyphen_score"),
            "canonical": tc.get("canonical", 0) == 1,
        })

    # Sort: canonical transcripts first, then by impact severity
    impact_order = {"HIGH": 0, "MODERATE": 1, "LOW": 2, "MODIFIER": 3}
    transcript_consequences.sort(
        key=lambda x: (
            0 if x["canonical"] else 1,
            impact_order.get(x["impact"], 4),
        )
    )

    # Find the most impactful consequence for the summary
    top_consequence = transcript_consequences[0] if transcript_consequences else {}
    gene_symbol = top_consequence.get("gene_symbol", "")
    impact = top_consequence.get("impact", "")
    aa_change = top_consequence.get("amino_acids", "")
    protein_pos = top_consequence.get("protein_position", "")

    aa_str = ""
    if aa_change and protein_pos:
        aa_str = f", p.{aa_change.replace('/', str(protein_pos))}"

    clin_str = ""
    if clinical_significance:
        unique_clin = list(set(clinical_significance))
        clin_str = f" Clinical: {', '.join(unique_clin)}."

    maf_str = ""
    maf_val = allele_frequencies.get("minor_allele_freq")
    if maf_val is not None:
        maf_str = f" MAF={maf_val:.4f} ({allele_frequencies.get('minor_allele', '')})."

    return {
        "summary": (
            f"VEP annotation for {variant_id}: {most_severe} ({impact}) "
            f"in {gene_symbol}{aa_str}.{clin_str}{maf_str}"
        ),
        "variant_id": variant_id,
        "input": input_str,
        "location": f"{seq_region}:{start}-{end}" if seq_region and start else "",
        "assembly": assembly,
        "allele_string": allele_string,
        "most_severe_consequence": most_severe,
        "existing_ids": existing_ids,
        "allele_frequencies": allele_frequencies,
        "clinical_significance": list(set(clinical_significance)),
        "transcript_consequences": transcript_consequences[:10],  # Top 10
        "n_transcript_consequences": len(transcript_consequences),
    }


@registry.register(
    name="genomics.mendelian_randomization_lookup",
    description="Look up Mendelian randomization and genetic evidence for a gene-disease pair via Open Targets",
    category="genomics",
    parameters={
        "gene": "Gene symbol (e.g. 'PCSK9', 'IL6R')",
        "disease": "Disease name or EFO ID (e.g. 'coronary artery disease' or 'EFO_0001645')",
    },
    requires_data=[],
    usage_guide="You want causal genetic evidence linking a gene to a disease. Use to evaluate target-disease relationships using Mendelian randomization, GWAS colocalisation, and genetic association evidence from Open Targets.",
)
def mendelian_randomization_lookup(gene: str, disease: str, **kwargs) -> dict:
    """Look up MR and genetic evidence from Open Targets Platform GraphQL API."""
    try:
        import httpx
    except ImportError:
        return {"error": "httpx required (pip install httpx)", "summary": "httpx required (pip install httpx)"}
    ot_url = "https://api.platform.opentargets.org/api/v4/graphql"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # Step 1: Resolve gene symbol to Ensembl ID via Open Targets search
    search_query = """
    query searchTarget($queryString: String!) {
        search(queryString: $queryString, entityNames: ["target"], page: {size: 5, index: 0}) {
            hits {
                id
                entity
                name
                description
            }
        }
    }
    """

    search_data, error = request_json(
        "POST",
        ot_url,
        json={"query": search_query, "variables": {"queryString": gene}},
        headers=headers,
        timeout=10,
        retries=2,
    )
    if error:
        return {"error": f"Open Targets search failed: {error}", "summary": f"Open Targets search failed: {error}"}
    hits = search_data.get("data", {}).get("search", {}).get("hits", [])
    target_hits = [h for h in hits if h.get("entity") == "target"]

    if not target_hits:
        return {"error": f"Gene '{gene}' not found in Open Targets", "summary": f"Gene '{gene}' not found in Open Targets"}
    # Match by gene symbol (case-insensitive)
    ensembl_id = None
    target_name = ""
    for hit in target_hits:
        if hit.get("name", "").upper() == gene.upper():
            ensembl_id = hit["id"]
            target_name = hit.get("name", "")
            break
    if not ensembl_id:
        ensembl_id = target_hits[0]["id"]
        target_name = target_hits[0].get("name", "")

    # Step 2: Resolve disease to EFO ID (if not already an EFO ID)
    if disease.upper().startswith("EFO_") or disease.upper().startswith("MONDO_") or disease.upper().startswith("HP_"):
        efo_id = disease
        disease_name = disease
    else:
        disease_search_query = """
        query searchDisease($queryString: String!) {
            search(queryString: $queryString, entityNames: ["disease"], page: {size: 5, index: 0}) {
                hits {
                    id
                    entity
                    name
                    description
                }
            }
        }
        """

        disease_data, error = request_json(
            "POST",
            ot_url,
            json={"query": disease_search_query, "variables": {"queryString": disease}},
            headers=headers,
            timeout=10,
            retries=2,
        )
        if error:
            return {"error": f"Open Targets disease search failed: {error}", "summary": f"Open Targets disease search failed: {error}"}
        disease_hits = disease_data.get("data", {}).get("search", {}).get("hits", [])
        disease_hits = [h for h in disease_hits if h.get("entity") == "disease"]

        if not disease_hits:
            return {"error": f"Disease '{disease}' not found in Open Targets", "summary": f"Disease '{disease}' not found in Open Targets"}
        efo_id = disease_hits[0]["id"]
        disease_name = disease_hits[0].get("name", disease)

    # Step 3: Query genetic evidence (evidences is on Target, not top-level)
    # Genetic datasources: gwas_credible_sets (L2G scores), eva, gene_burden,
    # gene2phenotype, genomics_england, uniprot_literature
    evidence_query = """
    query targetDiseaseEvidence($ensemblId: String!, $efoId: String!) {
        target(ensemblId: $ensemblId) {
            id
            approvedSymbol
            approvedName
            associatedDiseases(BFilter: $efoId, page: {size: 1, index: 0}) {
                rows {
                    score
                    disease { id name }
                    datasourceScores {
                        id
                        score
                    }
                }
            }
            evidences(
                efoIds: [$efoId]
                datasourceIds: [
                    "gwas_credible_sets", "gene_burden", "eva",
                    "gene2phenotype", "genomics_england", "uniprot_literature"
                ]
                size: 50
            ) {
                count
                rows {
                    datasourceId
                    datatypeId
                    score
                    resourceScore
                    studyId
                    beta
                    oddsRatio
                    confidence
                    studySampleSize
                    publicationYear
                    variantRsId
                    credibleSet {
                        studyLocusId
                        study { id projectId studyType }
                        variant { id rsIds }
                        pValueMantissa
                        pValueExponent
                        beta
                        finemappingMethod
                    }
                }
            }
        }
        disease(efoId: $efoId) {
            id
            name
            description
        }
    }
    """

    result_data, error = request_json(
        "POST",
        ot_url,
        json={
            "query": evidence_query,
            "variables": {"ensemblId": ensembl_id, "efoId": efo_id},
        },
        headers=headers,
        timeout=15,
        retries=2,
    )
    if error:
        return {"error": f"Open Targets evidence query failed: {error}", "summary": f"Open Targets evidence query failed: {error}"}
    if result_data.get("errors"):
        error_msgs = [e.get("message", "") for e in result_data["errors"]]
        return {"error": f"Open Targets GraphQL errors: {'; '.join(error_msgs)}", "summary": f"Open Targets GraphQL errors: {'; '.join(error_msgs)}"}
    data = result_data.get("data", {})

    # Parse target and disease info
    target_info = data.get("target") or {}
    disease_info = data.get("disease") or {}
    approved_symbol = target_info.get("approvedSymbol", gene)
    approved_name = target_info.get("approvedName", "")
    resolved_disease = disease_info.get("name", disease_name if disease_name else disease)

    # Parse overall association score
    assoc_rows = target_info.get("associatedDiseases", {}).get("rows", [])
    overall_score = assoc_rows[0].get("score") if assoc_rows else None
    datasource_scores = {}
    if assoc_rows:
        for ds in assoc_rows[0].get("datasourceScores", []):
            datasource_scores[ds["id"]] = ds["score"]

    # Parse evidence rows
    evidences_obj = target_info.get("evidences") or {}
    evidence_count = evidences_obj.get("count", 0)
    evidence_rows = evidences_obj.get("rows", [])

    # Categorize evidence by datasource
    gwas_evidence = []
    other_genetic_evidence = []

    for row in evidence_rows:
        datasource = row.get("datasourceId", "")

        # Extract variant info from credibleSet if available
        credible_set = row.get("credibleSet") or {}
        variant_info = credible_set.get("variant") or {}
        study_info = credible_set.get("study") or {}
        rs_ids = variant_info.get("rsIds", [])
        variant_rsid = rs_ids[0] if rs_ids else (row.get("variantRsId") or "")

        # Compute p-value from mantissa/exponent
        p_mantissa = credible_set.get("pValueMantissa")
        p_exponent = credible_set.get("pValueExponent")
        p_value = None
        if p_mantissa is not None and p_exponent is not None:
            try:
                p_value = float(p_mantissa) * (10 ** int(p_exponent))
            except (ValueError, TypeError):
                pass

        evidence_item = {
            "datasource": datasource,
            "datatype": row.get("datatypeId", ""),
            "score": row.get("score"),
            "resource_score": row.get("resourceScore"),
            "variant_id": variant_info.get("id", ""),
            "variant_rsid": variant_rsid,
            "study_id": study_info.get("id") or row.get("studyId", ""),
            "study_type": study_info.get("studyType", ""),
            "p_value": p_value,
            "beta": credible_set.get("beta") or row.get("beta"),
            "odds_ratio": row.get("oddsRatio"),
            "finemapping_method": credible_set.get("finemappingMethod", ""),
            "publication_year": row.get("publicationYear"),
        }

        if datasource == "gwas_credible_sets":
            gwas_evidence.append(evidence_item)
        else:
            other_genetic_evidence.append(evidence_item)

    # Compute summary statistics
    all_evidence = gwas_evidence + other_genetic_evidence
    max_score = max((e["score"] for e in all_evidence if e["score"] is not None), default=None)
    n_variants = len(set(e["variant_rsid"] for e in all_evidence if e["variant_rsid"]))
    n_studies = len(set(e["study_id"] for e in all_evidence if e["study_id"]))

    # Build summary
    parts = []
    if gwas_evidence:
        parts.append(f"{len(gwas_evidence)} GWAS credible set(s)")
    if other_genetic_evidence:
        parts.append(f"{len(other_genetic_evidence)} other genetic evidence(s)")
    if not parts:
        parts.append("no genetic evidence found")

    score_str = f" Overall association: {overall_score:.3f}." if overall_score is not None else ""
    max_str = f" Max L2G score: {max_score:.3f}." if max_score is not None else ""
    variant_str = f" {n_variants} unique variant(s) across {n_studies} study(ies)." if n_variants > 0 else ""

    return {
        "summary": (
            f"Genetic evidence for {approved_symbol} -> {resolved_disease}: "
            f"{', '.join(parts)}.{score_str}{max_str}{variant_str}"
        ),
        "gene": approved_symbol,
        "gene_name": approved_name,
        "ensembl_id": ensembl_id,
        "disease": resolved_disease,
        "disease_id": efo_id,
        "overall_association_score": overall_score,
        "datasource_scores": datasource_scores,
        "total_evidence_count": evidence_count,
        "gwas_credible_sets": gwas_evidence,
        "other_genetic_evidence": other_genetic_evidence,
        "max_l2g_score": max_score,
        "n_unique_variants": n_variants,
        "n_studies": n_studies,
    }


@registry.register(
    name="genomics.coloc",
    description="Look up GWAS-eQTL/pQTL colocalization evidence for a gene via Open Targets Platform",
    category="genomics",
    parameters={
        "gene": "Gene symbol (e.g. 'PCSK9', 'IL6R')",
        "study_id": "Specific GWAS study ID to filter (optional)",
    },
    requires_data=[],
    usage_guide="You want to assess whether a GWAS signal and an eQTL/pQTL signal share the same "
                "causal variant at a locus — the gold standard for connecting genetic associations "
                "to gene function. High H4 posterior probability (>0.8) indicates strong colocalization. "
                "Use for target validation and causal gene assignment at GWAS loci.",
)
def coloc(gene: str, study_id: str = None, **kwargs) -> dict:
    """Look up colocalization evidence from Open Targets Platform GraphQL API.

    Queries the Open Targets credibleSets and colocalisations data for a gene
    target, returning GWAS-QTL colocalization information including H4 posterior
    probabilities (evidence of shared causal variant), study details, and tissues.
    """
    try:
        import httpx
    except ImportError:
        return {"error": "httpx required (pip install httpx)", "summary": "httpx required (pip install httpx)"}
    ot_url = "https://api.platform.opentargets.org/api/v4/graphql"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    def _gene_symbol_candidates(input_gene: str) -> list[str]:
        alias_map = {
            "GBA1": "GBA",
            "PARK2": "PRKN",
        }
        token = (input_gene or "").strip()
        if not token:
            return []
        candidates = [token]
        mapped = alias_map.get(token.upper())
        if mapped:
            candidates.append(mapped)

        # Stable de-dup preserving order (case-insensitive).
        deduped = []
        seen = set()
        for c in candidates:
            k = c.upper()
            if k in seen:
                continue
            seen.add(k)
            deduped.append(c)
        return deduped

    def _resolve_ensembl_id(symbol: str) -> tuple[str | None, str | None]:
        ens_resp, resolve_error = request(
            "GET",
            f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{symbol}",
            params={"content-type": "application/json"},
            timeout=10,
            retries=2,
            headers={"Content-Type": "application/json"},
            raise_for_status=False,
        )
        if resolve_error:
            return None, f"Failed to resolve {symbol} to Ensembl ID: {resolve_error}"
        if ens_resp.status_code != 200:
            return None, f"Gene {symbol} not found in Ensembl (human)"
        try:
            ens_data = ens_resp.json()
        except Exception:
            return None, f"Failed to parse Ensembl response for {symbol}"
        ensembl = ens_data.get("id", "")
        if not ensembl:
            return None, f"Gene {symbol} not found in Ensembl (human)"
        return ensembl, None

    # Step 2: Query Open Targets for credible sets with colocalization data.
    # We keep a full query and a lower-complexity fallback query because some
    # genes can hit Open Targets GraphQL complexity limits.
    query_full = """
    query geneColoc($ensemblId: String!, $size: Int!, $colocSize: Int!) {
        target(ensemblId: $ensemblId) {
            id
            approvedSymbol
            approvedName
            credibleSets(page: {index: 0, size: $size}) {
                count
                rows {
                    studyLocusId
                    studyId
                    studyType
                    study {
                        id
                        studyType
                        traitFromSource
                        diseases {
                            id
                            name
                        }
                        nSamples
                    }
                    variant {
                        id
                        rsIds
                        chromosome
                        position
                    }
                    pValueMantissa
                    pValueExponent
                    beta
                    colocalisation(page: {index: 0, size: $colocSize}) {
                        count
                        rows {
                            h4
                            h3
                            clpp
                            colocalisationMethod
                            rightStudyType
                            betaRatioSignAverage
                            numberColocalisingVariants
                            otherStudyLocus {
                                studyLocusId
                                studyId
                                studyType
                                qtlGeneId
                                study {
                                    id
                                    traitFromSource
                                    condition
                                    biosample {
                                        biosampleId
                                        biosampleName
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    """

    query_lean = """
    query geneColocLean($ensemblId: String!, $size: Int!, $colocSize: Int!) {
        target(ensemblId: $ensemblId) {
            id
            approvedSymbol
            approvedName
            credibleSets(page: {index: 0, size: $size}) {
                count
                rows {
                    studyLocusId
                    studyId
                    studyType
                    study {
                        id
                        studyType
                        traitFromSource
                        diseases {
                            id
                            name
                        }
                    }
                    colocalisation(page: {index: 0, size: $colocSize}) {
                        count
                        rows {
                            h4
                            h3
                            clpp
                            colocalisationMethod
                            rightStudyType
                            otherStudyLocus {
                                studyLocusId
                                studyId
                                studyType
                                qtlGeneId
                                study {
                                    id
                                    traitFromSource
                                    condition
                                    biosample {
                                        biosampleId
                                        biosampleName
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    """

    def _query_target_coloc(ensembl: str) -> tuple[dict | None, str | None]:
        def _run_query(query_text: str, page_attempts: tuple[tuple[int, int], ...]) -> tuple[dict | None, str | None]:
            last_err = None
            for size, coloc_size in page_attempts:
                resp, query_error = request(
                    "POST",
                    ot_url,
                    json={
                        "query": query_text,
                        "variables": {
                            "ensemblId": ensembl,
                            "size": size,
                            "colocSize": coloc_size,
                        },
                    },
                    headers=headers,
                    timeout=15,
                    retries=2,
                    raise_for_status=False,
                )
                if query_error:
                    last_err = f"Open Targets API error: {query_error}"
                    continue
                if resp.status_code != 200:
                    last_err = f"Open Targets API returned HTTP {resp.status_code}"
                    # Retry with smaller page sizes for likely complexity-related rejections.
                    if resp.status_code in {400, 413, 422, 429, 500, 502, 503, 504}:
                        continue
                    break

                try:
                    payload = resp.json()
                except Exception:
                    last_err = "Open Targets API returned invalid JSON"
                    continue

                gql_errors = payload.get("errors") or []
                if gql_errors:
                    msgs = "; ".join(e.get("message", "") for e in gql_errors)
                    last_err = f"Open Targets GraphQL errors: {msgs}"
                    lower = msgs.lower()
                    if any(tok in lower for tok in ("complex", "depth", "cost", "too many", "timeout")):
                        continue
                    break
                return payload, None
            return None, (last_err or "Open Targets colocalization query failed")

        # Try richer query first, then lower-complexity fallback.
        attempts = (
            ("full", query_full, ((60, 40), (30, 20), (15, 10))),
            ("lean", query_lean, ((40, 20), (20, 10), (10, 5))),
        )
        errors = []
        for label, query_text, page_attempts in attempts:
            payload, err = _run_query(query_text, page_attempts)
            if payload is not None:
                return payload, None
            if err:
                errors.append(f"{label} query: {err}")
        if errors:
            return None, "; ".join(errors)
        return None, "Open Targets colocalization query failed"

    # Try primary symbol first, then common aliases (e.g., GBA1 -> GBA) if needed.
    gene_candidates = _gene_symbol_candidates(gene)
    ensembl_id = None
    result_data = None
    target_data = None
    candidate_errors = []
    query_failures = []
    resolved_candidates = []

    for gene_candidate in gene_candidates:
        ensembl_candidate, resolve_error = _resolve_ensembl_id(gene_candidate)
        if resolve_error:
            candidate_errors.append(resolve_error)
            continue
        resolved_candidates.append((gene_candidate, ensembl_candidate))

        payload, query_error = _query_target_coloc(ensembl_candidate)
        if query_error:
            candidate_errors.append(f"{gene_candidate}: {query_error}")
            query_failures.append((gene_candidate, ensembl_candidate, query_error))
            continue

        target_candidate = (payload or {}).get("data", {}).get("target")
        if not target_candidate:
            candidate_errors.append(
                f"{gene_candidate}: Open Targets has no entry for {ensembl_candidate}"
            )
            query_failures.append(
                (gene_candidate, ensembl_candidate, f"Open Targets has no entry for {ensembl_candidate}")
            )
            continue

        ensembl_id = ensembl_candidate
        result_data = payload
        target_data = target_candidate
        break

    if not target_data:
        last_error = candidate_errors[-1] if candidate_errors else "Open Targets colocalization query failed"
        if candidate_errors and all("not found in Ensembl" in e for e in candidate_errors):
            return {
                "error": last_error,
                "summary": f"Gene symbol {gene} could not be resolved to an Ensembl ID",
            }
        # Resolved gene(s) but Open Targets could not return colocalization payload.
        # Return a non-fatal unavailable result so workflows can continue.
        if resolved_candidates:
            chosen_symbol, chosen_ensembl = resolved_candidates[0]
            warning = query_failures[0][2] if query_failures else last_error
            return {
                "summary": (
                    f"Colocalization for {chosen_symbol}: unavailable from Open Targets "
                    f"(query failed). Try genomics.eqtl_lookup for orthogonal evidence."
                ),
                "gene": chosen_symbol,
                "ensembl_id": chosen_ensembl,
                "total_gwas_loci": 0,
                "n_colocalizations": 0,
                "n_strong_coloc": 0,
                "n_moderate_coloc": 0,
                "n_tissues": 0,
                "n_studies": 0,
                "tissues": [],
                "colocalizations": [],
                "data_unavailable": True,
                "warning": warning,
            }
        if "GraphQL errors" in last_error:
            return {
                "error": last_error,
                "summary": f"GraphQL query errors for {gene} colocalization",
            }
        return {
            "error": last_error,
            "summary": f"Open Targets colocalization query failed for {gene}",
        }

    approved_symbol = target_data.get("approvedSymbol", gene)
    # Backward-compatibility: some mocked test fixtures still use legacy field names.
    credible_sets = target_data.get("credibleSets") or target_data.get("gwasCredibleSets") or {}
    rows = credible_sets.get("rows", []) if isinstance(credible_sets, dict) else []

    # Keep only GWAS credible sets for this tool.
    def _is_gwas(row: dict) -> bool:
        st = (row.get("studyType") or (row.get("study") or {}).get("studyType") or "")
        return str(st).lower() == "gwas"

    if target_data.get("gwasCredibleSets") is not None:
        gwas_rows = rows
        total_loci = credible_sets.get("count", len(rows))
    else:
        gwas_rows = [row for row in rows if _is_gwas(row)]
        total_loci = len(gwas_rows)

    # Parse colocalization results
    coloc_results = []
    tissues_seen = set()
    studies_seen = set()

    for row in gwas_rows:
        study = row.get("study") or {}
        gwas_study_id = row.get("studyId") or study.get("id", "")

        # Filter by study_id if provided
        if study_id and gwas_study_id != study_id:
            continue

        variant = row.get("variant") or {}
        rs_ids = variant.get("rsIds", [])
        lead_rsid = rs_ids[0] if rs_ids else ""

        # Compute p-value
        p_mantissa = row.get("pValueMantissa")
        p_exponent = row.get("pValueExponent")
        p_value = None
        if p_mantissa is not None and p_exponent is not None:
            try:
                p_value = float(p_mantissa) * (10 ** int(p_exponent))
            except (ValueError, TypeError):
                pass

        # Extract L2G score for this gene
        l2g_score = None
        l2g_preds_raw = row.get("l2GPredictions") or []
        if isinstance(l2g_preds_raw, dict):
            l2g_preds = l2g_preds_raw.get("rows") or []
        else:
            l2g_preds = l2g_preds_raw
        for pred in l2g_preds:
            pred_target = pred.get("target") or {}
            if pred_target.get("id") == ensembl_id:
                l2g_score = pred.get("score")
                if l2g_score is None:
                    l2g_score = pred.get("yProbaModel")
                break

        trait = study.get("traitFromSource", "")
        diseases = study.get("diseases") or []
        disease_names = [d.get("name", "") for d in diseases if d.get("name")]

        # Parse current Open Targets schema: colocalisation.rows
        coloc_obj = row.get("colocalisation") or {}
        qtl_colocs = coloc_obj.get("rows", []) if isinstance(coloc_obj, dict) else []
        for qtl in qtl_colocs:
            h4 = qtl.get("h4")
            h3 = qtl.get("h3")
            right_study_type = str(qtl.get("rightStudyType") or "").lower()
            if right_study_type and "qtl" not in right_study_type:
                continue

            other = qtl.get("otherStudyLocus") or {}
            other_study = other.get("study") or {}
            biosample = other_study.get("biosample") or {}

            tissue_name = (
                biosample.get("biosampleName")
                or other_study.get("condition")
                or other_study.get("traitFromSource")
                or ""
            )
            tissue_id = biosample.get("biosampleId", "")
            qtl_study = other.get("studyId") or other_study.get("id", "")
            phenotype = other.get("qtlGeneId", "")

            log2_h4_h3 = None
            if h4 is not None and h3 not in (None, 0):
                try:
                    if float(h4) > 0 and float(h3) > 0:
                        log2_h4_h3 = math.log2(float(h4) / float(h3))
                except (TypeError, ValueError, ZeroDivisionError):
                    log2_h4_h3 = None

            if tissue_name:
                tissues_seen.add(tissue_name)
            studies_seen.add(gwas_study_id)

            coloc_results.append({
                "gwas_study_id": gwas_study_id,
                "trait": trait,
                "diseases": disease_names,
                "lead_variant": variant.get("id", ""),
                "lead_rsid": lead_rsid,
                "p_value": p_value,
                "l2g_score": round(l2g_score, 4) if l2g_score is not None else None,
                "qtl_study_id": qtl_study,
                "phenotype_id": phenotype,
                "tissue": tissue_name,
                "tissue_id": tissue_id,
                "h4": round(h4, 4) if h4 is not None else None,
                "h3": round(h3, 4) if h3 is not None else None,
                "log2_h4_h3": round(log2_h4_h3, 4) if log2_h4_h3 is not None else None,
                "colocalisation_method": qtl.get("colocalisationMethod"),
                "right_study_type": qtl.get("rightStudyType"),
                "clpp": round(qtl.get("clpp"), 4) if qtl.get("clpp") is not None else None,
            })

        # Backward compatibility with legacy schema field name used in old fixtures.
        legacy_qtls = row.get("colocalisationsQtl") or []
        for qtl in legacy_qtls:
            h4 = qtl.get("h4")
            tissue_info = qtl.get("tissue") or {}
            tissue_name = tissue_info.get("name", "")
            tissue_id = tissue_info.get("id", "")
            qtl_study = qtl.get("qtlStudyId", "")
            phenotype = qtl.get("phenotypeId", "")

            if tissue_name:
                tissues_seen.add(tissue_name)
            studies_seen.add(gwas_study_id)

            coloc_results.append({
                "gwas_study_id": gwas_study_id,
                "trait": trait,
                "diseases": disease_names,
                "lead_variant": variant.get("id", ""),
                "lead_rsid": lead_rsid,
                "p_value": p_value,
                "l2g_score": round(l2g_score, 4) if l2g_score is not None else None,
                "qtl_study_id": qtl_study,
                "phenotype_id": phenotype,
                "tissue": tissue_name,
                "tissue_id": tissue_id,
                "h4": round(h4, 4) if h4 is not None else None,
                "h3": round(qtl.get("h3", 0), 4) if qtl.get("h3") is not None else None,
                "log2_h4_h3": round(qtl.get("log2h4h3", 0), 4) if qtl.get("log2h4h3") is not None else None,
                "colocalisation_method": None,
                "right_study_type": None,
                "clpp": None,
            })

    # Sort by H4 (strongest colocalization first)
    coloc_results.sort(key=lambda x: x["h4"] if x["h4"] is not None else 0, reverse=True)

    n_strong = sum(1 for c in coloc_results if c["h4"] is not None and c["h4"] > 0.8)
    n_moderate = sum(1 for c in coloc_results if c["h4"] is not None and 0.5 < c["h4"] <= 0.8)

    # Build summary
    study_filter_str = f" (study {study_id})" if study_id else ""
    if coloc_results:
        top_coloc = coloc_results[0]
        top_str = (
            f"Strongest: {top_coloc['trait']} / {top_coloc['tissue']} "
            f"(H4={top_coloc['h4']:.3f})" if top_coloc['h4'] is not None
            else f"Strongest: {top_coloc['trait']} / {top_coloc['tissue']}"
        )
        summary = (
            f"Colocalization for {approved_symbol}{study_filter_str}: "
            f"{len(coloc_results)} GWAS-QTL pairs across {len(tissues_seen)} tissues, "
            f"{len(studies_seen)} GWAS studies. "
            f"{n_strong} strong (H4>0.8), {n_moderate} moderate (0.5<H4<=0.8). "
            f"{top_str}"
        )
    else:
        summary = (
            f"Colocalization for {approved_symbol}{study_filter_str}: "
            f"no QTL colocalization data found ({total_loci} GWAS loci scanned)"
        )

    return {
        "summary": summary,
        "gene": approved_symbol,
        "ensembl_id": ensembl_id,
        "total_gwas_loci": total_loci,
        "n_colocalizations": len(coloc_results),
        "n_strong_coloc": n_strong,
        "n_moderate_coloc": n_moderate,
        "n_tissues": len(tissues_seen),
        "n_studies": len(studies_seen),
        "tissues": sorted(tissues_seen),
        "colocalizations": coloc_results[:50],  # Cap at 50
    }


# ---------------------------------------------------------------------------
# Variant classification (code-gen tool)
# ---------------------------------------------------------------------------

VARIANT_CLASSIFY_PROMPT = """You are an expert bioinformatics data analyst classifying and analyzing genomic variants.

{namespace_description}

## Available Data
{data_files_description}

## DATA LOADING
- **ZIP files**: Extract first with `zipfile.ZipFile(path, "r").extractall("/tmp/extracted")`
- **Excel .xls**: `pd.read_excel(path, engine='xlrd')`
- **Excel .xlsx**: `pd.read_excel(path, engine='openpyxl')`
- **VCF**: parse with pandas or cyvcf2; standard columns: CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO

Always check `pd.ExcelFile(path).sheet_names` and try both `skiprows=0` and `skiprows=1`
(clinical variant files often have multi-row headers).

## DATA EXPLORATION (DO THIS FIRST)
```python
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("Head:\\n", df.head(3))
print("Dtypes:\\n", df.dtypes)
```

## VARIANT ANALYSIS

### VAF (Variant Allele Frequency) Column Discovery
VAF columns have many naming conventions. Search broadly:
```python
vaf_terms = ['variant allele freq', 'allele freq', 'allele frac', 'vaf',
             'tumor_f', 't_alt_freq', 'af', 'allelic fraction']
vaf_col = None
for col in df.columns:
    if any(term in str(col).lower() for term in vaf_terms):
        vaf_col = col
        break
# Fallback: find float column with values in [0, 1]
if vaf_col is None:
    for col in df.columns:
        if df[col].dtype in [float, np.float64]:
            vals = df[col].dropna()
            if len(vals) > 0 and vals.min() >= 0 and vals.max() <= 1:
                vaf_col = col
                break
```

### Effect/Consequence Annotation
Variant files often have multiple annotation columns at different granularity levels.
Always use the most granular (e.g., Sequence Ontology terms over broad "Effect" categories).
```python
effect_cols = [c for c in df.columns if any(k in str(c).lower()
               for k in ['effect', 'consequence', 'ontology', 'classification'])]
for col in effect_cols:
    print(f"  {{col}}: {{sorted(df[col].dropna().unique())}}")
```

### Coding vs Noncoding Classification
**Coding** (affect protein sequence): synonymous_variant, missense_variant, frameshift_variant,
stop_gained, stop_lost, start_lost, inframe_insertion, inframe_deletion,
splice_donor_variant, splice_acceptor_variant.

**Noncoding**: intron_variant, intergenic_variant, 3_prime_UTR_variant, 5_prime_UTR_variant,
splice_region_variant, upstream_gene_variant, downstream_gene_variant.

### Ts/Tv Ratio (Transition/Transversion)
Only count SNPs using REF and the first ALT allele (`ALT.split(',')[0]`) so multi-allelic
records with SNP first-alleles are not discarded.
For raw bacterial VCFs, apply a high-confidence depth filter using the sample FORMAT depth
(`FORMAT` field DP, not INFO-level DP): keep SNPs with FORMAT/DP >= 12 before final Ts/Tv
reporting unless the question explicitly requests unfiltered raw calls.
```python
transitions = {{'AG', 'GA', 'CT', 'TC'}}
transversions = {{'AC', 'CA', 'AT', 'TA', 'GC', 'CG', 'GT', 'TG'}}
ts = tv = 0
for _, row in df.iterrows():
    ref = str(row['REF']).upper()
    alt = str(row['ALT']).split(',')[0].upper()
    if len(ref) == 1 and len(alt) == 1:
        pair = ref + alt
        if pair in transitions: ts += 1
        elif pair in transversions: tv += 1
tstv = ts / tv if tv > 0 else 0
```

### Carrier/Cohort Analysis
When analyzing multiple samples:
1. Explore directory to find all variant files and any metadata/annotation files
2. Read metadata to identify sample groups (carriers vs controls, etc.)
3. Match variant files to samples by ID patterns in filenames
4. Filter variants per sample (e.g., non-reference zygosity, VAF thresholds)

## Rules
1. Do NOT import libraries already in the namespace (pd, np, plt, sns, scipy_stats, etc.)
2. Save plots to OUTPUT_DIR: `plt.savefig(OUTPUT_DIR / "filename.png", dpi=150, bbox_inches="tight")`; `plt.close()`
3. Assign result: `result = {{"summary": "...", "answer": "PRECISE_ANSWER"}}`
4. Use print() for intermediate output to verify correctness.
5. If 0 results from a filter: print the column values and debug — do not return "N/A".

Write ONLY the Python code. No explanation, no markdown fences.
"""


@registry.register(
    name="genomics.gnomad_lookup",
    description="Query gnomAD for population variant frequencies and annotations for a gene or specific variant",
    category="genomics",
    parameters={
        "gene": "Gene symbol (e.g. 'BRCA1', 'TP53')",
        "variant": "Specific variant ID to look up (optional, e.g. '1-55516888-G-A')",
        "dataset": "gnomAD dataset version (default 'gnomad_r4')",
    },
    requires_data=[],
    usage_guide="You want population allele frequencies across ancestries for variants in a gene from gnomAD. Use for variant interpretation, assessing pathogenicity (rare vs common), and understanding population-specific frequencies.",
)
def gnomad_lookup(gene: str, variant: str = None, dataset: str = "gnomad_r4", **kwargs) -> dict:
    """Query the gnomAD GraphQL API for variant frequencies and annotations."""
    from ct.tools.http_client import request_json

    gnomad_url = "https://gnomad.broadinstitute.org/api"
    headers = {"Content-Type": "application/json"}

    gene = (gene or "").strip().upper()
    if not gene:
        return {"error": "Gene symbol is required", "summary": "gnomAD lookup requires a gene symbol"}

    if variant:
        # Query a specific variant
        variant_query = """
        query GnomadVariant($variantId: String!, $datasetId: DatasetId!) {
            variant(variantId: $variantId, dataset: $datasetId) {
                variant_id
                chrom
                pos
                ref
                alt
                rsids
                exome {
                    ac
                    an
                    af
                    homozygote_count
                    populations {
                        id
                        ac
                        an
                        af
                        homozygote_count
                    }
                }
                genome {
                    ac
                    an
                    af
                    homozygote_count
                    populations {
                        id
                        ac
                        an
                        af
                        homozygote_count
                    }
                }
                transcript_consequences {
                    gene_symbol
                    transcript_id
                    consequence
                    hgvsc
                    hgvsp
                    lof
                    lof_filter
                    is_canonical
                }
            }
        }
        """
        data, error = request_json(
            "POST", gnomad_url,
            json={"query": variant_query, "variables": {"variantId": variant, "datasetId": dataset}},
            headers=headers, timeout=20, retries=2,
        )
        if error:
            return {"error": f"gnomAD variant query failed: {error}", "summary": f"gnomAD query failed: {error}"}

        var_data = (data or {}).get("data", {}).get("variant")
        if not var_data:
            return {"summary": f"Variant {variant} not found in gnomAD ({dataset})", "variant": variant, "found": False}

        # Combine exome + genome
        exome = var_data.get("exome") or {}
        genome = var_data.get("genome") or {}
        total_ac = (exome.get("ac") or 0) + (genome.get("ac") or 0)
        total_an = (exome.get("an") or 0) + (genome.get("an") or 0)
        total_af = total_ac / total_an if total_an > 0 else 0
        total_hom = (exome.get("homozygote_count") or 0) + (genome.get("homozygote_count") or 0)

        # Population frequencies from genome (larger dataset)
        pop_freqs = []
        for pop in (genome.get("populations") or exome.get("populations") or []):
            if pop.get("an", 0) > 0:
                pop_freqs.append({
                    "population": pop["id"],
                    "ac": pop.get("ac", 0),
                    "an": pop.get("an", 0),
                    "af": round(pop.get("af", 0), 8),
                    "homozygote_count": pop.get("homozygote_count", 0),
                })
        pop_freqs.sort(key=lambda x: x["af"], reverse=True)

        # Consequences
        consequences = []
        for tc in (var_data.get("transcript_consequences") or []):
            consequences.append({
                "gene": tc.get("gene_symbol", ""),
                "transcript": tc.get("transcript_id", ""),
                "consequence": tc.get("consequence", ""),
                "hgvsc": tc.get("hgvsc", ""),
                "hgvsp": tc.get("hgvsp", ""),
                "lof": tc.get("lof", ""),
                "canonical": tc.get("is_canonical", False),
            })

        return {
            "summary": (
                f"gnomAD {variant}: AF={total_af:.6g} ({total_ac}/{total_an}), "
                f"{total_hom} homozygotes. rsIDs: {', '.join(var_data.get('rsids', []))}"
            ),
            "variant_id": var_data.get("variant_id", variant),
            "rsids": var_data.get("rsids", []),
            "total_af": round(total_af, 8),
            "total_ac": total_ac,
            "total_an": total_an,
            "homozygote_count": total_hom,
            "population_frequencies": pop_freqs,
            "consequences": consequences[:10],
            "found": True,
        }

    else:
        # Query gene-level variant summary
        gene_query = """
        query GnomadGene($geneSymbol: String!, $datasetId: DatasetId!) {
            gene(gene_symbol: $geneSymbol, reference_genome: GRCh38) {
                gene_id
                symbol
                name
                chrom
                start
                stop
                variants(dataset: $datasetId) {
                    variant_id
                    rsids
                    pos
                    ref
                    alt
                    exome {
                        ac
                        an
                        af
                    }
                    genome {
                        ac
                        an
                        af
                    }
                    transcript_consequences {
                        consequence
                        is_canonical
                    }
                }
            }
        }
        """
        data, error = request_json(
            "POST", gnomad_url,
            json={"query": gene_query, "variables": {"geneSymbol": gene, "datasetId": dataset}},
            headers=headers, timeout=30, retries=2,
        )
        if error:
            return {"error": f"gnomAD gene query failed: {error}", "summary": f"gnomAD query failed for {gene}: {error}"}

        gene_data = (data or {}).get("data", {}).get("gene")
        if not gene_data:
            return {"summary": f"Gene {gene} not found in gnomAD", "gene": gene, "found": False}

        variants = gene_data.get("variants") or []

        # Categorize by consequence
        consequence_counts = {}
        for v in variants:
            for tc in (v.get("transcript_consequences") or []):
                if tc.get("is_canonical"):
                    csq = tc.get("consequence", "unknown")
                    consequence_counts[csq] = consequence_counts.get(csq, 0) + 1

        # Find most common/rare variants
        variant_summaries = []
        for v in variants[:50]:
            exome = v.get("exome") or {}
            genome = v.get("genome") or {}
            af = (genome.get("af") or exome.get("af") or 0)
            variant_summaries.append({
                "variant_id": v.get("variant_id", ""),
                "rsids": v.get("rsids", []),
                "af": round(af, 8),
            })
        variant_summaries.sort(key=lambda x: x["af"], reverse=True)

        return {
            "summary": (
                f"gnomAD {gene}: {len(variants)} variants in {dataset}. "
                f"Consequences: {', '.join(f'{k}={v}' for k, v in sorted(consequence_counts.items(), key=lambda x: -x[1])[:5])}"
            ),
            "gene": gene_data.get("symbol", gene),
            "gene_id": gene_data.get("gene_id", ""),
            "gene_name": gene_data.get("name", ""),
            "location": f"chr{gene_data.get('chrom', '')}:{gene_data.get('start', '')}-{gene_data.get('stop', '')}",
            "n_variants": len(variants),
            "consequence_counts": consequence_counts,
            "top_variants": variant_summaries[:20],
            "found": True,
        }


@registry.register(
    name="genomics.cosmic_lookup",
    description="Look up somatic mutation data from COSMIC for a gene, optionally filtered by cancer type",
    category="genomics",
    parameters={
        "gene": "Gene symbol (e.g. 'TP53', 'KRAS', 'BRAF')",
        "cancer_type": "Cancer type filter (optional, e.g. 'lung', 'breast')",
    },
    requires_data=[],
    usage_guide="You want somatic mutation data for a gene in cancer — mutation frequencies, hotspots, and mutation spectrum. Use for oncology target validation, understanding mutation patterns in cancer, and identifying actionable mutations.",
)
def cosmic_lookup(gene: str, cancer_type: str = None, **kwargs) -> dict:
    """Query NLM Clinical Tables COSMIC mirror for somatic mutations (no auth required)."""
    from ct.tools.http_client import request_json

    gene = (gene or "").strip().upper()
    if not gene:
        return {"error": "Gene symbol is required", "summary": "COSMIC lookup requires a gene symbol"}

    # NLM Clinical Tables API — free, no authentication
    base_url = "https://clinicaltables.nlm.nih.gov/api/cosmic/v4/search"
    terms = gene
    if cancer_type:
        terms = f"{gene} {cancer_type}"

    data, error = request_json(
        "GET", base_url,
        params={"terms": terms, "maxList": 500},
        timeout=20, retries=2,
    )
    if error:
        return {"error": f"COSMIC query failed: {error}", "summary": f"COSMIC query failed for {gene}: {error}"}

    # NLM Clinical Tables returns [total_count, codes, extras, display_strings]
    if not isinstance(data, list) or len(data) < 4:
        return {"error": "Unexpected COSMIC API response format", "summary": f"COSMIC query returned unexpected format for {gene}"}

    total_count = data[0] if isinstance(data[0], int) else 0
    display_strings = data[3] if len(data) > 3 else []

    if total_count == 0 or not display_strings:
        cancer_str = f" in {cancer_type}" if cancer_type else ""
        return {
            "summary": f"No COSMIC mutations found for {gene}{cancer_str}",
            "gene": gene,
            "cancer_type": cancer_type,
            "n_mutations": 0,
            "mutations": [],
        }

    # Parse mutation records
    mutations = []
    mutation_types = {}
    aa_changes = {}

    for entry in display_strings:
        if not isinstance(entry, list):
            continue
        # Fields vary by API version; extract what we can
        record = {
            "gene": entry[0] if len(entry) > 0 else "",
            "mutation_cds": entry[1] if len(entry) > 1 else "",
            "mutation_aa": entry[2] if len(entry) > 2 else "",
            "primary_site": entry[3] if len(entry) > 3 else "",
            "mutation_id": entry[4] if len(entry) > 4 else "",
        }
        mutations.append(record)

        # Count mutation types
        aa = record["mutation_aa"]
        if aa:
            aa_changes[aa] = aa_changes.get(aa, 0) + 1
            if "fs" in aa or "del" in aa.lower():
                mtype = "frameshift/deletion"
            elif "*" in aa or "Ter" in aa:
                mtype = "nonsense"
            elif "?" not in aa and aa != "p.?":
                mtype = "missense"
            else:
                mtype = "other"
            mutation_types[mtype] = mutation_types.get(mtype, 0) + 1

    # Top hotspot mutations
    top_mutations = sorted(aa_changes.items(), key=lambda x: -x[1])[:10]

    cancer_str = f" in {cancer_type}" if cancer_type else ""
    top_mut_str = ", ".join(f"{aa}(n={n})" for aa, n in top_mutations[:5])

    return {
        "summary": (
            f"COSMIC {gene}{cancer_str}: {total_count} mutations. "
            f"Spectrum: {', '.join(f'{k}={v}' for k, v in sorted(mutation_types.items(), key=lambda x: -x[1]))}. "
            f"Hotspots: {top_mut_str}"
        ),
        "gene": gene,
        "cancer_type": cancer_type,
        "n_mutations": total_count,
        "mutation_spectrum": mutation_types,
        "top_mutations": [{"mutation": aa, "count": n} for aa, n in top_mutations],
        "mutations": mutations[:30],
    }


@registry.register(
    name="genomics.variant_classify",
    description=(
        "Classify and analyze genomic variants from VCF, Excel, or clinical variant files "
        "(VAF filtering, coding/noncoding classification, ClinVar annotation, carrier analysis)"
    ),
    category="genomics",
    parameters={"goal": "Variant analysis to perform"},
    usage_guide=(
        "Use for variant classification tasks: VAF filtering, Ts/Tv ratios, coding vs noncoding, "
        "CHIP analysis, carrier genotype analysis, ClinVar classification lookups. "
        "Handles multi-row Excel headers, various VAF column naming conventions. "
        "Do NOT use for GWAS, eQTL, or Mendelian randomization — use genomics.gwas_lookup for those."
    ),
)
def variant_classify(goal: str, _session=None, _prior_results=None, **kwargs) -> dict:
    """Classify and analyze genomic variants using generated code in a sandbox."""
    from ct.tools.code import _generate_and_execute_code

    return _generate_and_execute_code(
        goal=goal,
        system_prompt_template=VARIANT_CLASSIFY_PROMPT,
        session=_session,
        prior_results=_prior_results,
    )


# ---------------------------------------------------------------------------
# AlphaMissense pathogenicity lookup
# ---------------------------------------------------------------------------


@registry.register(
    name="genomics.alphamissense_lookup",
    description="Query AlphaMissense pathogenicity scores for human missense variants via Ensembl VEP",
    category="genomics",
    parameters={
        "variant": "Variant in HGVS notation (e.g. 'BRAF:p.Val600Glu') or rsID (e.g. 'rs113488022')",
        "gene": "Gene symbol to help resolve ambiguous variants (optional)",
        "transcript": "Ensembl transcript ID for specific isoform (optional)",
    },
    requires_data=[],
    usage_guide="You want pathogenicity predictions for missense variants. AlphaMissense covers 71M precomputed human missense predictions. Scores: 0-1 (benign < 0.34, ambiguous 0.34-0.564, pathogenic > 0.564). Complements CADD (broader scope) and ClinVar (curated evidence).",
)
def alphamissense_lookup(variant: str, gene: str = None, transcript: str = None, **kwargs) -> dict:
    """Query AlphaMissense pathogenicity scores via the Ensembl VEP REST API."""
    variant = (variant or "").strip()
    if not variant:
        return {"error": "Variant is required", "summary": "AlphaMissense lookup requires a variant"}

    ensembl_base = "https://rest.ensembl.org"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # Determine variant type and query VEP
    if variant.lower().startswith("rs"):
        url = f"{ensembl_base}/vep/human/id/{variant}"
    else:
        url = f"{ensembl_base}/vep/human/hgvs/{variant}"

    params = {}
    if transcript:
        params["transcript_id"] = transcript

    resp, error = request(
        "GET", url, headers=headers, params=params,
        timeout=30, retries=2, raise_for_status=False,
    )
    if error:
        return {"error": f"Ensembl VEP query failed: {error}", "summary": f"AlphaMissense lookup failed: {error}"}
    if resp.status_code == 400:
        return {"error": f"Invalid variant format: '{variant}'", "summary": f"Invalid variant: {variant}"}
    if resp.status_code >= 400:
        return {"error": f"Ensembl VEP returned HTTP {resp.status_code}", "summary": f"VEP query failed for {variant}"}

    try:
        data = resp.json()
    except Exception:
        return {"error": "Invalid JSON from Ensembl VEP", "summary": "Failed to parse VEP response"}

    if not data or not isinstance(data, list):
        return {"error": f"No VEP results for {variant}", "summary": f"No results for {variant}"}

    vep_result = data[0]

    # Extract AlphaMissense scores from transcript consequences
    am_results = []
    for tc in vep_result.get("transcript_consequences", []):
        am_pathogenicity = tc.get("am_pathogenicity")
        am_class = tc.get("am_class")
        if am_pathogenicity is None and am_class is None:
            continue

        # Filter by gene if specified
        if gene and tc.get("gene_symbol", "").upper() != gene.upper():
            continue

        # Filter by transcript if specified
        if transcript and tc.get("transcript_id", "") != transcript:
            continue

        am_results.append({
            "gene_symbol": tc.get("gene_symbol", ""),
            "transcript_id": tc.get("transcript_id", ""),
            "consequence": ", ".join(tc.get("consequence_terms", [])),
            "amino_acids": tc.get("amino_acids", ""),
            "protein_position": tc.get("protein_position", ""),
            "am_pathogenicity": am_pathogenicity,
            "am_class": am_class,
            "sift_prediction": tc.get("sift_prediction", ""),
            "sift_score": tc.get("sift_score"),
            "polyphen_prediction": tc.get("polyphen_prediction", ""),
            "polyphen_score": tc.get("polyphen_score"),
            "canonical": tc.get("canonical", 0) == 1,
        })

    # Sort canonical first, then by pathogenicity score descending
    am_results.sort(key=lambda x: (0 if x["canonical"] else 1, -(x["am_pathogenicity"] or 0)))

    if not am_results:
        return {
            "summary": f"No AlphaMissense scores available for {variant} (may not be a missense variant)",
            "variant": variant,
            "results": [],
            "n_results": 0,
        }

    top = am_results[0]
    gene_str = top["gene_symbol"] or gene or ""
    aa_str = f" ({top['amino_acids']} pos {top['protein_position']})" if top["amino_acids"] else ""

    return {
        "summary": (
            f"AlphaMissense for {variant}: {top['am_class']} "
            f"(score={top['am_pathogenicity']:.3f}) in {gene_str}{aa_str}. "
            f"{len(am_results)} transcript(s) with AM scores."
        ),
        "variant": variant,
        "gene": gene_str,
        "top_score": top["am_pathogenicity"],
        "top_class": top["am_class"],
        "n_results": len(am_results),
        "results": am_results[:20],
    }


# ---------------------------------------------------------------------------
# SpliceAI splice site prediction
# ---------------------------------------------------------------------------


@registry.register(
    name="genomics.spliceai_predict",
    description="Predict splice site variant effects using SpliceAI (Illumina)",
    category="genomics",
    parameters={
        "variant": "Variant in VCF-style format: 'chr-pos-ref-alt' (e.g. '1-55505647-G-A')",
        "distance": "Maximum distance from variant to splice site to consider (default 500, max 10000)",
        "genome": "Reference genome build: 'hg38' (default) or 'hg19'",
    },
    requires_data=[],
    usage_guide="You want to predict whether a variant affects mRNA splicing. SpliceAI is the Illumina gold standard for splice variant interpretation. Scores 0-1 for acceptor/donor gain/loss. Score >= 0.5 = high impact, 0.2-0.5 = moderate, < 0.2 = low. Use for variants near exon-intron boundaries or deep intronic variants.",
)
def spliceai_predict(variant: str, distance: int = 500, genome: str = "hg38", **kwargs) -> dict:
    """Predict splice site effects using the SpliceAI Python package."""
    variant = (variant or "").strip()
    if not variant:
        return {"error": "Variant is required (format: chr-pos-ref-alt)", "summary": "SpliceAI requires a variant"}

    distance = max(50, min(int(distance or 500), 10000))

    try:
        from spliceai.utils import Annotator, get_delta_scores
    except ImportError:
        return {
            "error": "spliceai package not installed",
            "summary": "SpliceAI not installed. Install with: pip install spliceai",
            "install_instructions": {
                "pip": "pip install spliceai",
                "note": "Also requires reference genome FASTA. See https://github.com/Illumina/SpliceAI",
            },
        }

    # Parse variant: expected format chr-pos-ref-alt
    parts = variant.replace(":", "-").split("-")
    if len(parts) != 4:
        return {
            "error": f"Invalid variant format: '{variant}'. Expected 'chr-pos-ref-alt' (e.g. '1-55505647-G-A')",
            "summary": f"Invalid variant format: {variant}",
        }

    chrom, pos_str, ref, alt = parts
    try:
        pos = int(pos_str)
    except ValueError:
        return {"error": f"Invalid position: {pos_str}", "summary": f"Non-numeric position in variant: {pos_str}"}

    # Normalize chromosome
    if not chrom.startswith("chr"):
        chrom_full = f"chr{chrom}"
    else:
        chrom_full = chrom

    try:
        # Build VCF-style record for SpliceAI
        import pysam

        genome_build = "hg38" if "38" in genome else "hg19"
        ref_fasta = f"/usr/share/genomes/{genome_build}.fa"  # standard location

        annotator = Annotator(ref_fasta, "grch38" if genome_build == "hg38" else "grch37")
        record = f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t."
        scores = get_delta_scores(record, annotator, distance)

        results = {
            "acceptor_gain": round(scores[0], 4),
            "acceptor_loss": round(scores[1], 4),
            "donor_gain": round(scores[2], 4),
            "donor_loss": round(scores[3], 4),
        }

        max_score = max(results.values())
        max_effect = max(results, key=results.get)

        if max_score >= 0.5:
            interpretation = "high splice impact"
        elif max_score >= 0.2:
            interpretation = "moderate splice impact"
        else:
            interpretation = "low splice impact"

        return {
            "summary": (
                f"SpliceAI for {variant}: {interpretation} (max delta={max_score:.3f}, {max_effect}). "
                f"AG={results['acceptor_gain']}, AL={results['acceptor_loss']}, "
                f"DG={results['donor_gain']}, DL={results['donor_loss']}"
            ),
            "variant": variant,
            "scores": results,
            "max_score": max_score,
            "max_effect": max_effect,
            "interpretation": interpretation,
            "distance": distance,
            "genome": genome_build,
        }
    except ImportError:
        return {
            "error": "pysam is required for SpliceAI. Install with: pip install pysam",
            "summary": "pysam not installed. Install with: pip install pysam",
        }
    except FileNotFoundError:
        return {
            "error": f"Reference genome FASTA not found for {genome}. SpliceAI requires a local reference genome.",
            "summary": f"Reference genome not found. Download {genome} FASTA for SpliceAI.",
            "install_instructions": {
                "hg38": "Download from https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
                "note": "Place at /usr/share/genomes/hg38.fa or set SPLICEAI_GENOME environment variable",
            },
        }
    except Exception as e:
        return {"error": f"SpliceAI prediction failed: {e}", "summary": f"SpliceAI error: {e}"}


# ---------------------------------------------------------------------------
# CADD pathogenicity scoring
# ---------------------------------------------------------------------------


@registry.register(
    name="genomics.cadd_score",
    description="Query CADD (Combined Annotation Dependent Depletion) scores for variant pathogenicity",
    category="genomics",
    parameters={
        "variant": "Variant in format 'chr:pos:ref:alt' (e.g. '1:55505647:G:A') or 'chr-pos-ref-alt'",
        "genome_build": "Reference genome: 'GRCh38' (default) or 'GRCh37'",
    },
    requires_data=[],
    usage_guide="You want a comprehensive pathogenicity score integrating 100+ annotations. CADD PHRED >= 20 = top 1% most deleterious variants, >= 30 = top 0.1%. Covers all variant types (not just missense like AlphaMissense). Use as first-pass filter for variant prioritization.",
)
def cadd_score(variant: str, genome_build: str = "GRCh38", **kwargs) -> dict:
    """Query the CADD web API for variant pathogenicity scores."""
    variant = (variant or "").strip()
    if not variant:
        return {"error": "Variant is required", "summary": "CADD scoring requires a variant"}

    # Parse variant — accept multiple formats
    parts = variant.replace(":", "-").replace("/", "-").split("-")
    if len(parts) != 4:
        return {
            "error": f"Invalid variant format: '{variant}'. Expected 'chr:pos:ref:alt' (e.g. '1:55505647:G:A')",
            "summary": f"Invalid variant format: {variant}",
        }

    chrom, pos_str, ref, alt = parts
    chrom = chrom.replace("chr", "")

    try:
        pos = int(pos_str)
    except ValueError:
        return {"error": f"Invalid position: {pos_str}", "summary": f"Non-numeric position: {pos_str}"}

    # Determine API version based on genome build
    build = genome_build.upper()
    if "37" in build:
        api_version = "v1.0"
        build_name = "GRCh37"
    else:
        api_version = "v1.0"
        build_name = "GRCh38"

    # Query CADD API
    cadd_url = f"https://cadd.gs.washington.edu/api/{api_version}/{build_name}/{chrom}:{pos}"

    data, error = request_json(
        "GET", cadd_url,
        timeout=30, retries=2,
    )
    if error:
        return {"error": f"CADD API query failed: {error}", "summary": f"CADD query failed: {error}"}

    if not data:
        return {"summary": f"No CADD scores found for {variant}", "variant": variant, "found": False}

    # Parse results — find matching alt allele
    results = []
    matched = None

    entries = data if isinstance(data, list) else [data]
    for entry in entries:
        if isinstance(entry, dict):
            entry_ref = entry.get("Ref", entry.get("ref", ""))
            entry_alt = entry.get("Alt", entry.get("alt", ""))
            raw_score = entry.get("RawScore", entry.get("rawScore"))
            phred_score = entry.get("PHRED", entry.get("phred"))

            result = {
                "chrom": chrom,
                "pos": pos,
                "ref": entry_ref,
                "alt": entry_alt,
                "raw_score": float(raw_score) if raw_score is not None else None,
                "phred_score": float(phred_score) if phred_score is not None else None,
            }
            results.append(result)

            if entry_alt.upper() == alt.upper():
                matched = result

    if not matched and results:
        matched = results[0]

    if not matched:
        return {"summary": f"No CADD scores found for {variant}", "variant": variant, "found": False}

    phred = matched["phred_score"]
    raw = matched["raw_score"]

    # Interpretation
    if phred is not None:
        if phred >= 30:
            interpretation = "likely pathogenic (top 0.1% most deleterious)"
        elif phred >= 20:
            interpretation = "possibly pathogenic (top 1% most deleterious)"
        elif phred >= 15:
            interpretation = "uncertain significance"
        elif phred >= 10:
            interpretation = "likely benign"
        else:
            interpretation = "benign"
    else:
        interpretation = "score unavailable"

    phred_str = f"{phred:.2f}" if phred is not None else "N/A"
    raw_str = f"{raw:.4f}" if raw is not None else "N/A"

    return {
        "summary": (
            f"CADD for {chrom}:{pos}:{ref}>{alt}: PHRED={phred_str}, raw={raw_str}. "
            f"Interpretation: {interpretation}"
        ),
        "variant": variant,
        "chrom": chrom,
        "pos": pos,
        "ref": ref,
        "alt": alt,
        "phred_score": phred,
        "raw_score": raw,
        "interpretation": interpretation,
        "genome_build": build_name,
        "all_results": results[:10],
        "found": True,
    }
