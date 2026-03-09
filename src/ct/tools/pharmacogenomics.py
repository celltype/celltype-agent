"""
Pharmacogenomics tools: PharmGKB and CPIC guideline lookups.

Provides access to pharmacogenomic clinical annotations, drug labels,
dosing guidelines, and genotype-phenotype relationships.
"""

from ct.tools import registry
from ct.tools.http_client import request_json


@registry.register(
    name="pharmacogenomics.pharmgkb_lookup",
    description="Query PharmGKB for pharmacogenomic annotations — clinical evidence linking genes, drugs, and variants",
    category="pharmacogenomics",
    parameters={
        "query": "Search term (gene symbol, drug name, or rsID)",
        "entity_type": "Entity type to search: 'gene', 'drug', or 'variant' (default 'gene')",
    },
    requires_data=[],
    usage_guide="You need pharmacogenomic evidence for a gene-drug interaction — clinical annotations, drug labels, dosing guidelines, and level of evidence. Use for precision medicine, understanding how genetic variants affect drug response, and identifying FDA-labeled pharmacogenomic biomarkers.",
)
def pharmgkb_lookup(query: str, entity_type: str = "gene", **kwargs) -> dict:
    """Query PharmGKB API for pharmacogenomic data."""
    query = (query or "").strip()
    if not query:
        return {"error": "Search query is required", "summary": "PharmGKB lookup requires a query term"}

    entity_type = (entity_type or "gene").lower()
    if entity_type not in ("gene", "drug", "variant"):
        entity_type = "gene"

    base_url = "https://api.pharmgkb.org/v1/data"

    # Map entity type to PharmGKB endpoint
    type_map = {
        "gene": "gene",
        "drug": "chemical",
        "variant": "variant",
    }
    endpoint = type_map.get(entity_type, "gene")

    # Search for the entity
    search_url = f"{base_url}/{endpoint}"
    data, error = request_json(
        "GET", search_url,
        params={"view": "max", "name": query},
        headers={"Accept": "application/json"},
        timeout=20, retries=2,
    )
    if error:
        # Try alternative search parameter
        data, error = request_json(
            "GET", search_url,
            params={"view": "max", "symbol": query},
            headers={"Accept": "application/json"},
            timeout=20, retries=2,
        )
        if error:
            return {"error": f"PharmGKB query failed: {error}", "summary": f"PharmGKB search failed for '{query}': {error}"}

    # PharmGKB returns {"data": [...], "total": N} or {"data": {}}
    if not isinstance(data, dict):
        return {"error": "Unexpected PharmGKB response", "summary": f"PharmGKB returned unexpected format for '{query}'"}

    entries = data.get("data", [])
    if isinstance(entries, dict):
        entries = [entries]
    if not entries:
        return {"summary": f"No PharmGKB results for '{query}' (type={entity_type})", "query": query, "entity_type": entity_type, "found": False}

    entry = entries[0]

    # Extract common fields
    pharmgkb_id = entry.get("id", "")
    name = entry.get("name", entry.get("symbol", query))
    symbol = entry.get("symbol", "")

    # Extract cross-references
    xrefs = entry.get("crossReferences", []) or []
    cross_refs = {}
    for xref in xrefs:
        if isinstance(xref, dict):
            resource = xref.get("resource", "")
            res_id = xref.get("resourceId", "")
            if resource and res_id:
                cross_refs[resource] = res_id

    # Get clinical annotations for this entity
    clinical_annotations = []
    ann_url = f"{base_url}/clinicalAnnotation"
    ann_data, ann_error = request_json(
        "GET", ann_url,
        params={"view": "max", "relatedEntity": pharmgkb_id},
        headers={"Accept": "application/json"},
        timeout=20, retries=1,
    )
    if not ann_error and isinstance(ann_data, dict):
        for ann in (ann_data.get("data") or [])[:20]:
            if isinstance(ann, dict):
                evidence_level = ann.get("level", {})
                if isinstance(evidence_level, dict):
                    evidence_level = evidence_level.get("term", "")

                related_chemicals = []
                related_genes = []
                for rel in (ann.get("relatedChemicals") or []):
                    if isinstance(rel, dict):
                        related_chemicals.append(rel.get("name", ""))
                for rel in (ann.get("relatedGenes") or []):
                    if isinstance(rel, dict):
                        related_genes.append(rel.get("symbol", rel.get("name", "")))

                clinical_annotations.append({
                    "id": ann.get("id", ""),
                    "evidence_level": evidence_level,
                    "phenotype_categories": ann.get("phenotypeCategories", []),
                    "related_chemicals": related_chemicals,
                    "related_genes": related_genes,
                    "text": (ann.get("text") or "")[:300],
                })

    # Get drug labels
    drug_labels = []
    label_url = f"{base_url}/drugLabel"
    label_data, label_error = request_json(
        "GET", label_url,
        params={"view": "max", "relatedEntity": pharmgkb_id},
        headers={"Accept": "application/json"},
        timeout=20, retries=1,
    )
    if not label_error and isinstance(label_data, dict):
        for label in (label_data.get("data") or [])[:10]:
            if isinstance(label, dict):
                drug_labels.append({
                    "id": label.get("id", ""),
                    "name": label.get("name", ""),
                    "source": label.get("source", ""),
                    "testing_level": label.get("testingLevel", ""),
                    "prescribing_info": label.get("prescribingInfo", ""),
                })

    # Get dosing guidelines
    guidelines = []
    guide_url = f"{base_url}/guideline"
    guide_data, guide_error = request_json(
        "GET", guide_url,
        params={"view": "max", "relatedEntity": pharmgkb_id},
        headers={"Accept": "application/json"},
        timeout=20, retries=1,
    )
    if not guide_error and isinstance(guide_data, dict):
        for guide in (guide_data.get("data") or [])[:10]:
            if isinstance(guide, dict):
                guidelines.append({
                    "id": guide.get("id", ""),
                    "name": guide.get("name", ""),
                    "source": guide.get("source", ""),
                    "url": guide.get("url", guide.get("@id", "")),
                })

    # Build summary
    ann_str = f"{len(clinical_annotations)} clinical annotation(s)" if clinical_annotations else "no clinical annotations"
    levels = [a["evidence_level"] for a in clinical_annotations if a.get("evidence_level")]
    level_str = f" (levels: {', '.join(sorted(set(levels))[:5])})" if levels else ""
    label_str = f", {len(drug_labels)} drug label(s)" if drug_labels else ""
    guide_str = f", {len(guidelines)} guideline(s)" if guidelines else ""

    summary = f"PharmGKB {name} ({pharmgkb_id}): {ann_str}{level_str}{label_str}{guide_str}"

    return {
        "summary": summary,
        "pharmgkb_id": pharmgkb_id,
        "name": name,
        "symbol": symbol,
        "entity_type": entity_type,
        "cross_references": cross_refs,
        "clinical_annotations": clinical_annotations,
        "drug_labels": drug_labels,
        "guidelines": guidelines,
        "found": True,
    }


@registry.register(
    name="pharmacogenomics.cpic_guidelines",
    description="Query CPIC (Clinical Pharmacogenetics Implementation Consortium) for genotype-based drug dosing guidelines",
    category="pharmacogenomics",
    parameters={
        "gene": "Gene symbol (optional, e.g. 'CYP2D6', 'CYP2C19')",
        "drug": "Drug name (optional, e.g. 'codeine', 'clopidogrel')",
    },
    requires_data=[],
    usage_guide="You need genotype-guided dosing recommendations from CPIC — the gold standard for pharmacogenomic clinical guidelines. Use when a patient's genotype is known and you need to determine the recommended drug/dose based on their metabolizer phenotype.",
)
def cpic_guidelines(gene: str = None, drug: str = None, **kwargs) -> dict:
    """Query CPIC API for genotype-based dosing guidelines."""
    if not gene and not drug:
        return {"error": "At least one of 'gene' or 'drug' is required", "summary": "CPIC lookup requires a gene or drug name"}

    base_url = "https://api.cpicpgx.org/v1"

    # Build query parameters
    params = {}
    if gene:
        params["genesymbol"] = gene.strip().upper()
    if drug:
        params["drugname"] = f"eq.{drug.strip().lower()}"

    # Get gene-drug pairs
    pairs_url = f"{base_url}/pair"
    pair_params = {}
    if gene:
        pair_params["genesymbol"] = f"eq.{gene.strip().upper()}"
    if drug:
        pair_params["drugname"] = f"eq.{drug.strip().lower()}"

    pairs_data, pairs_error = request_json(
        "GET", pairs_url,
        params=pair_params if pair_params else None,
        headers={"Accept": "application/json"},
        timeout=20, retries=2,
    )
    if pairs_error:
        return {"error": f"CPIC pair query failed: {pairs_error}", "summary": f"CPIC query failed: {pairs_error}"}

    if not isinstance(pairs_data, list):
        pairs_data = [pairs_data] if isinstance(pairs_data, dict) else []

    if not pairs_data:
        gene_str = f" gene={gene}" if gene else ""
        drug_str = f" drug={drug}" if drug else ""
        return {
            "summary": f"No CPIC guidelines found for{gene_str}{drug_str}",
            "gene": gene,
            "drug": drug,
            "found": False,
        }

    # Extract guideline information from pairs
    guidelines = []
    for pair in pairs_data:
        if not isinstance(pair, dict):
            continue

        guideline_info = {
            "gene": pair.get("genesymbol", ""),
            "drug": pair.get("drugname", ""),
            "cpic_level": pair.get("cpiclevel", ""),
            "pgkb_level": pair.get("pgkblevelofevidence", ""),
            "pgx_on_fda_label": pair.get("pgxonfdalabel", ""),
            "cpic_guideline_url": pair.get("url", pair.get("guidelineurl", "")),
            "guideline_id": pair.get("guidelineid", ""),
        }
        guidelines.append(guideline_info)

    # Get recommendations (phenotype-based dosing)
    recommendations = []
    for guide in guidelines[:5]:
        guideline_id = guide.get("guideline_id", "")
        gene_sym = guide.get("gene", "")
        drug_name = guide.get("drug", "")

        if not gene_sym or not drug_name:
            continue

        rec_url = f"{base_url}/recommendation"
        rec_params = {
            "drugrecommendation": f"cs.{{{drug_name}}}",
        }

        rec_data, rec_error = request_json(
            "GET", rec_url,
            params=rec_params,
            headers={"Accept": "application/json"},
            timeout=20, retries=1,
        )
        if rec_error or not isinstance(rec_data, list):
            continue

        for rec in rec_data[:20]:
            if not isinstance(rec, dict):
                continue
            recommendations.append({
                "gene": gene_sym,
                "drug": drug_name,
                "phenotype": rec.get("phenotypes", {}) if isinstance(rec.get("phenotypes"), dict) else rec.get("lookupkey", {}),
                "activity_score": rec.get("activityscore", ""),
                "recommendation": rec.get("drugrecommendation", ""),
                "classification": rec.get("classification", ""),
                "strength": rec.get("strength", ""),
            })

    # Build summary
    gene_drug_pairs = [f"{g['gene']}/{g['drug']} (level {g['cpic_level']})" for g in guidelines[:5]]
    pairs_str = "; ".join(gene_drug_pairs)
    rec_count = len(recommendations)

    summary = f"CPIC guidelines: {len(guidelines)} gene-drug pair(s): {pairs_str}. {rec_count} dosing recommendation(s)."

    return {
        "summary": summary,
        "gene": gene,
        "drug": drug,
        "n_pairs": len(guidelines),
        "guidelines": guidelines,
        "n_recommendations": rec_count,
        "recommendations": recommendations[:30],
        "found": True,
    }
