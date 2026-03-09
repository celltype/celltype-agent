"""
Metabolomics tools: HMDB metabolite lookup and search.

Provides access to the Human Metabolome Database (HMDB) for metabolite
identification, pathway analysis, and biospecimen/disease associations.
"""

from ct.tools import registry
from ct.tools.http_client import request_json, request


@registry.register(
    name="metabolomics.hmdb_lookup",
    description="Look up a metabolite in the Human Metabolome Database (HMDB) by ID, name, or InChIKey",
    category="metabolomics",
    parameters={
        "query": "HMDB ID (e.g. 'HMDB0000001'), metabolite name (e.g. 'glucose'), or InChIKey",
    },
    requires_data=[],
    usage_guide="You need comprehensive information about a metabolite — chemical properties, biological pathways, biospecimen locations, normal concentrations, and disease associations. Use for metabolomics data interpretation and biomarker discovery.",
)
def hmdb_lookup(query: str, **kwargs) -> dict:
    """Query HMDB for metabolite information."""
    query = (query or "").strip()
    if not query:
        return {"error": "Query is required (HMDB ID, name, or InChIKey)", "summary": "No query provided for HMDB lookup"}

    # Determine query type and resolve to HMDB ID
    hmdb_id = None
    if query.upper().startswith("HMDB"):
        hmdb_id = query.upper()
        # Normalize HMDB ID to HMDB00XXXXX format
        digits = hmdb_id.replace("HMDB", "")
        if len(digits) < 7:
            hmdb_id = f"HMDB{digits.zfill(7)}"
    else:
        # Search by name or InChIKey first
        search_result = metabolite_search(name=query)
        if search_result.get("n_results", 0) > 0:
            results = search_result.get("results", [])
            if results:
                hmdb_id = results[0].get("hmdb_id", "")
        if not hmdb_id:
            return {
                "summary": f"No HMDB metabolite found for '{query}'",
                "query": query,
                "found": False,
            }

    # Fetch metabolite details from HMDB JSON API
    url = f"https://hmdb.ca/metabolites/{hmdb_id}.json"
    data, error = request_json("GET", url, timeout=20, retries=2)
    if error:
        return {"error": f"HMDB lookup failed: {error}", "summary": f"HMDB query failed for {hmdb_id}: {error}"}

    if not isinstance(data, dict):
        return {"error": "Unexpected HMDB response format", "summary": f"HMDB returned unexpected format for {hmdb_id}"}

    # Extract key fields
    name = data.get("name", "")
    chemical_formula = data.get("chemical_formula", "")
    avg_weight = data.get("average_molecular_weight")
    mono_weight = data.get("monisotopic_molecular_weight")
    smiles = data.get("smiles", "")
    inchikey = data.get("inchikey", "")
    state = data.get("state", "")
    description = data.get("description", "")

    # Taxonomy
    taxonomy = data.get("taxonomy", {}) or {}
    kingdom = taxonomy.get("kingdom", "")
    super_class = taxonomy.get("super_class", "")
    direct_parent = taxonomy.get("direct_parent", "")

    # Biological properties
    bio_properties = data.get("biological_properties", {}) or {}
    pathways = []
    for pw in (bio_properties.get("pathways") or []):
        pathways.append({
            "name": pw.get("name", ""),
            "smpdb_id": pw.get("smpdb_id", ""),
            "kegg_map_id": pw.get("kegg_map_id", ""),
        })

    # Biospecimen locations
    biospecimen_locations = []
    for bs in (data.get("biospecimen_locations") or []):
        if isinstance(bs, dict):
            biospecimen_locations.append(bs.get("biospecimen", ""))
        elif isinstance(bs, str):
            biospecimen_locations.append(bs)

    # Normal concentrations
    normal_concentrations = []
    for nc in (data.get("normal_concentrations") or [])[:10]:
        if isinstance(nc, dict):
            normal_concentrations.append({
                "biospecimen": nc.get("biospecimen", ""),
                "concentration_value": nc.get("concentration_value", ""),
                "concentration_units": nc.get("concentration_units", ""),
                "subject_age": nc.get("subject_age", ""),
                "subject_sex": nc.get("subject_sex", ""),
                "subject_condition": nc.get("subject_condition", ""),
            })

    # Disease associations
    diseases = []
    for d in (data.get("diseases") or []):
        if isinstance(d, dict):
            diseases.append({
                "name": d.get("name", ""),
                "omim_id": d.get("omim_id", ""),
                "references": [r.get("pubmed_id", "") for r in (d.get("references") or [])[:3]],
            })

    # Ontology terms
    ontology = data.get("ontology", {}) or {}
    status = ontology.get("status", "")
    origins = ontology.get("origins", []) or []
    biofunctions = ontology.get("biofunctions", []) or []

    # Protein associations
    protein_associations = []
    for pa in (data.get("protein_associations") or [])[:10]:
        if isinstance(pa, dict):
            protein_associations.append({
                "protein_accession": pa.get("protein_accession", ""),
                "name": pa.get("name", ""),
                "gene_name": pa.get("gene_name", ""),
            })

    # Build summary
    desc_short = description[:200] + "..." if len(description) > 200 else description
    pathway_names = ", ".join(p["name"] for p in pathways[:5]) if pathways else "none"
    disease_names = ", ".join(d["name"] for d in diseases[:3]) if diseases else "none"

    summary = (
        f"{name} ({hmdb_id}): {chemical_formula}, MW={avg_weight}. "
        f"Pathways: {pathway_names}. Diseases: {disease_names}."
    )

    return {
        "summary": summary,
        "hmdb_id": hmdb_id,
        "name": name,
        "description": desc_short,
        "chemical_formula": chemical_formula,
        "average_molecular_weight": avg_weight,
        "monoisotopic_weight": mono_weight,
        "smiles": smiles,
        "inchikey": inchikey,
        "state": state,
        "taxonomy": {
            "kingdom": kingdom,
            "super_class": super_class,
            "direct_parent": direct_parent,
        },
        "pathways": pathways[:15],
        "biospecimen_locations": biospecimen_locations,
        "normal_concentrations": normal_concentrations,
        "diseases": diseases[:10],
        "protein_associations": protein_associations,
        "biofunctions": biofunctions[:10],
        "origins": origins[:5],
        "found": True,
    }


@registry.register(
    name="metabolomics.metabolite_search",
    description="Search HMDB for metabolites by name, molecular mass, or chemical formula",
    category="metabolomics",
    parameters={
        "name": "Metabolite name to search (optional, e.g. 'glucose', 'tryptophan')",
        "mass": "Exact molecular mass to search (optional, float in Da)",
        "formula": "Chemical formula to search (optional, e.g. 'C6H12O6')",
        "mass_tolerance": "Mass tolerance in Da for mass search (default 0.01)",
    },
    requires_data=[],
    usage_guide="You need to identify an unknown metabolite from mass spec data or find metabolites by name/formula. Use for metabolomics peak annotation, compound identification, and metabolite database searches.",
)
def metabolite_search(name: str = None, mass: float = None, formula: str = None, mass_tolerance: float = 0.01, **kwargs) -> dict:
    """Search HMDB for metabolites by name, mass, or formula."""
    if not name and mass is None and not formula:
        return {"error": "At least one search parameter required (name, mass, or formula)", "summary": "No search parameters provided"}

    mass_tolerance = max(0.001, min(float(mass_tolerance or 0.01), 1.0))
    results = []

    # Search by name using HMDB search
    if name:
        name = name.strip()
        resp, error = request(
            "GET",
            "https://hmdb.ca/unearth/q",
            params={"query": name, "searcher": "metabolites", "button": ""},
            headers={"Accept": "application/json"},
            timeout=20, retries=2,
            raise_for_status=False,
        )
        if not error and resp and resp.status_code == 200:
            try:
                data = resp.json()
                if isinstance(data, list):
                    for item in data[:20]:
                        if isinstance(item, dict):
                            results.append({
                                "hmdb_id": item.get("hmdb_id", item.get("accession", "")),
                                "name": item.get("name", ""),
                                "chemical_formula": item.get("chemical_formula", ""),
                                "average_molecular_weight": item.get("average_molecular_weight"),
                            })
                elif isinstance(data, dict) and "results" in data:
                    for item in data["results"][:20]:
                        results.append({
                            "hmdb_id": item.get("hmdb_id", item.get("accession", "")),
                            "name": item.get("name", ""),
                            "chemical_formula": item.get("chemical_formula", ""),
                            "average_molecular_weight": item.get("average_molecular_weight"),
                        })
            except Exception:
                pass

        # If JSON search didn't work, try XML API
        if not results:
            xml_url = f"https://hmdb.ca/metabolites.json?utf8=%E2%9C%93&query={name}&search_type=name"
            data2, error2 = request_json("GET", xml_url, timeout=20, retries=1)
            if not error2 and isinstance(data2, list):
                for item in data2[:20]:
                    if isinstance(item, dict):
                        results.append({
                            "hmdb_id": item.get("accession", ""),
                            "name": item.get("name", ""),
                            "chemical_formula": item.get("chemical_formula", ""),
                            "average_molecular_weight": item.get("average_molecular_weight"),
                        })

    # Search by mass
    if mass is not None:
        mass = float(mass)
        mass_url = "https://hmdb.ca/spectra/ms/search"
        data, error = request_json(
            "GET", mass_url,
            params={
                "query_masses": str(mass),
                "tolerance": str(mass_tolerance),
                "tolerance_units": "Da",
                "adduct_type": "M-H",
                "mode": "negative",
            },
            timeout=20, retries=1,
        )
        if not error and isinstance(data, list):
            for item in data[:20]:
                if isinstance(item, dict):
                    results.append({
                        "hmdb_id": item.get("hmdb_id", item.get("accession", "")),
                        "name": item.get("name", ""),
                        "chemical_formula": item.get("chemical_formula", ""),
                        "average_molecular_weight": item.get("average_molecular_weight"),
                        "mass_difference": round(abs(float(item.get("average_molecular_weight", 0) or 0) - mass), 6),
                    })

    # Search by formula
    if formula:
        formula = formula.strip()
        formula_url = f"https://hmdb.ca/metabolites.json?utf8=%E2%9C%93&query={formula}&search_type=formula"
        data, error = request_json("GET", formula_url, timeout=20, retries=1)
        if not error and isinstance(data, list):
            for item in data[:20]:
                if isinstance(item, dict):
                    results.append({
                        "hmdb_id": item.get("accession", ""),
                        "name": item.get("name", ""),
                        "chemical_formula": item.get("chemical_formula", ""),
                        "average_molecular_weight": item.get("average_molecular_weight"),
                    })

    # Deduplicate by HMDB ID
    seen = set()
    unique_results = []
    for r in results:
        hmdb_id = r.get("hmdb_id", "")
        if hmdb_id and hmdb_id not in seen:
            seen.add(hmdb_id)
            unique_results.append(r)

    # Build search description
    search_parts = []
    if name:
        search_parts.append(f"name='{name}'")
    if mass is not None:
        search_parts.append(f"mass={mass}±{mass_tolerance}Da")
    if formula:
        search_parts.append(f"formula='{formula}'")
    search_desc = ", ".join(search_parts)

    if unique_results:
        names_str = ", ".join(r["name"] for r in unique_results[:5] if r.get("name"))
        summary = f"HMDB search ({search_desc}): {len(unique_results)} results. Top: {names_str}"
    else:
        summary = f"HMDB search ({search_desc}): no results found"

    return {
        "summary": summary,
        "search_params": {"name": name, "mass": mass, "formula": formula, "mass_tolerance": mass_tolerance},
        "n_results": len(unique_results),
        "results": unique_results[:20],
    }
