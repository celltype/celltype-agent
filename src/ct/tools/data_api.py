"""
Data API tools: rich wrappers for major biomedical data platforms.

Provides general-purpose access to DepMap, Open Targets, UniProt, PDB,
Ensembl, NCBI, ChEMBL, and DrugBank/PubChem.
"""

import logging
import re

from ct.tools import registry
from ct.tools.http_client import request


def _http_get(url: str, *, params=None, headers=None, timeout: int = 15, retries: int = 2):
    """GET helper with transient retry/backoff semantics."""
    import httpx

    # Preserve historical semantics where `retries` represented max attempts.
    resp, error = request(
        "GET",
        url,
        params=params,
        headers=headers,
        timeout=timeout,
        retries=max(retries - 1, 0),
        raise_for_status=False,
    )
    if error:
        raise httpx.HTTPError(error)
    return resp


def _http_post(url: str, *, json=None, data=None, params=None,
               headers=None, timeout: int = 15, retries: int = 2):
    """POST helper with transient retry/backoff semantics."""
    import httpx

    # Preserve historical semantics where `retries` represented max attempts.
    resp, error = request(
        "POST",
        url,
        json=json,
        data=data,
        params=params,
        headers=headers,
        timeout=timeout,
        retries=max(retries - 1, 0),
        raise_for_status=False,
    )
    if error:
        raise httpx.HTTPError(error)
    return resp

_logger = logging.getLogger("ct.data_api")


def _normalize_gene_name(gene: str) -> str:
    """Normalize a gene symbol: uppercase, strip whitespace, remove common prefixes."""
    gene = gene.strip()
    # Strip common noise prefixes that confuse APIs
    for prefix in ("gene ", "Gene ", "GENE ", "human ", "Human "):
        if gene.startswith(prefix):
            gene = gene[len(prefix):]
    gene = gene.strip()
    # Gene symbols should be uppercase alphanumeric (with hyphens/dots allowed)
    # If it looks like a gene symbol, uppercase it
    if re.match(r'^[A-Za-z][A-Za-z0-9._-]*$', gene):
        gene = gene.upper()
    return gene


def _normalize_drug_query(query: str) -> str:
    """Strip noise words from drug name queries that confuse APIs."""
    noise_prefixes = [
        "fda-approved ", "fda approved ", "approved drug ",
        "drug ", "compound ", "the drug ", "the compound ",
        "investigational ", "experimental ",
    ]
    cleaned = query.strip()
    # Keep stripping prefixes (case-insensitive) until none match
    changed = True
    while changed:
        changed = False
        lower = cleaned.lower()
        for prefix in noise_prefixes:
            if lower.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                changed = True
                break
    return cleaned.strip()


# ---------------------------------------------------------------------------
# 1. DepMap search
# ---------------------------------------------------------------------------

@registry.register(
    name="data_api.depmap_search",
    description="Search DepMap for gene dependency scores across cancer cell lines",
    category="data_api",
    parameters={
        "gene": "Gene symbol (e.g. BRCA1, TP53)",
        "dataset": "Dataset to query: 'crispr', 'expression', 'mutations', or 'cn' (default 'crispr')",
    },
    requires_data=[],
    usage_guide="You want DepMap gene dependency data across cell lines. Returns dependency scores, most/least dependent lineages. Uses local DepMap data when available, or the Cell Model Passports API as fallback.",
)
def depmap_search(gene: str, dataset: str = "crispr", **kwargs) -> dict:
    """Search DepMap for gene dependency / expression / mutation data.

    Tries local DepMap data first (via ct data loaders), then falls back to
    the Cell Model Passports API (public, no key required).
    """
    import numpy as np

    valid_datasets = ("crispr", "expression", "mutations", "cn")
    if dataset not in valid_datasets:
        return {"error": f"Invalid dataset '{dataset}'. Choose from: {', '.join(valid_datasets)}", "summary": f"Invalid dataset '{dataset}'"}

    # Normalize gene name
    gene = _normalize_gene_name(gene)

    # --- Attempt local DepMap data ---
    if dataset == "crispr":
        try:
            from ct.data.loaders import load_crispr
            crispr = load_crispr()
            # Try exact match first, then common variations
            if gene not in crispr.columns:
                # Try with/without hyphens, dots, etc.
                found = False
                for variant in [gene.replace("-", ""), gene.replace(".", ""), gene + "A"]:
                    if variant in crispr.columns:
                        _logger.warning("Gene '%s' not found, using variant '%s'", gene, variant)
                        gene = variant
                        found = True
                        break
                if not found:
                    # Try partial match (e.g., "CD274" matches "CD274 (PD-L1)")
                    matches = [c for c in crispr.columns if c.startswith(gene + " ") or c == gene]
                    if matches:
                        gene = matches[0]
                        _logger.warning("Gene exact match not found, using '%s'", gene)
                    else:
                        return {"error": f"Gene {gene} not found in local DepMap CRISPR data", "summary": f"Gene {gene} not in DepMap CRISPR"}

            scores = crispr[gene].dropna()
            n_lines = len(scores)
            essential = (scores < -0.5).sum()
            mean_score = float(scores.mean())
            min_score = float(scores.min())

            # Lineage info if model metadata available
            lineage_stats = []
            try:
                from ct.data.loaders import load_model_metadata
                model = load_model_metadata()
                merged = scores.to_frame(name="score").join(
                    model.set_index("ModelID")["OncotreeLineage"], how="left"
                )
                if "OncotreeLineage" in merged.columns:
                    for lin, grp in merged.groupby("OncotreeLineage"):
                        lineage_stats.append({
                            "lineage": lin,
                            "mean_score": round(float(grp["score"].mean()), 4),
                            "n_lines": len(grp),
                            "n_essential": int((grp["score"] < -0.5).sum()),
                        })
                    lineage_stats.sort(key=lambda x: x["mean_score"])
            except Exception:
                pass

            most_dependent = [ls["lineage"] for ls in lineage_stats[:3]] if lineage_stats else []
            least_dependent = [ls["lineage"] for ls in lineage_stats[-3:]] if lineage_stats else []

            return {
                "summary": (
                    f"{gene} dependency (DepMap CRISPR): essential in {essential}/{n_lines} lines, "
                    f"mean score {mean_score:.3f}"
                    + (f", most dependent: {', '.join(most_dependent)}" if most_dependent else "")
                ),
                "gene": gene,
                "dataset": "crispr",
                "n_cell_lines": n_lines,
                "n_essential": int(essential),
                "mean_score": round(mean_score, 4),
                "min_score": round(min_score, 4),
                "lineage_stats": lineage_stats[:20],
                "most_dependent_lineages": most_dependent,
                "least_dependent_lineages": least_dependent,
            }
        except (ImportError, FileNotFoundError):
            pass  # Fall through to API

    if dataset == "mutations":
        try:
            from ct.data.loaders import load_mutations
            mutations = load_mutations()
            if gene not in mutations.columns:
                return {"error": f"Gene {gene} not found in local DepMap mutation data"}

            mutated = mutations[gene].dropna()
            n_lines = len(mutated)
            n_mutated = int((mutated > 0).sum())
            mutation_rate = n_mutated / n_lines if n_lines > 0 else 0

            return {
                "summary": (
                    f"{gene} mutations (DepMap): mutated in {n_mutated}/{n_lines} lines "
                    f"({mutation_rate:.1%})"
                ),
                "gene": gene,
                "dataset": "mutations",
                "n_cell_lines": n_lines,
                "n_mutated": n_mutated,
                "mutation_rate": round(mutation_rate, 4),
            }
        except (ImportError, FileNotFoundError):
            pass  # Fall through to API

    # --- Fallback: Cell Model Passports API ---
    try:
        resp = _http_get(
            "https://www.cellmodelpassports.sanger.ac.uk/api/v1/genes",
            params={"search": gene, "page_size": 5},
            timeout=15,
        )
        if resp.status_code != 200:
            return {
                "error": f"Cell Model Passports API returned HTTP {resp.status_code}",
                "summary": f"Could not query DepMap/CMP for {gene}",
            }
        data = resp.json()
    except Exception as e:
        import httpx
        if isinstance(e, httpx.TimeoutException):
            return {"error": "Cell Model Passports API timed out", "summary": f"CMP timeout for {gene}"}
        if isinstance(e, httpx.HTTPError):
            return {"error": f"CMP API error: {e}", "summary": f"CMP query failed for {gene}"}
        return {"error": f"CMP API error: {e}", "summary": f"CMP query failed for {gene}"}

    results = data.get("data", data.get("results", []))
    if not results:
        return {
            "error": f"Gene {gene} not found in Cell Model Passports",
            "summary": f"No results for {gene} in CMP",
        }

    gene_info = results[0] if isinstance(results, list) else results
    return {
        "summary": f"DepMap/CMP: {gene} — found in Cell Model Passports database",
        "gene": gene,
        "dataset": dataset,
        "source": "cell_model_passports",
        "gene_info": gene_info,
    }


# ---------------------------------------------------------------------------
# 2. Open Targets search
# ---------------------------------------------------------------------------

@registry.register(
    name="data_api.opentargets_search",
    description="Search Open Targets Platform for comprehensive target, disease, or drug profiles",
    category="data_api",
    parameters={
        "query": "Gene name, disease name, or drug name",
        "entity_type": "Entity type: 'target', 'disease', or 'drug' (default 'target')",
    },
    requires_data=[],
    usage_guide="You want a comprehensive profile from Open Targets: disease associations for a target, associated targets for a disease, or indications/mechanisms for a drug. General-purpose Open Targets access.",
)
def opentargets_search(query: str, entity_type: str = "target", **kwargs) -> dict:
    """Query Open Targets Platform GraphQL API for target/disease/drug profiles."""
    ot_url = "https://api.platform.opentargets.org/api/v4/graphql"
    headers = {"Content-Type": "application/json"}

    valid_types = ("target", "disease", "drug")
    if entity_type not in valid_types:
        return {"error": f"Invalid entity_type '{entity_type}'. Choose from: {', '.join(valid_types)}", "summary": f"Invalid entity type '{entity_type}'"}

    # Normalize query based on entity type
    if entity_type == "target":
        query = _normalize_gene_name(query)
    elif entity_type == "drug":
        query = _normalize_drug_query(query)

    # Step 1: Search to resolve ID
    search_gql = """
    query search($q: String!, $entities: [String!]!) {
        search(queryString: $q, entityNames: $entities, page: {size: 5, index: 0}) {
            total
            hits { id entity name description }
        }
    }
    """
    entity_names = {
        "target": ["target"],
        "disease": ["disease"],
        "drug": ["drug"],
    }

    try:
        search_resp = _http_post(
            ot_url,
            json={"query": search_gql, "variables": {"q": query, "entities": entity_names[entity_type]}},
            headers=headers,
            timeout=15,
        )
        search_resp.raise_for_status()
        search_data = search_resp.json()
    except Exception as e:
        import httpx
        if isinstance(e, httpx.TimeoutException):
            return {"error": f"Open Targets search timed out for '{query}'", "summary": f"Open Targets timed out for '{query}'"}
        if isinstance(e, httpx.HTTPError):
            return {"error": f"Open Targets search failed: {e}", "summary": f"Open Targets search failed"}
        return {"error": f"Open Targets search failed: {e}", "summary": f"Open Targets search failed"}

    hits = search_data.get("data", {}).get("search", {}).get("hits", [])
    total = search_data.get("data", {}).get("search", {}).get("total", 0)

    if not hits:
        return {
            "error": f"No {entity_type} found for '{query}' in Open Targets",
            "summary": f"Open Targets: no {entity_type} matches for '{query}'",
        }

    top_hit = hits[0]
    entity_id = top_hit["id"]
    entity_name = top_hit.get("name", query)

    # Step 2: Fetch detailed profile
    if entity_type == "target":
        detail_gql = """
        query targetProfile($id: String!) {
            target(ensemblId: $id) {
                id
                approvedSymbol
                approvedName
                biotype
                functionDescriptions
                subcellularLocations { location }
                tractability {
                    label
                    modality
                    value
                }
                associatedDiseases(page: {size: 10, index: 0}) {
                    count
                    rows {
                        disease { id name }
                        score
                    }
                }
                knownDrugs(size: 10) {
                    uniqueDrugs
                    rows {
                        prefName
                        drugType
                        mechanismOfAction
                        phase
                    }
                }
            }
        }
        """
        variables = {"id": entity_id}

    elif entity_type == "disease":
        detail_gql = """
        query diseaseProfile($id: String!) {
            disease(efoId: $id) {
                id
                name
                description
                therapeuticAreas { id name }
                associatedTargets(page: {size: 10, index: 0}) {
                    count
                    rows {
                        target { id approvedSymbol }
                        score
                    }
                }
                knownDrugs(size: 10) {
                    uniqueDrugs
                    rows {
                        prefName
                        drugType
                        phase
                        mechanismOfAction
                    }
                }
            }
        }
        """
        variables = {"id": entity_id}

    else:  # drug
        detail_gql = """
        query drugProfile($id: String!) {
            drug(chemblId: $id) {
                id
                name
                drugType
                maximumClinicalTrialPhase
                hasBeenWithdrawn
                description
                mechanismsOfAction {
                    rows {
                        mechanismOfAction
                        targets { id approvedSymbol }
                    }
                }
                indications {
                    count
                    rows {
                        disease { id name }
                        maxPhaseForIndication
                    }
                }
            }
        }
        """
        variables = {"id": entity_id}

    try:
        detail_resp = _http_post(
            ot_url,
            json={"query": detail_gql, "variables": variables},
            headers=headers,
            timeout=15,
        )
        detail_resp.raise_for_status()
        detail_data = detail_resp.json()
    except Exception as e:
        import httpx
        if isinstance(e, httpx.TimeoutException):
            return {"error": f"Open Targets detail query timed out for {entity_id}", "summary": f"Open Targets detail timed out"}
        if isinstance(e, httpx.HTTPError):
            return {"error": f"Open Targets detail query failed: {e}", "summary": f"Open Targets detail query failed"}
        return {"error": f"Open Targets detail query failed: {e}", "summary": f"Open Targets detail query failed"}

    data_root = detail_data.get("data", {})

    if entity_type == "target":
        target = data_root.get("target") or {}
        assoc = target.get("associatedDiseases", {})
        n_diseases = assoc.get("count", 0)
        top_diseases = [
            {"disease": r["disease"]["name"], "score": round(r["score"], 3)}
            for r in assoc.get("rows", [])
        ]
        known_drugs = target.get("knownDrugs", {})
        n_drugs = known_drugs.get("uniqueDrugs", 0)
        drug_rows = known_drugs.get("rows", [])
        tractability = target.get("tractability", [])

        top_disease_str = ", ".join(
            f"{d['disease']} ({d['score']:.2f})" for d in top_diseases[:3]
        )
        return {
            "summary": (
                f"Open Targets: {target.get('approvedSymbol', query)} — "
                f"{n_diseases} disease associations, "
                f"top: {top_disease_str or 'none'}. "
                f"{n_drugs} known drug(s)."
            ),
            "entity_type": "target",
            "entity_id": entity_id,
            "approved_symbol": target.get("approvedSymbol", ""),
            "approved_name": target.get("approvedName", ""),
            "biotype": target.get("biotype", ""),
            "function": target.get("functionDescriptions", []),
            "tractability": tractability,
            "n_disease_associations": n_diseases,
            "top_diseases": top_diseases,
            "n_known_drugs": n_drugs,
            "known_drugs": [
                {
                    "name": d.get("prefName", ""),
                    "type": d.get("drugType", ""),
                    "mechanism": d.get("mechanismOfAction", ""),
                    "phase": d.get("phase", 0),
                }
                for d in drug_rows[:10]
            ],
        }

    elif entity_type == "disease":
        disease = data_root.get("disease") or {}
        assoc = disease.get("associatedTargets", {})
        n_targets = assoc.get("count", 0)
        top_targets = [
            {"gene": r["target"]["approvedSymbol"], "score": round(r["score"], 3)}
            for r in assoc.get("rows", [])
        ]
        therapeutic_areas = [ta["name"] for ta in disease.get("therapeuticAreas", [])]
        known_drugs = disease.get("knownDrugs", {})
        n_drugs = known_drugs.get("uniqueDrugs", 0)

        top_target_str = ", ".join(
            f"{t['gene']} ({t['score']:.2f})" for t in top_targets[:3]
        )
        return {
            "summary": (
                f"Open Targets: {disease.get('name', query)} — "
                f"{n_targets} associated targets, "
                f"top: {top_target_str or 'none'}. "
                f"Areas: {', '.join(therapeutic_areas[:3]) or 'N/A'}."
            ),
            "entity_type": "disease",
            "entity_id": entity_id,
            "name": disease.get("name", ""),
            "description": disease.get("description", ""),
            "therapeutic_areas": therapeutic_areas,
            "n_associated_targets": n_targets,
            "top_targets": top_targets,
            "n_known_drugs": n_drugs,
        }

    else:  # drug
        drug = data_root.get("drug") or {}
        moa_rows = drug.get("mechanismsOfAction", {}).get("rows", [])
        indications = drug.get("indications", {})
        n_indications = indications.get("count", 0)
        ind_rows = indications.get("rows", [])

        mechanisms = [m.get("mechanismOfAction", "") for m in moa_rows]
        return {
            "summary": (
                f"Open Targets: {drug.get('name', query)} — "
                f"{drug.get('drugType', 'unknown')} drug, "
                f"max phase {drug.get('maximumClinicalTrialPhase', 'N/A')}, "
                f"{n_indications} indications."
            ),
            "entity_type": "drug",
            "entity_id": entity_id,
            "name": drug.get("name", ""),
            "drug_type": drug.get("drugType", ""),
            "max_clinical_phase": drug.get("maximumClinicalTrialPhase"),
            "withdrawn": drug.get("hasBeenWithdrawn", False),
            "description": drug.get("description", ""),
            "mechanisms": mechanisms,
            "n_indications": n_indications,
            "indications": [
                {"disease": r["disease"]["name"], "max_phase": r.get("maxPhaseForIndication")}
                for r in ind_rows[:15]
            ],
        }


# ---------------------------------------------------------------------------
# 3. UniProt lookup
# ---------------------------------------------------------------------------

_UNIPROT_NON_HUMAN_HINTS = (
    "helminth",
    "parasite",
    "schistosoma",
    "fasciola",
    "heligmosomoides",
    "nematode",
    "trematode",
    "cestode",
    "worm",
    "brugia",
    "filaria",
)

_UNIPROT_QUERY_STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "from", "with", "without",
    "in", "on", "of", "to", "by", "via", "as", "that", "this", "these",
    "those", "are", "is", "was", "were", "be", "been", "being", "it",
    "its", "their", "minimal", "annotation", "annotations", "key", "keys",
    "look", "lookup", "search", "find", "protein", "proteins", "immunomodulatory",
}


def _query_has_non_human_hints(query: str) -> bool:
    q = (query or "").lower()
    return any(hint in q for hint in _UNIPROT_NON_HUMAN_HINTS)


def _keyword_fallback_query(query: str, max_terms: int = 7) -> str:
    tokens = re.findall(r"[A-Za-z0-9_-]+", (query or "").lower())
    selected = []
    for tok in tokens:
        if len(tok) < 3:
            continue
        if tok in _UNIPROT_QUERY_STOPWORDS:
            continue
        if tok not in selected:
            selected.append(tok)
        if len(selected) >= max_terms:
            break
    return " ".join(selected)


def _extract_species_phrases(query: str, max_species: int = 3) -> list[str]:
    """Extract likely binomial species names from free-text query."""
    tokens = re.findall(r"[A-Za-z][A-Za-z-]*", query or "")
    species: list[str] = []
    for i in range(len(tokens) - 1):
        genus_raw = tokens[i]
        species_raw = tokens[i + 1]
        genus = genus_raw.lower()
        epithet = species_raw.lower()

        if len(genus) < 3 or len(epithet) < 3:
            continue
        if genus in _UNIPROT_QUERY_STOPWORDS or epithet in _UNIPROT_QUERY_STOPWORDS:
            continue
        if not epithet.isalpha():
            continue
        if not (genus in _UNIPROT_NON_HUMAN_HINTS or genus_raw[0].isupper()):
            continue

        phrase = f"{genus.capitalize()} {epithet}"
        if phrase not in species:
            species.append(phrase)
        if len(species) >= max_species:
            break
    return species


def _build_uniprot_search_candidates(
    *,
    query: str,
    compact_query: str,
    org_clause: str | None,
) -> list[str]:
    """Generate ranked UniProt search candidates for robust retrieval."""
    q = (query or "").strip()
    q_lc = q.lower()
    species = _extract_species_phrases(q)
    candidates: list[str] = []

    def add(candidate: str):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    if q:
        if org_clause:
            add(f"({q}) AND {org_clause}")
        add(q)

    if compact_query and compact_query != q:
        if org_clause:
            add(f"({compact_query}) AND {org_clause}")
        add(compact_query)

    wants_secreted = any(x in q_lc for x in ("secreted", "excretory", "extracellular", "vesicle", "ev "))
    wants_uncharacterized = any(x in q_lc for x in ("uncharacterized", "understudied", "novel", "hypothetical"))
    wants_scp_taps = any(
        x in q_lc for x in ("venom allergen", "scp", "taps", "val", "cap superfamily", "allergen-like")
    )

    for sp in species:
        sp_clause = f'organism_name:"{sp}"'
        add(sp_clause)
        if wants_secreted:
            add(f'{sp_clause} AND (secreted OR excretory OR extracellular)')
        if wants_uncharacterized:
            add(f'{sp_clause} AND (uncharacterized OR hypothetical)')
        if wants_scp_taps:
            add(f'{sp_clause} AND ("venom allergen" OR SCP OR TAPS OR VAL)')

    if _query_has_non_human_hints(q):
        add("parasite")
        add("helminth")
        add("schistosoma")
        add("fasciola")
        add("heligmosomoides")
        if wants_secreted:
            add("(parasite OR helminth) AND (secreted OR excretory OR extracellular)")
        if wants_scp_taps:
            add('(parasite OR helminth) AND ("venom allergen" OR SCP OR TAPS OR VAL)')

    # Keep search bounded to avoid excessive API calls.
    return candidates[:12]


def _entry_text_blob(entry: dict) -> str:
    parts: list[str] = []
    pd = entry.get("proteinDescription", {}) or {}
    rec = (pd.get("recommendedName", {}) or {}).get("fullName", {}) or {}
    if rec.get("value"):
        parts.append(str(rec.get("value")))
    for alt in (pd.get("alternativeNames", []) or []):
        full = (alt.get("fullName", {}) or {}).get("value")
        if full:
            parts.append(str(full))
    for kw in (entry.get("keywords", []) or []):
        name = kw.get("name")
        if name:
            parts.append(str(name))
    org = entry.get("organism", {}) or {}
    sci = org.get("scientificName")
    if sci:
        parts.append(str(sci))
    return " ".join(parts).lower()


def _entry_relevance_score(
    entry: dict,
    *,
    original_query: str,
    species_phrases: list[str],
    non_human_hints: bool,
) -> float:
    q_lc = (original_query or "").lower()
    blob = _entry_text_blob(entry)
    score = 0.0

    wants_secreted = any(x in q_lc for x in ("secreted", "excretory", "extracellular", "vesicle", "ev "))
    wants_uncharacterized = any(x in q_lc for x in ("uncharacterized", "understudied", "novel", "hypothetical"))
    wants_scp_taps = any(x in q_lc for x in ("venom allergen", "scp", "taps", "val", "cap superfamily"))

    # Species alignment dominates ranking.
    for sp in species_phrases:
        if sp.lower() in blob:
            score += 8.0
    if non_human_hints and "homo sapiens" in blob:
        score -= 10.0

    if wants_secreted and any(x in blob for x in ("secreted", "excretory", "extracellular", "signal peptide")):
        score += 3.0
    if wants_uncharacterized and any(x in blob for x in ("uncharacterized", "hypothetical", "putative")):
        score += 3.0
    if wants_scp_taps and any(x in blob for x in ("venom allergen", "scp", "taps", "val", "cap")):
        score += 4.0

    # Penalize clearly off-target "query not represented in entry text".
    query_tokens = [t for t in re.findall(r"[a-z0-9_-]+", q_lc) if len(t) >= 4]
    overlap = sum(1 for t in query_tokens[:8] if t in blob)
    score += min(overlap * 0.5, 2.0)

    entry_type = str(entry.get("entryType", "")).lower()
    if wants_uncharacterized and "unreviewed" in entry_type:
        score += 1.0

    return score


@registry.register(
    name="data_api.uniprot_lookup",
    description="Look up comprehensive protein information from UniProt by gene symbol, UniProt ID, or protein name",
    category="data_api",
    parameters={
        "query": "Gene symbol, UniProt accession (e.g. P04637), or protein name",
        "organism": "Organism filter: common name (human/mouse/...), taxonomy ID, or 'any' (default 'human')",
    },
    requires_data=[],
    usage_guide="You need detailed protein information: function, domains, subcellular location, GO terms, PDB structures, disease involvement, tissue specificity. Comprehensive UniProt protein profile.",
)
def uniprot_lookup(query: str, organism: str = "human", **kwargs) -> dict:
    """Look up comprehensive protein data from UniProt REST API."""
    organism_ids = {
        "human": 9606, "mouse": 10090, "rat": 10116,
        "zebrafish": 7955, "drosophila": 7227, "yeast": 559292,
    }
    organism_clean = (organism or "human").strip()
    organism_lc = organism_clean.lower()

    org_clause = None
    if organism_lc not in ("", "any", "all", "none"):
        if organism_lc.isdigit():
            org_clause = f"organism_id:{organism_lc}"
        elif organism_lc in organism_ids:
            org_clause = f"organism_id:{organism_ids[organism_lc]}"
        else:
            escaped = organism_clean.replace('"', "")
            if escaped:
                org_clause = f'organism_name:"{escaped}"'

    # If caller left default "human" but query clearly targets non-human organisms
    # (e.g., helminth parasite proteins), do not force a human-only filter.
    if organism_lc == "human" and _query_has_non_human_hints(query):
        org_clause = None

    # Determine if query is a UniProt accession (e.g. P04637, Q9Y6K9)
    is_accession = len(query) >= 6 and query[0].isalpha() and any(c.isdigit() for c in query)
    species_phrases = _extract_species_phrases(query)
    non_human_hints = _query_has_non_human_hints(query)

    try:
        if is_accession and not " " in query:
            # Direct accession lookup
            resp = _http_get(
                f"https://rest.uniprot.org/uniprotkb/{query}",
                headers={"Accept": "application/json"},
                timeout=15,
                retries=2,
            )
            if resp.status_code == 200:
                entries = [resp.json()]
            else:
                entries = []
        else:
            entries = []

        # If direct lookup failed, search
        if not entries:
            base_query = " ".join((query or "").split())
            compact_query = _keyword_fallback_query(base_query)
            search_candidates = _build_uniprot_search_candidates(
                query=base_query,
                compact_query=compact_query,
                org_clause=org_clause,
            )

            attempted_queries = []
            last_status = None
            matched_query = None
            best_entry = None
            best_score = float("-inf")
            for search_query in search_candidates:
                attempted_queries.append(search_query)
                resp = _http_get(
                    "https://rest.uniprot.org/uniprotkb/search",
                    params={
                        "query": search_query,
                        "format": "json",
                        "size": 10,
                    },
                    headers={"Accept": "application/json"},
                    timeout=15,
                    retries=2,
                )
                last_status = resp.status_code
                if resp.status_code != 200:
                    continue
                data = resp.json()
                hits = data.get("results", [])
                if not hits:
                    continue

                for hit in hits:
                    s = _entry_relevance_score(
                        hit,
                        original_query=query,
                        species_phrases=species_phrases,
                        non_human_hints=non_human_hints,
                    )
                    if s > best_score:
                        best_score = s
                        best_entry = hit
                        matched_query = search_query

                if best_score >= 4.0:
                    break

            if best_entry is not None:
                entries = [best_entry]

            if not entries and last_status not in (None, 200):
                return {
                    "error": f"UniProt search failed (HTTP {last_status})",
                    "summary": f"UniProt search failed for '{query}'",
                    "search_attempts": attempted_queries,
                }

    except Exception as e:
        return {"error": f"UniProt API error: {e}", "summary": f"UniProt query failed for '{query}'"}

    if entries and non_human_hints:
        org_name = str((entries[0].get("organism", {}) or {}).get("scientificName", "")).lower()
        if "homo sapiens" in org_name and (matched_query is not None):
            return {
                "error": (
                    "Only human hits were returned for a non-human/parasite query. "
                    "Please specify organism='any' or a concrete parasite species (taxid/scientific name)."
                ),
                "summary": f"UniProt: no reliable non-human match for '{query}'",
                "search_attempts": attempted_queries if "attempted_queries" in locals() else [],
            }

    if not entries:
        return {
            "error": f"No UniProt entry found for '{query}' (organism: {organism_clean or 'any'})",
            "summary": f"UniProt: no results for '{query}'",
            "search_attempts": attempted_queries if "attempted_queries" in locals() else [],
        }

    entry = entries[0]

    # Extract fields
    accession = entry.get("primaryAccession", "")
    gene_names = []
    for g in entry.get("genes", []):
        gn = g.get("geneName", {}).get("value")
        if gn:
            gene_names.append(gn)
        for syn in g.get("synonyms", []):
            gene_names.append(syn.get("value", ""))

    protein_name = (
        entry.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value", "Unknown")
    )

    seq_info = entry.get("sequence", {})
    seq_length = seq_info.get("length", 0)

    # Function
    function_texts = []
    for c in entry.get("comments", []):
        if c.get("commentType") == "FUNCTION":
            for t in c.get("texts", []):
                function_texts.append(t.get("value", ""))

    # Subcellular location
    subcellular = []
    for c in entry.get("comments", []):
        if c.get("commentType") == "SUBCELLULAR LOCATION":
            for sl in c.get("subcellularLocations", []):
                loc = sl.get("location", {}).get("value", "")
                if loc:
                    subcellular.append(loc)

    # Tissue specificity
    tissue_specificity = ""
    for c in entry.get("comments", []):
        if c.get("commentType") == "TISSUE SPECIFICITY":
            for t in c.get("texts", []):
                tissue_specificity = t.get("value", "")

    # Disease involvement
    diseases = []
    for c in entry.get("comments", []):
        if c.get("commentType") == "DISEASE":
            disease = c.get("disease", {})
            if disease:
                diseases.append({
                    "name": disease.get("diseaseId", ""),
                    "description": disease.get("description", ""),
                    "acronym": disease.get("acronym", ""),
                })

    # Features: domains, GO terms
    features = entry.get("features", [])
    domains = [
        {"name": f.get("description", ""), "type": f.get("type", "")}
        for f in features
        if f.get("type") in ("Domain", "Repeat", "Zinc finger", "Motif")
    ]

    # GO terms from cross-references
    xrefs = entry.get("uniProtKBCrossReferences", [])
    go_terms = []
    pdb_ids = []
    for xref in xrefs:
        db = xref.get("database", "")
        if db == "GO":
            props = {p["key"]: p["value"] for p in xref.get("properties", [])}
            go_terms.append({
                "id": xref.get("id", ""),
                "term": props.get("GoTerm", ""),
                "evidence": props.get("GoEvidenceType", ""),
            })
        elif db == "PDB":
            pdb_ids.append(xref.get("id", ""))

    # Keywords
    keywords = [kw.get("name", "") for kw in entry.get("keywords", [])]

    primary_gene = gene_names[0] if gene_names else query
    n_pdb = len(pdb_ids)

    return {
        "summary": (
            f"UniProt {accession} ({primary_gene}): {protein_name}, "
            f"{seq_length} aa. "
            + (f"{function_texts[0][:120]}... " if function_texts else "")
            + f"{n_pdb} PDB structure(s)."
        ),
        "matched_query": matched_query if "matched_query" in locals() else query,
        "organism_filter": org_clause or "none",
        "accession": accession,
        "gene_names": gene_names,
        "protein_name": protein_name,
        "sequence_length": seq_length,
        "function": function_texts,
        "subcellular_location": subcellular,
        "tissue_specificity": tissue_specificity,
        "diseases": diseases[:10],
        "domains": domains[:20],
        "go_terms": go_terms[:30],
        "pdb_ids": pdb_ids[:30],
        "n_pdb_structures": n_pdb,
        "keywords": keywords,
        "uniprot_url": f"https://www.uniprot.org/uniprot/{accession}",
    }


# ---------------------------------------------------------------------------
# 4. PDB search
# ---------------------------------------------------------------------------

@registry.register(
    name="data_api.pdb_search",
    description="Search RCSB PDB for protein structures by gene name, UniProt ID, or PDB ID",
    category="data_api",
    parameters={
        "query": "Gene name, UniProt accession, or 4-character PDB ID",
        "method": "Optional experimental method filter: 'X-RAY', 'EM', 'NMR'",
        "max_results": "Maximum number of structures to return (default 10)",
    },
    requires_data=[],
    usage_guide="You want to find 3D protein structures for a target — PDB IDs, resolution, method, ligands. Use for structure-based drug design and target assessment.",
)
def pdb_search(query: str, method: str = None, max_results: int = 10, **kwargs) -> dict:
    """Search RCSB PDB for structures using the search and data APIs."""
    query_clean = query.strip()

    # If query looks like a PDB ID (4 chars), fetch directly
    if len(query_clean) == 4 and query_clean.isalnum():
        return _fetch_pdb_entry(query_clean)

    # Build RCSB search query
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"

    # Split multi-term queries into individual search nodes (AND logic)
    terms = query_clean.split()
    fallback_note = ""
    if len(terms) > 1:
        text_nodes = [
            {
                "type": "terminal",
                "service": "full_text",
                "parameters": {"value": term},
            }
            for term in terms
        ]
    else:
        text_nodes = [
            {
                "type": "terminal",
                "service": "full_text",
                "parameters": {"value": query_clean},
            }
        ]

    # Construct search JSON
    query_json = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": text_nodes,
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_results},
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
    }

    # Add method filter if specified
    method_value = None
    if method:
        method_upper = method.upper()
        valid_methods = ("X-RAY DIFFRACTION", "ELECTRON MICROSCOPY", "SOLUTION NMR",
                        "X-RAY", "EM", "NMR")
        if method_upper not in valid_methods:
            return {"error": f"Invalid method '{method}'. Use 'X-RAY', 'EM', or 'NMR'", "summary": f"Invalid PDB method '{method}'"}

        method_map = {
            "X-RAY": "X-RAY DIFFRACTION",
            "EM": "ELECTRON MICROSCOPY",
            "NMR": "SOLUTION NMR",
        }
        method_value = method_map.get(method_upper, method_upper)

        query_json["query"]["nodes"].append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "exptl.method",
                "operator": "exact_match",
                "value": method_value,
            },
        })

    try:
        resp = _http_post(search_url, json=query_json, timeout=15, retries=2)
        if resp.status_code != 200:
            return {
                "error": f"RCSB PDB search failed (HTTP {resp.status_code})",
                "summary": f"PDB search failed for '{query}'",
            }
        data = resp.json()
    except Exception as e:
        return {"error": f"PDB search error: {e}", "summary": f"PDB search failed for '{query}'"}

    total_count = data.get("total_count", 0)
    result_set = data.get("result_set", [])

    if not result_set and len(terms) > 1:
        # Retry with just the first term (likely the protein/gene name)
        fallback_json = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "full_text",
                        "parameters": {"value": terms[0]},
                    }
                ],
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": 0, "rows": max_results},
                "sort": [{"sort_by": "score", "direction": "desc"}],
            },
        }
        if method and method_value:
            # re-add method filter
            fallback_json["query"]["nodes"].append({
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "exptl.method",
                    "operator": "exact_match",
                    "value": method_value,
                },
            })
        try:
            resp2 = _http_post(search_url, json=fallback_json, timeout=15, retries=2)
            if resp2.status_code == 200:
                data2 = resp2.json()
                result_set = data2.get("result_set", [])
                total_count = data2.get("total_count", 0)
                if result_set:
                    fallback_note = f" (broadened from '{query}' to '{terms[0]}')"
        except Exception:
            pass  # Keep original empty result

    if not result_set:
        return {
            "summary": f"No PDB structures found for '{query}'",
            "query": query,
            "total_count": 0,
            "structures": [],
        }

    pdb_ids = [r.get("identifier", "") for r in result_set if r.get("identifier")]

    # Fetch details for each PDB entry
    structures = []
    for pdb_id in pdb_ids[:max_results]:
        try:
            detail_resp = _http_get(
                f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}",
                timeout=10,
                retries=2,
            )
            if detail_resp.status_code != 200:
                structures.append({"pdb_id": pdb_id, "error": "detail fetch failed"})
                continue
            detail = detail_resp.json()

            struct_info = detail.get("struct", {})
            exptl = detail.get("exptl", [{}])[0] if detail.get("exptl") else {}
            cell = detail.get("cell", {})
            rcsb_info = detail.get("rcsb_entry_info", {})

            # Get resolution
            resolution = None
            for refl in detail.get("reflns", []):
                resolution = refl.get("d_resolution_high")
            if resolution is None:
                resolution = rcsb_info.get("resolution_combined", [None])
                resolution = resolution[0] if isinstance(resolution, list) and resolution else resolution

            # Get ligands from nonpolymer entities
            ligands = []
            for entity in detail.get("rcsb_entry_container_identifiers", {}).get("non_polymer_entity_ids", []):
                ligands.append(entity)

            structures.append({
                "pdb_id": pdb_id,
                "title": struct_info.get("title", ""),
                "method": exptl.get("method", ""),
                "resolution": resolution,
                "deposition_date": detail.get("rcsb_accession_info", {}).get("deposit_date", ""),
                "organism": detail.get("rcsb_entry_info", {}).get("deposited_model_count", ""),
                "n_ligands": len(ligands),
            })
        except Exception:
            structures.append({"pdb_id": pdb_id, "error": "detail fetch failed"})

    # Find best resolution
    resolutions = [s["resolution"] for s in structures if s.get("resolution")]
    best_res = min(resolutions) if resolutions else None
    best_id = None
    if best_res is not None:
        for s in structures:
            if s.get("resolution") == best_res:
                best_id = s["pdb_id"]
                break

    method_str = f" ({method})" if method else ""
    best_str = f", best resolution {best_res:.1f}A ({best_id})" if best_res and best_id else ""

    return {
        "summary": (
            f"PDB structures for {query}{method_str}: {total_count} total"
            f"{best_str}{fallback_note}"
        ),
        "query": query,
        "total_count": total_count,
        "n_returned": len(structures),
        "best_resolution": best_res,
        "best_pdb_id": best_id,
        "structures": structures,
    }


def _fetch_pdb_entry(pdb_id: str) -> dict:
    """Fetch a single PDB entry by ID."""
    try:
        resp = _http_get(
            f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}",
            timeout=10,
            retries=2,
        )
        if resp.status_code == 404:
            return {"error": f"PDB entry {pdb_id} not found", "summary": f"No PDB entry for {pdb_id}"}
        if resp.status_code != 200:
            return {"error": f"PDB API returned HTTP {resp.status_code}", "summary": f"PDB API error: HTTP {resp.status_code}"}
        detail = resp.json()
    except Exception as e:
        return {"error": f"PDB API error: {e}", "summary": f"PDB API error: {e}"}
    struct_info = detail.get("struct", {})
    exptl = detail.get("exptl", [{}])[0] if detail.get("exptl") else {}
    rcsb_info = detail.get("rcsb_entry_info", {})

    resolution = None
    for refl in detail.get("reflns", []):
        resolution = refl.get("d_resolution_high")
    if resolution is None:
        res_list = rcsb_info.get("resolution_combined", [])
        resolution = res_list[0] if isinstance(res_list, list) and res_list else None

    return {
        "summary": f"PDB {pdb_id}: {struct_info.get('title', 'N/A')} ({exptl.get('method', 'N/A')}, {resolution or 'N/A'}A)",
        "pdb_id": pdb_id,
        "title": struct_info.get("title", ""),
        "method": exptl.get("method", ""),
        "resolution": resolution,
        "deposition_date": detail.get("rcsb_accession_info", {}).get("deposit_date", ""),
        "total_count": 1,
        "structures": [{
            "pdb_id": pdb_id,
            "title": struct_info.get("title", ""),
            "method": exptl.get("method", ""),
            "resolution": resolution,
        }],
    }


# ---------------------------------------------------------------------------
# 5. Ensembl lookup
# ---------------------------------------------------------------------------

@registry.register(
    name="data_api.ensembl_lookup",
    description="Look up gene information from Ensembl: genomic coordinates, transcripts, cross-references",
    category="data_api",
    parameters={
        "gene": "Gene symbol (e.g. BRCA1) or Ensembl ID (e.g. ENSG00000012048)",
        "species": "Species name (default 'human')",
    },
    requires_data=[],
    usage_guide="You need gene-level genomic information: Ensembl ID, chromosome location, transcripts, biotype, cross-references. Use for gene annotation and ID mapping.",
)
def ensembl_lookup(gene: str, species: str = "human", **kwargs) -> dict:
    """Look up gene information from the Ensembl REST API."""
    ensembl_base = "https://rest.ensembl.org"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    species_map = {
        "human": "homo_sapiens", "mouse": "mus_musculus", "rat": "rattus_norvegicus",
        "zebrafish": "danio_rerio", "drosophila": "drosophila_melanogaster",
    }
    species_name = species_map.get(species.lower(), species.lower().replace(" ", "_"))

    gene_clean = gene.strip()

    # Determine if this is an Ensembl ID or a symbol
    if gene_clean.upper().startswith("ENSG") or gene_clean.upper().startswith("ENSMUSG"):
        # Direct ID lookup
        url = f"{ensembl_base}/lookup/id/{gene_clean}"
        params = {"expand": 1}
    else:
        # Symbol lookup
        url = f"{ensembl_base}/lookup/symbol/{species_name}/{gene_clean}"
        params = {"expand": 1}

    try:
        resp = _http_get(url, params=params, headers=headers, timeout=15, retries=2)
        if resp.status_code == 400:
            return {
                "error": f"Gene '{gene}' not found in Ensembl ({species})",
                "summary": f"Ensembl: gene '{gene}' not found for {species}",
            }
        if resp.status_code != 200:
            return {"error": f"Ensembl API returned HTTP {resp.status_code}", "summary": f"Ensembl API error: HTTP {resp.status_code}"}
        data = resp.json()
    except Exception as e:
        return {"error": f"Ensembl API error: {e}", "summary": f"Ensembl API error: {e}"}
    ensembl_id = data.get("id", "")
    display_name = data.get("display_name", gene)
    description = data.get("description", "")
    biotype = data.get("biotype", "")
    chromosome = data.get("seq_region_name", "")
    start = data.get("start")
    end = data.get("end")
    strand = data.get("strand")

    # Parse transcripts
    transcripts = []
    for t in data.get("Transcript", []):
        transcripts.append({
            "transcript_id": t.get("id", ""),
            "display_name": t.get("display_name", ""),
            "biotype": t.get("biotype", ""),
            "is_canonical": t.get("is_canonical", 0) == 1,
            "length": t.get("length"),
        })

    n_transcripts = len(transcripts)

    # Fetch cross-references (UniProt mapping)
    xrefs = []
    try:
        xref_resp = _http_get(
            f"{ensembl_base}/xrefs/id/{ensembl_id}",
            params={"external_db": "UniProt%"},
            headers=headers,
            timeout=10,
            retries=2,
        )
        if xref_resp.status_code == 200:
            for xref in xref_resp.json():
                xrefs.append({
                    "database": xref.get("dbname", ""),
                    "primary_id": xref.get("primary_id", ""),
                    "display_id": xref.get("display_id", ""),
                })
    except Exception:
        pass

    strand_str = "+" if strand == 1 else "-" if strand == -1 else "?"
    loc_str = f"chr{chromosome}:{start:,}-{end:,} ({strand_str})" if start and end else "unknown"

    return {
        "summary": (
            f"{display_name} ({ensembl_id}): {biotype}, "
            f"{loc_str}, {n_transcripts} transcripts"
        ),
        "ensembl_id": ensembl_id,
        "display_name": display_name,
        "description": description,
        "biotype": biotype,
        "chromosome": chromosome,
        "start": start,
        "end": end,
        "strand": strand,
        "location": loc_str,
        "n_transcripts": n_transcripts,
        "transcripts": transcripts[:20],
        "cross_references": xrefs[:10],
    }


# ---------------------------------------------------------------------------
# 6. NCBI Gene
# ---------------------------------------------------------------------------

@registry.register(
    name="data_api.ncbi_gene",
    description="Query NCBI databases for gene information, ClinVar variants, or dbSNP data",
    category="data_api",
    parameters={
        "query": "Gene symbol (e.g. BRCA1) or NCBI Gene ID (e.g. 672)",
        "database": "Database to query: 'gene', 'clinvar', or 'dbsnp' (default 'gene')",
    },
    requires_data=[],
    usage_guide="You need NCBI gene summaries, ClinVar clinical variant data, or dbSNP information for a gene. Use for gene annotation, variant interpretation, and clinical genetics.",
)
def ncbi_gene(query: str, database: str = "gene", **kwargs) -> dict:
    """Query NCBI E-utilities for gene, ClinVar, or dbSNP data."""
    valid_dbs = ("gene", "clinvar", "dbsnp")
    if database not in valid_dbs:
        return {"error": f"Invalid database '{database}'. Choose from: {', '.join(valid_dbs)}", "summary": f"Invalid NCBI database '{database}'"}

    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # Step 1: Search for the gene/variant
    if database == "gene":
        search_term = f"{query}[Gene Name] AND Homo sapiens[Organism]"
        db = "gene"
    elif database == "clinvar":
        search_term = f"{query}[Gene Name]"
        db = "clinvar"
    else:  # dbsnp
        search_term = f"{query}[Gene Name]"
        db = "snp"

    try:
        search_resp = _http_get(
            f"{base}/esearch.fcgi",
            params={
                "db": db,
                "term": search_term,
                "retmax": 20,
                "retmode": "json",
                "sort": "relevance",
            },
            timeout=15,
            retries=2,
        )
        search_resp.raise_for_status()
        search_data = search_resp.json()
    except Exception as e:
        return {"error": f"NCBI search failed: {e}", "summary": f"NCBI query failed for '{query}'"}

    result = search_data.get("esearchresult", {})
    ids = result.get("idlist", [])
    total_count = int(result.get("count", 0))

    if not ids:
        return {
            "summary": f"No NCBI {database} results for '{query}'",
            "query": query,
            "database": database,
            "total_count": 0,
            "results": [],
        }

    # Step 2: Fetch summaries
    try:
        summary_resp = _http_get(
            f"{base}/esummary.fcgi",
            params={
                "db": db,
                "id": ",".join(ids[:20]),
                "retmode": "json",
            },
            timeout=15,
            retries=2,
        )
        summary_resp.raise_for_status()
        summary_data = summary_resp.json()
    except Exception as e:
        return {"error": f"NCBI summary failed: {e}", "summary": f"NCBI summary lookup failed for '{query}'"}

    results_dict = summary_data.get("result", {})

    if database == "gene":
        gene_results = []
        for gid in ids:
            info = results_dict.get(gid, {})
            if not info or gid == "uids":
                continue
            gene_results.append({
                "gene_id": gid,
                "symbol": info.get("name", ""),
                "description": info.get("description", ""),
                "chromosome": info.get("chromosome", ""),
                "organism": info.get("organism", {}).get("scientificname", ""),
                "aliases": info.get("otheraliases", ""),
                "summary": info.get("summary", ""),
                "gene_type": info.get("geneticSource", ""),
                "map_location": info.get("maplocation", ""),
            })

        top = gene_results[0] if gene_results else {}
        return {
            "summary": (
                f"NCBI Gene {top.get('gene_id', '')} ({top.get('symbol', query)}): "
                f"{top.get('description', 'N/A')}, "
                f"chr{top.get('chromosome', '?')}, "
                f"{total_count} total ClinVar variants"
            ),
            "query": query,
            "database": "gene",
            "total_count": total_count,
            "genes": gene_results,
        }

    elif database == "clinvar":
        variants = []
        for vid in ids:
            info = results_dict.get(vid, {})
            if not info or vid == "uids":
                continue
            variants.append({
                "uid": vid,
                "title": info.get("title", ""),
                "clinical_significance": info.get("clinical_significance", {}).get("description", ""),
                "gene_sort": info.get("gene_sort", ""),
                "variation_set": info.get("variation_set", []),
                "obj_type": info.get("obj_type", ""),
            })

        return {
            "summary": f"ClinVar for {query}: {total_count} total variants, showing {len(variants)}",
            "query": query,
            "database": "clinvar",
            "total_count": total_count,
            "variants": variants,
        }

    else:  # dbsnp
        snps = []
        for sid in ids:
            info = results_dict.get(sid, {})
            if not info or sid == "uids":
                continue
            snps.append({
                "uid": sid,
                "snp_id": info.get("snp_id", sid),
                "snp_class": info.get("snp_class", ""),
                "global_maf": info.get("global_mafs", []),
                "genes": info.get("genes", []),
                "clinical_significance": info.get("clinical_significance", ""),
            })

        return {
            "summary": f"dbSNP for {query}: {total_count} total SNPs, showing {len(snps)}",
            "query": query,
            "database": "dbsnp",
            "total_count": total_count,
            "snps": snps,
        }


# ---------------------------------------------------------------------------
# 7. ChEMBL advanced
# ---------------------------------------------------------------------------

@registry.register(
    name="data_api.chembl_advanced",
    description="Advanced ChEMBL queries: compound details, target activity statistics, mechanisms, drug indications",
    category="data_api",
    parameters={
        "query": "Compound name/ChEMBL ID, target gene, or drug name",
        "search_type": "Query type: 'compound', 'target_activities', 'mechanism', or 'drug_indication' (default 'compound')",
    },
    requires_data=[],
    usage_guide="You want detailed ChEMBL data: full compound properties, aggregated bioactivity statistics for a target (min/max/median IC50), drug mechanisms of action, or approved indications. More detailed than literature.chembl_query.",
)
def chembl_advanced(query: str, search_type: str = "compound", **kwargs) -> dict:
    """Advanced ChEMBL REST API queries with aggregated statistics."""
    valid_types = ("compound", "target_activities", "mechanism", "drug_indication")
    if search_type not in valid_types:
        return {"error": f"Invalid search_type '{search_type}'. Choose from: {', '.join(valid_types)}", "summary": f"Invalid ChEMBL search type '{search_type}'"}

    chembl_base = "https://www.ebi.ac.uk/chembl/api/data"
    headers = {"Accept": "application/json"}

    if search_type == "compound":
        return _chembl_compound_search(query, chembl_base, headers)
    elif search_type == "target_activities":
        return _chembl_target_activities(query, chembl_base, headers)
    elif search_type == "mechanism":
        return _chembl_mechanism(query, chembl_base, headers)
    else:  # drug_indication
        return _chembl_drug_indication(query, chembl_base, headers)


def _chembl_compound_search(query: str, base: str, headers: dict) -> dict:
    """Search ChEMBL for a compound with full property details."""
    try:
        resp = _http_get(
            f"{base}/molecule/search.json",
            params={"q": query, "limit": 5},
            headers=headers,
            timeout=15,
            retries=2,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"error": f"ChEMBL compound search failed: {e}", "summary": f"ChEMBL compound search failed: {e}"}
    molecules = data.get("molecules", [])
    if not molecules:
        return {
            "summary": f"No ChEMBL compounds found for '{query}'",
            "query": query,
            "compounds": [],
        }

    compounds = []
    for mol in molecules:
        props = mol.get("molecule_properties", {}) or {}
        structs = mol.get("molecule_structures", {}) or {}
        chembl_id = mol.get("molecule_chembl_id", "")

        compounds.append({
            "chembl_id": chembl_id,
            "pref_name": mol.get("pref_name", ""),
            "molecule_type": mol.get("molecule_type", ""),
            "max_phase": mol.get("max_phase", 0),
            "oral": mol.get("oral", False),
            "parenteral": mol.get("parenteral", False),
            "topical": mol.get("topical", False),
            "natural_product": mol.get("natural_product", -1),
            "canonical_smiles": structs.get("canonical_smiles", ""),
            "inchi_key": structs.get("standard_inchi_key", ""),
            "molecular_weight": props.get("full_mwt"),
            "alogp": props.get("alogp"),
            "hba": props.get("hba"),
            "hbd": props.get("hbd"),
            "psa": props.get("psa"),
            "rtb": props.get("rtb"),
            "ro5_violations": props.get("num_ro5_violations"),
            "aromatic_rings": props.get("aromatic_rings"),
            "heavy_atoms": props.get("heavy_atoms"),
            "qed_weighted": props.get("qed_weighted"),
        })

    top = compounds[0]
    return {
        "summary": (
            f"ChEMBL compound {top['chembl_id']} ({top['pref_name'] or query}): "
            f"MW={top['molecular_weight'] or 'N/A'}, ALogP={top['alogp'] or 'N/A'}, "
            f"max phase {top['max_phase']}"
        ),
        "query": query,
        "n_results": len(compounds),
        "compounds": compounds,
    }


def _chembl_target_activities(query: str, base: str, headers: dict) -> dict:
    """Get aggregated bioactivity statistics for a target."""
    # Find the target
    try:
        tgt_resp = _http_get(
            f"{base}/target/search.json",
            params={"q": query, "limit": 5},
            headers=headers,
            timeout=15,
            retries=2,
        )
        tgt_resp.raise_for_status()
        tgt_data = tgt_resp.json()
    except Exception as e:
        return {"error": f"ChEMBL target search failed: {e}", "summary": f"ChEMBL target search failed: {e}"}
    targets = tgt_data.get("targets", [])
    if not targets:
        return {"summary": f"No ChEMBL target found for '{query}'", "query": query}

    # Prefer human SINGLE PROTEIN
    target = None
    for t in targets:
        if t.get("organism") == "Homo sapiens" and t.get("target_type") == "SINGLE PROTEIN":
            target = t
            break
    if not target:
        target = targets[0]

    chembl_target_id = target.get("target_chembl_id", "")
    target_name = target.get("pref_name", query)

    # Fetch activities
    try:
        act_resp = _http_get(
            f"{base}/activity.json",
            params={
                "target_chembl_id": chembl_target_id,
                "limit": 100,
                "standard_type__in": "IC50,Ki,Kd,EC50",
            },
            headers=headers,
            timeout=15,
            retries=2,
        )
        act_resp.raise_for_status()
        act_data = act_resp.json()
    except Exception as e:
        return {"error": f"ChEMBL activity query failed: {e}", "summary": f"ChEMBL activity query failed: {e}"}
    activities = act_data.get("activities", [])

    # Aggregate statistics
    import statistics

    by_type = {}
    unique_molecules = set()
    for act in activities:
        mol_id = act.get("molecule_chembl_id", "")
        unique_molecules.add(mol_id)
        std_type = act.get("standard_type", "")
        std_value = act.get("standard_value")
        if std_value is not None:
            try:
                val = float(std_value)
                by_type.setdefault(std_type, []).append(val)
            except (ValueError, TypeError):
                pass

    stats = {}
    for activity_type, values in by_type.items():
        sorted_vals = sorted(values)
        stats[activity_type] = {
            "count": len(values),
            "min_nM": round(min(values), 2),
            "max_nM": round(max(values), 2),
            "median_nM": round(statistics.median(values), 2),
            "mean_nM": round(statistics.mean(values), 2),
        }

    total_activities = sum(s["count"] for s in stats.values())
    median_str = ""
    if "IC50" in stats:
        median_str = f", median IC50 = {stats['IC50']['median_nM']:.0f} nM"

    return {
        "summary": (
            f"ChEMBL target {chembl_target_id} ({target_name}): "
            f"{total_activities} activities, "
            f"{len(unique_molecules)} unique compounds"
            f"{median_str}"
        ),
        "query": query,
        "target_chembl_id": chembl_target_id,
        "target_name": target_name,
        "organism": target.get("organism", ""),
        "target_type": target.get("target_type", ""),
        "n_unique_compounds": len(unique_molecules),
        "n_activities": total_activities,
        "activity_statistics": stats,
    }


def _chembl_mechanism(query: str, base: str, headers: dict) -> dict:
    """Look up drug mechanisms of action."""
    try:
        resp = _http_get(
            f"{base}/mechanism.json",
            params={"molecule_chembl_id": query, "limit": 20},
            headers=headers,
            timeout=15,
            retries=2,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"error": f"ChEMBL mechanism query failed: {e}", "summary": f"ChEMBL mechanism query failed: {e}"}
    mechanisms = data.get("mechanisms", [])

    # If no results by molecule ID, try searching by name
    if not mechanisms:
        try:
            mol_resp = _http_get(
                f"{base}/molecule/search.json",
                params={"q": query, "limit": 1},
                headers=headers,
                timeout=10,
                retries=2,
            )
            mol_resp.raise_for_status()
            mol_data = mol_resp.json()
            mols = mol_data.get("molecules", [])
            if mols:
                mol_id = mols[0].get("molecule_chembl_id", "")
                resp2 = _http_get(
                    f"{base}/mechanism.json",
                    params={"molecule_chembl_id": mol_id, "limit": 20},
                    headers=headers,
                    timeout=10,
                    retries=2,
                )
                resp2.raise_for_status()
                mechanisms = resp2.json().get("mechanisms", [])
        except Exception:
            pass

    if not mechanisms:
        return {
            "summary": f"No mechanisms of action found in ChEMBL for '{query}'",
            "query": query,
            "mechanisms": [],
        }

    parsed = []
    for mech in mechanisms:
        parsed.append({
            "mechanism": mech.get("mechanism_of_action", ""),
            "action_type": mech.get("action_type", ""),
            "target_name": mech.get("target_chembl_id", ""),
            "molecule_chembl_id": mech.get("molecule_chembl_id", ""),
            "max_phase": mech.get("max_phase"),
            "direct_interaction": mech.get("direct_interaction"),
        })

    return {
        "summary": (
            f"ChEMBL mechanisms for {query}: {len(parsed)} mechanism(s). "
            + "; ".join(m["mechanism"] for m in parsed[:3])
        ),
        "query": query,
        "n_mechanisms": len(parsed),
        "mechanisms": parsed,
    }


def _chembl_drug_indication(query: str, base: str, headers: dict) -> dict:
    """Look up approved drug indications."""
    # Resolve molecule ID
    mol_id = query
    try:
        if not query.upper().startswith("CHEMBL"):
            mol_resp = _http_get(
                f"{base}/molecule/search.json",
                params={"q": query, "limit": 1},
                headers=headers,
                timeout=10,
                retries=2,
            )
            mol_resp.raise_for_status()
            mols = mol_resp.json().get("molecules", [])
            if mols:
                mol_id = mols[0].get("molecule_chembl_id", "")
    except Exception:
        pass

    try:
        resp = _http_get(
            f"{base}/drug_indication.json",
            params={"molecule_chembl_id": mol_id, "limit": 30},
            headers=headers,
            timeout=15,
            retries=2,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"error": f"ChEMBL indication query failed: {e}", "summary": f"ChEMBL indication query failed: {e}"}
    indications = data.get("drug_indications", [])
    if not indications:
        return {
            "summary": f"No drug indications found in ChEMBL for '{query}'",
            "query": query,
            "indications": [],
        }

    parsed = []
    for ind in indications:
        parsed.append({
            "indication": ind.get("mesh_heading", ""),
            "mesh_id": ind.get("mesh_id", ""),
            "efo_id": ind.get("efo_id", ""),
            "max_phase": ind.get("max_phase_for_ind"),
            "molecule_chembl_id": ind.get("molecule_chembl_id", ""),
        })

    approved = [p for p in parsed if p.get("max_phase") == 4]
    return {
        "summary": (
            f"ChEMBL indications for {query} ({mol_id}): "
            f"{len(parsed)} total, {len(approved)} approved. "
            + "; ".join(p["indication"] for p in parsed[:5])
        ),
        "query": query,
        "molecule_chembl_id": mol_id,
        "n_indications": len(parsed),
        "n_approved": len(approved),
        "indications": parsed,
    }


# ---------------------------------------------------------------------------
# 8. Drug information lookup (via PubChem)
# ---------------------------------------------------------------------------

@registry.register(
    name="data_api.drug_info",
    description="Look up comprehensive drug information: pharmacology, properties, interactions, indications",
    category="data_api",
    parameters={
        "query": "Drug name (e.g. 'imatinib') or compound name",
        "include": "Information to include: list of 'pharmacology', 'interactions', 'properties' (default ['pharmacology', 'interactions'])",
    },
    requires_data=[],
    usage_guide="You want drug pharmacology, properties, and interaction data. Uses PubChem PUG REST and PUG View APIs for comprehensive drug information.",
)
def drug_info(query: str, include: list = None, **kwargs) -> dict:
    """Look up drug information via PubChem REST API.

    Uses PubChem PUG REST and PUG View APIs to retrieve drug properties,
    pharmacology, and interaction data.
    """
    if include is None:
        include = ["pharmacology", "interactions"]

    # Normalize drug name
    raw_query = query
    query = _normalize_drug_query(query)
    if not query:
        return {
            "error": "Drug query is required",
            "summary": "PubChem: query cannot be empty",
        }

    pug_base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    pugview_base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"

    def _query_candidates(text: str) -> list[str]:
        candidates = []
        seen = set()

        def _add(candidate: str):
            c = " ".join((candidate or "").split()).strip()
            if not c or c.lower() in seen:
                return
            seen.add(c.lower())
            candidates.append(c)

        _add(text)
        for part in re.split(r"[;,/|()]|\bor\b|\band\b", text, flags=re.IGNORECASE):
            _add(part)
        for token in text.split():
            cleaned = token.strip(" ,;:/|()[]{}")
            if len(cleaned) >= 3 and re.search(r"[A-Za-z]", cleaned):
                _add(cleaned)
        return candidates

    # Step 1: Resolve drug name to CID (with alias/fallback attempts)
    import urllib.parse

    cid = None
    resolved_query = raw_query
    lookup_errors = []
    for candidate in _query_candidates(raw_query):
        encoded_query = urllib.parse.quote(candidate, safe="")
        try:
            resp = _http_get(
                f"{pug_base}/compound/name/{encoded_query}/cids/JSON",
                timeout=10,
                retries=2,
            )
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            cid_data = resp.json()
        except Exception as e:
            lookup_errors.append(f"{candidate}: {e}")
            continue

        cids = cid_data.get("IdentifierList", {}).get("CID", [])
        if cids:
            cid = cids[0]
            resolved_query = candidate
            break

    if cid is None:
        if lookup_errors:
            return {
                "error": f"PubChem CID lookup failed: {lookup_errors[0]}",
                "summary": f"PubChem CID lookup failed for '{raw_query}'",
                "tried_queries": _query_candidates(raw_query)[:5],
            }
        return {
            "error": f"Drug '{raw_query}' not found in PubChem",
            "summary": f"PubChem: no compound found for '{raw_query}'",
            "tried_queries": _query_candidates(raw_query)[:5],
        }

    # Step 2: Get compound properties
    properties = {}
    try:
        props_resp = _http_get(
            f"{pug_base}/compound/cid/{cid}/property/"
            "MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,"
            "XLogP,ExactMass,TPSA,HBondDonorCount,HBondAcceptorCount,"
            "RotatableBondCount,HeavyAtomCount,Complexity,InChIKey/JSON",
            timeout=10,
            retries=2,
        )
        if props_resp.status_code == 200:
            prop_table = props_resp.json().get("PropertyTable", {}).get("Properties", [])
            if prop_table:
                properties = prop_table[0]
    except Exception:
        pass

    # Step 3: Get drug/medication information from PUG View
    pharmacology = {}
    interactions = []
    drug_info = {}

    if "pharmacology" in include or "interactions" in include:
        try:
            view_resp = _http_get(
                f"{pugview_base}/data/compound/{cid}/JSON",
                params={"heading": "Drug and Medication Information"},
                timeout=15,
                retries=2,
            )
            if view_resp.status_code == 200:
                view_data = view_resp.json()
                record = view_data.get("Record", {})
                sections = record.get("Section", [])

                for section in sections:
                    heading = section.get("TOCHeading", "")
                    for subsection in section.get("Section", []):
                        sub_heading = subsection.get("TOCHeading", "")
                        info_list = subsection.get("Information", [])

                        if sub_heading == "Drug Indication":
                            for info in info_list:
                                val = info.get("Value", {}).get("StringWithMarkup", [])
                                if val:
                                    drug_info["indication"] = val[0].get("String", "")[:500]

                        elif sub_heading == "Mechanism of Action":
                            for info in info_list:
                                val = info.get("Value", {}).get("StringWithMarkup", [])
                                if val:
                                    pharmacology["mechanism_of_action"] = val[0].get("String", "")[:500]

                        elif sub_heading == "Pharmacology":
                            for info in info_list:
                                val = info.get("Value", {}).get("StringWithMarkup", [])
                                if val:
                                    pharmacology["pharmacology"] = val[0].get("String", "")[:500]

                        elif sub_heading == "Absorption":
                            for info in info_list:
                                val = info.get("Value", {}).get("StringWithMarkup", [])
                                if val:
                                    pharmacology["absorption"] = val[0].get("String", "")[:300]

                        elif "Drug Interaction" in sub_heading or "Drug-Drug" in sub_heading:
                            for info in info_list:
                                val = info.get("Value", {}).get("StringWithMarkup", [])
                                if val:
                                    interactions.append(val[0].get("String", "")[:200])
        except Exception:
            pass

    # Step 4: Get synonyms for the drug
    synonyms = []
    try:
        syn_resp = _http_get(
            f"{pug_base}/compound/cid/{cid}/synonyms/JSON",
            timeout=10,
            retries=2,
        )
        if syn_resp.status_code == 200:
            syn_list = syn_resp.json().get("InformationList", {}).get("Information", [])
            if syn_list:
                synonyms = syn_list[0].get("Synonym", [])[:15]
    except Exception:
        pass

    # Find DrugBank ID in synonyms
    drugbank_id = ""
    for syn in synonyms:
        if syn.upper().startswith("DB") and len(syn) == 7 and syn[2:].isdigit():
            drugbank_id = syn
            break

    mw = properties.get("MolecularWeight", "N/A")
    formula = properties.get("MolecularFormula", "N/A")
    smiles = properties.get("CanonicalSMILES", "N/A")
    mechanism = pharmacology.get("mechanism_of_action", "N/A")

    drugbank_str = f" ({drugbank_id})" if drugbank_id else ""
    mech_short = mechanism[:80] + "..." if len(mechanism) > 80 else mechanism
    resolved_note = ""
    if resolved_query.lower() != raw_query.lower():
        resolved_note = f" [resolved as '{resolved_query}']"

    return {
        "summary": (
            f"{raw_query}{resolved_note}{drugbank_str}: {mech_short}, "
            f"MW {mw}, {len(interactions)} known drug interactions."
        ),
        "query": raw_query,
        "resolved_query": resolved_query,
        "cid": cid,
        "drugbank_id": drugbank_id,
        "properties": {
            "molecular_formula": formula,
            "molecular_weight": mw,
            "canonical_smiles": smiles,
            "isomeric_smiles": properties.get("IsomericSMILES", ""),
            "xlogp": properties.get("XLogP"),
            "tpsa": properties.get("TPSA"),
            "hbd": properties.get("HBondDonorCount"),
            "hba": properties.get("HBondAcceptorCount"),
            "rotatable_bonds": properties.get("RotatableBondCount"),
            "inchi_key": properties.get("InChIKey", ""),
        },
        "pharmacology": pharmacology,
        "drug_info": drug_info,
        "interactions": interactions[:20],
        "n_interactions": len(interactions),
        "synonyms": synonyms,
        "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
    }


@registry.register(
    name="data_api.bindingdb_search",
    description="Search BindingDB for binding affinity data (Ki/IC50/Kd/EC50) by target name or compound",
    category="data_api",
    parameters={
        "query": "Search term (target protein name, gene symbol, or compound name)",
        "search_type": "Type of search: 'target' (default) or 'compound'",
        "activity_type": "Filter by activity type: 'Ki', 'IC50', 'Kd', 'EC50', or None for all (optional)",
        "max_results": "Maximum number of results to return (default 20)",
    },
    requires_data=[],
    usage_guide="You need binding affinity data for a drug-target interaction — Ki, IC50, Kd, or EC50 values. Use for SAR analysis, comparing compound potencies against a target, or finding known ligands for a target.",
)
def bindingdb_search(query: str, search_type: str = "target", activity_type: str = None, max_results: int = 20, **kwargs) -> dict:
    """Search BindingDB REST API for binding affinity data."""
    import xml.etree.ElementTree as ET
    from ct.tools.http_client import request

    query = (query or "").strip()
    if not query:
        return {"error": "Search query is required", "summary": "BindingDB search requires a query term"}

    max_results = max(1, min(int(max_results or 20), 100))
    search_type = (search_type or "target").lower()

    base_url = "https://bindingdb.org/axis2/services/BDBService"

    if search_type == "compound":
        # Search by compound name → get monomer ID first, then affinities
        endpoint = f"{base_url}/getLigandsByName"
        params = {"name": query}
    else:
        # Search by target name
        endpoint = f"{base_url}/getBindingsByTarget"
        params = {"target": query, "response": "application/xml"}

    resp, error = request(
        "GET", endpoint,
        params=params,
        headers={"Accept": "application/xml"},
        timeout=30, retries=2,
        raise_for_status=False,
    )
    if error:
        return {"error": f"BindingDB query failed: {error}", "summary": f"BindingDB search failed for '{query}': {error}"}

    if resp.status_code != 200:
        # Fallback: try the other search type or simpler endpoint
        fallback_endpoint = f"{base_url}/getLigandsByName" if search_type == "target" else f"{base_url}/getBindingsByTarget"
        fallback_params = {"name": query} if search_type == "target" else {"target": query}
        resp, error = request(
            "GET", fallback_endpoint,
            params=fallback_params,
            headers={"Accept": "application/xml"},
            timeout=30, retries=2,
            raise_for_status=False,
        )
        if error or resp.status_code != 200:
            return {
                "error": f"BindingDB returned HTTP {resp.status_code if resp else 'N/A'}",
                "summary": f"No BindingDB results for '{query}' (search_type={search_type})",
            }

    # Parse XML response
    try:
        text = resp.text or ""
        if not text.strip():
            return {"summary": f"No BindingDB results for '{query}'", "query": query, "records": [], "n_records": 0}
        root = ET.fromstring(text)
    except ET.ParseError:
        return {"error": "Failed to parse BindingDB XML response", "summary": f"Invalid response from BindingDB for '{query}'"}

    # Extract binding records — BindingDB uses various XML schemas
    records = []
    ns = {"bdb": "http://ws.bindingdb.org/xsd"}

    # Try multiple possible element paths
    for entry_tag in ["affinities", "affinity", "bdb:affinities", "bdb:affinity", "hit", "bdb:hit"]:
        for entry in root.iter(entry_tag.split(":")[-1]):
            record = {}
            for child in entry:
                tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                record[tag] = (child.text or "").strip()

            if not record:
                continue

            parsed = {
                "monomer_id": record.get("monomerid", record.get("MonomerID", "")),
                "target_name": record.get("target", record.get("Target_Name", "")),
                "ki": record.get("ki", record.get("Ki", "")),
                "ic50": record.get("ic50", record.get("IC50", "")),
                "kd": record.get("kd", record.get("Kd", "")),
                "ec50": record.get("ec50", record.get("EC50", "")),
                "smiles": record.get("smiles", record.get("SMILES", record.get("Ligand_SMILES", ""))),
                "ligand_name": record.get("ligand_name", record.get("Ligand_Name", "")),
                "doi": record.get("doi", record.get("Article_DOI", "")),
                "pmid": record.get("pmid", record.get("PMID", "")),
            }

            # Filter by activity type if specified
            if activity_type:
                atype = activity_type.lower()
                if atype == "ki" and not parsed["ki"]:
                    continue
                elif atype == "ic50" and not parsed["ic50"]:
                    continue
                elif atype == "kd" and not parsed["kd"]:
                    continue
                elif atype == "ec50" and not parsed["ec50"]:
                    continue

            # Clean empty strings
            parsed = {k: v for k, v in parsed.items() if v}
            if parsed:
                records.append(parsed)
                if len(records) >= max_results:
                    break

    # If no structured records found, try a simpler flat text parse
    if not records:
        for elem in root.iter():
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag.lower() in ("return", "entry", "result") and elem.text:
                records.append({"raw_data": elem.text.strip()})
                if len(records) >= max_results:
                    break

    # Compute activity stats
    activity_values = {"ki": [], "ic50": [], "kd": [], "ec50": []}
    for r in records:
        for atype in activity_values:
            val_str = r.get(atype, "")
            if val_str:
                try:
                    # Handle ">" and "<" prefixed values
                    clean = val_str.replace(">", "").replace("<", "").replace("=", "").strip()
                    activity_values[atype].append(float(clean))
                except (ValueError, TypeError):
                    pass

    activity_stats = {}
    import numpy as _np
    for atype, vals in activity_values.items():
        if vals:
            activity_stats[atype] = {
                "n": len(vals),
                "min_nM": round(min(vals), 2),
                "max_nM": round(max(vals), 2),
                "median_nM": round(float(_np.median(vals)), 2),
            }

    records_str = f"{len(records)} binding records"
    stats_str = ", ".join(f"{k}: n={v['n']}, median={v['median_nM']}nM" for k, v in activity_stats.items())

    return {
        "summary": f"BindingDB '{query}': {records_str}. {stats_str}" if activity_stats else f"BindingDB '{query}': {records_str}",
        "query": query,
        "search_type": search_type,
        "n_records": len(records),
        "activity_stats": activity_stats,
        "records": records[:max_results],
    }


@registry.register(
    name="data_api.ttd_search",
    description="Search the Therapeutic Target Database (TTD) for drug-target-disease relationships",
    category="data_api",
    parameters={
        "query": "Search term (target name, drug name, or disease name)",
        "search_type": "Type of search: 'target' (default), 'drug', or 'disease'",
    },
    requires_data=[],
    usage_guide="You want curated drug-target-disease relationships. TTD has 2,025 targets and 17,816 drugs with clinical status. Complements ChEMBL (bioactivity data) and Open Targets (genetic evidence) with manually curated therapeutic target annotations and drug development status.",
)
def ttd_search(query: str, search_type: str = "target", **kwargs) -> dict:
    """Search the Therapeutic Target Database for drug-target relationships."""
    query = (query or "").strip()
    if not query:
        return {"error": "Search query is required", "summary": "TTD search requires a query term"}

    search_type = (search_type or "target").lower()
    if search_type not in ("target", "drug", "disease"):
        return {"error": f"Invalid search_type: {search_type}. Use 'target', 'drug', or 'disease'", "summary": f"Invalid search type: {search_type}"}

    # TTD API endpoint
    base_url = "https://db.idrblab.net/ttd/api/v1"

    if search_type == "target":
        url = f"{base_url}/target/search"
    elif search_type == "drug":
        url = f"{base_url}/drug/search"
    else:
        url = f"{base_url}/disease/search"

    try:
        resp = _http_get(url, params={"query": query, "limit": 20}, timeout=20, retries=2)
    except Exception as e:
        # Fallback: try the simple search endpoint
        try:
            fallback_url = f"{base_url}/search"
            resp = _http_get(fallback_url, params={"q": query, "type": search_type, "limit": 20}, timeout=20, retries=2)
        except Exception as e2:
            return {"error": f"TTD search failed: {e2}", "summary": f"TTD search failed for '{query}': {e2}"}

    if resp.status_code != 200:
        return {"error": f"TTD returned HTTP {resp.status_code}", "summary": f"TTD search failed (HTTP {resp.status_code})"}

    try:
        data = resp.json()
    except Exception:
        return {"error": "Invalid JSON from TTD", "summary": "TTD returned invalid response"}

    # Parse results — TTD response format varies
    results = []
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("results", data.get("data", data.get("targets", data.get("drugs", []))))
        if not isinstance(entries, list):
            entries = [data]
    else:
        entries = []

    for entry in entries[:20]:
        if not isinstance(entry, dict):
            continue
        result = {
            "id": entry.get("ttd_id", entry.get("id", "")),
            "name": entry.get("name", entry.get("target_name", entry.get("drug_name", ""))),
            "type": entry.get("type", search_type),
            "uniprot_id": entry.get("uniprot_id", entry.get("uniprot", "")),
            "gene": entry.get("gene_name", entry.get("gene", "")),
            "status": entry.get("clinical_status", entry.get("status", "")),
            "drugs": entry.get("drugs", entry.get("associated_drugs", [])),
            "diseases": entry.get("diseases", entry.get("indications", [])),
            "description": entry.get("description", entry.get("function", ""))[:300],
        }
        results.append(result)

    if not results:
        return {
            "summary": f"No TTD results for '{query}' (search_type={search_type})",
            "query": query,
            "search_type": search_type,
            "results": [],
            "n_results": 0,
        }

    top = results[0]
    return {
        "summary": (
            f"TTD {search_type} search for '{query}': {len(results)} result(s). "
            f"Top: {top['name']} ({top['id']})"
        ),
        "query": query,
        "search_type": search_type,
        "n_results": len(results),
        "results": results,
    }
