"""
Safety profiling tools: anti-target screening, multi-modal safety classification, SALL4 risk.

References crews-glue-discovery/analysis/safety_profile.md for classification logic
and anti-target lists.
"""

import pandas as pd
import numpy as np
from ct.tools import registry
from ct.tools.http_client import request_json


# UniProt accession → gene symbol mapping for safety-relevant proteins.
# The proteomics matrix uses UniProt IDs as row index; all gene-symbol
# lookups must go through this mapping.
UNIPROT_TO_GENE = {
    # SALL family (teratogenicity)
    "Q9UJQ4": "SALL4", "Q9NSC2": "SALL1", "Q9Y467": "SALL2", "Q8N3A9": "SALL3",
    # IKZF family (heme TFs / CRBN substrates)
    "Q13422": "IKZF1", "Q9UKT9": "IKZF3", "Q9H2S1": "IKZF4",
    "Q96PU5": "IKZF2", "Q9H193": "IKZF5",
    # Other CRBN substrates
    "P15170": "GSPT1", "Q8IYD1": "GSPT2", "P48729": "CSNK1A1", "Q96SW2": "ZFP91",
    # Tumor suppressors
    "P04637": "TP53", "P06400": "RB1", "P60484": "PTEN", "P25054": "APC",
    "P38398": "BRCA1", "P51587": "BRCA2", "P40337": "VHL",
    "P21359": "NF1", "P35240": "NF2",
    "P42771": "CDKN2A", "P42772": "CDKN2B", "P19544": "WT1",
    "Q13315": "SMAD4", "Q15831": "STK11", "Q969H0": "FBXW7", "Q92560": "BAP1",
    "O14497": "ARID1A", "Q8NFD5": "ARID1B", "Q68CP9": "ARID2",
    "O14686": "KMT2D", "Q8NEZ4": "KMT2C", "Q9BYW2": "SETD2",
    "Q01196": "RUNX1", "Q13761": "RUNX3", "P23771": "GATA3",
    "P10914": "IRF1", "O15524": "SOCS1",
    # Heme TFs
    "P15976": "GATA1", "P23769": "GATA2", "P17542": "TAL1",
    "P17947": "SPI1", "P49715": "CEBPA", "P17676": "CEBPB",
    "P10242": "MYB", "P41212": "ETV6", "Q01543": "FLI1",
    # TP63 (teratogenic)
    "Q9H3D4": "TP63",
}
GENE_TO_UNIPROT = {v: k for k, v in UNIPROT_TO_GENE.items()}


def _gene_ids(gene_set, all_proteins):
    """Return the subset of protein IDs (UniProt or gene symbol) present in *all_proteins*
    that correspond to any gene in *gene_set*."""
    # Direct gene-symbol matches (in case the index already uses symbols)
    hits = gene_set & all_proteins
    # UniProt-ID matches
    for uid, gene in UNIPROT_TO_GENE.items():
        if gene in gene_set and uid in all_proteins:
            hits.add(uid)
    return hits


def _display_name(protein_id):
    """Return a human-readable name: 'GENE (UNIPROT)' when a mapping exists."""
    gene = UNIPROT_TO_GENE.get(protein_id)
    if gene:
        return f"{gene} ({protein_id})"
    return protein_id


# Known tumor suppressor genes (anti-targets for degradation)
TUMOR_SUPPRESSORS = {
    "TP53", "RB1", "PTEN", "APC", "BRCA1", "BRCA2", "VHL", "NF1", "NF2",
    "CDKN2A", "CDKN2B", "WT1", "SMAD4", "STK11", "FBXW7", "BAP1",
    "ARID1A", "ARID1B", "ARID2", "KMT2D", "KMT2C", "SETD2",
    "RUNX1", "RUNX3", "GATA3", "IRF1", "SOCS1",
}

# Essential hematopoietic transcription factors (high-risk degradation targets)
HEME_TFS = {
    "IKZF1", "IKZF3", "IKZF4", "GATA1", "GATA2", "TAL1", "RUNX1",
    "SPI1", "CEBPA", "CEBPB", "MYB", "ETV6", "FLI1",
}

# Known teratogenicity-associated CRBN substrates
TERATOGENIC_SUBSTRATES = {
    "SALL4", "SALL1", "SALL3",  # limb development TFs
    "p63", "TP63",  # epithelial development
}

# Known CRBN neosubstrates (degraded by IMiDs/molecular glues)
CRBN_SUBSTRATES = {
    "IKZF1", "IKZF3", "CK1A", "CSNK1A1", "GSPT1", "GSPT2",
    "ZFP91", "AIOLOS", "IKAROS",
}

_OPENFDA_DRUG_EVENT_URL = "https://api.fda.gov/drug/event.json"
_OPENFDA_DRUG_LABEL_URL = "https://api.fda.gov/drug/label.json"


def _openfda_escape(term: str) -> str:
    """Escape a value for openFDA search string usage."""
    return str(term or "").replace("\\", "\\\\").replace('"', '\\"').strip()


def _openfda_total(search: str = "") -> tuple[int | None, str | None]:
    """Return total matching records from openFDA endpoint."""
    params = {"limit": "1"}
    if search:
        params["search"] = search

    data, error = request_json(
        "GET",
        _OPENFDA_DRUG_EVENT_URL,
        params=params,
        timeout=20,
        retries=2,
    )
    if error:
        return None, error

    total = data.get("meta", {}).get("results", {}).get("total")
    try:
        return int(total), None
    except Exception:
        return None, "openFDA response missing total count"


def _faers_signal_metrics(
    a: int,
    b: int,
    c: int,
    d: int,
    *,
    min_case_count: int = 3,
) -> dict:
    """Compute basic disproportionality metrics (PRR/ROR/chi-square)."""
    import math

    a = max(int(a), 0)
    b = max(int(b), 0)
    c = max(int(c), 0)
    d = max(int(d), 0)

    # Haldane-Anscombe correction stabilizes estimates when cells are zero.
    ac, bc, cc, dc = [x + 0.5 for x in (a, b, c, d)]

    prr = (ac / (ac + bc)) / (cc / (cc + dc))
    ror = (ac / bc) / (cc / dc)
    se_log_ror = math.sqrt((1 / ac) + (1 / bc) + (1 / cc) + (1 / dc))
    ror_ci95_lower = math.exp(math.log(ror) - 1.96 * se_log_ror)
    ror_ci95_upper = math.exp(math.log(ror) + 1.96 * se_log_ror)

    total = a + b + c + d
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    chi_square = ((total * ((a * d - b * c) ** 2)) / denom) if denom > 0 else 0.0

    # Classic pharmacovigilance heuristic gate.
    signal = (a >= min_case_count) and (prr >= 2.0) and (chi_square >= 4.0)

    return {
        "prr": round(float(prr), 4),
        "ror": round(float(ror), 4),
        "ror_ci95_lower": round(float(ror_ci95_lower), 4),
        "ror_ci95_upper": round(float(ror_ci95_upper), 4),
        "chi_square": round(float(chi_square), 4),
        "meets_signal_criteria": bool(signal),
    }


@registry.register(
    name="safety.antitarget_profile",
    description="Screen degradation data for anti-target hits (tumor suppressors, essential genes, heme TFs)",
    category="safety",
    parameters={
        "compound_id": "Compound to profile (or 'all')",
        "lfc_threshold": "LFC threshold for degradation call (default -0.5)",
    },
    requires_data=["proteomics"],
    usage_guide="You need to check if a compound degrades dangerous off-targets (tumor suppressors, essential heme TFs, teratogenic substrates). Run this first in any safety assessment workflow.",
)
def antitarget_profile(compound_id: str = "all", lfc_threshold: float = -0.5, **kwargs) -> dict:
    """Screen proteomics data for degradation of anti-target proteins.

    Anti-targets: tumor suppressors, essential heme TFs, teratogenic substrates,
    and known CRBN substrates. Degrading these = safety liability.
    """
    from ct.tools._compound_resolver import resolve_compound
    if compound_id != "all":
        compound_id = resolve_compound(compound_id, dataset="proteomics")

    try:
        from ct.data.loaders import load_proteomics
        prot = load_proteomics()
    except FileNotFoundError:
        return {
            "error": "Proteomics data not available.",
            "summary": "Proteomics data not available — skipping. Provide proteomics data for full analysis.",
        }

    compounds = [compound_id] if compound_id != "all" else prot.columns.tolist()
    all_proteins = set(prot.index)

    # Categorize known anti-targets present in data (handles both gene symbols and UniProt IDs)
    tsg_present = _gene_ids(TUMOR_SUPPRESSORS, all_proteins)
    heme_present = _gene_ids(HEME_TFS, all_proteins)
    terat_present = _gene_ids(TERATOGENIC_SUBSTRATES, all_proteins)
    crbn_present = _gene_ids(CRBN_SUBSTRATES, all_proteins)

    results = []
    for cpd in compounds:
        if cpd not in prot.columns:
            continue

        values = prot[cpd].dropna()
        degraded = values[values < lfc_threshold]

        # Check anti-target categories
        hits = {
            "tumor_suppressors": sorted([p for p in degraded.index if p in tsg_present]),
            "heme_tfs": sorted([p for p in degraded.index if p in heme_present]),
            "teratogenic": sorted([p for p in degraded.index if p in terat_present]),
            "crbn_substrates": sorted([p for p in degraded.index if p in crbn_present]),
        }

        n_antitargets = sum(len(v) for v in hits.values())

        # Compute safety penalty score
        penalty = 0.0
        for p in hits["teratogenic"]:
            penalty += 10.0  # highest risk
        for p in hits["heme_tfs"]:
            penalty += 5.0
        for p in hits["tumor_suppressors"]:
            penalty += 3.0
        for p in hits["crbn_substrates"]:
            penalty += 2.0

        # Get LFC values for flagged proteins
        flagged_details = []
        for category, proteins in hits.items():
            for p in proteins:
                flagged_details.append({
                    "protein": _display_name(p),
                    "protein_id": p,
                    "category": category,
                    "lfc": round(float(values[p]), 3),
                })

        results.append({
            "compound": cpd,
            "n_total_degraded": len(degraded),
            "n_antitargets": n_antitargets,
            "n_tumor_suppressors": len(hits["tumor_suppressors"]),
            "n_heme_tfs": len(hits["heme_tfs"]),
            "n_teratogenic": len(hits["teratogenic"]),
            "n_crbn_substrates": len(hits["crbn_substrates"]),
            "safety_penalty": round(penalty, 1),
            "flagged_proteins": flagged_details,
        })

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "flagged_proteins"} for r in results])
    if len(df) > 0:
        df = df.sort_values("safety_penalty", ascending=False)

    if compound_id != "all":
        r = results[0] if results else {}
        flagged_str = ", ".join([f"{d['protein']}({d['category']})" for d in r.get("flagged_proteins", [])])
        summary = (
            f"Anti-target profile for {compound_id}: "
            f"{r.get('n_antitargets', 0)} anti-targets hit, "
            f"penalty={r.get('safety_penalty', 0)}\n"
            f"Flagged: {flagged_str if flagged_str else 'none'}"
        )
    else:
        n_clean = (df["n_antitargets"] == 0).sum() if len(df) > 0 else 0
        summary = (
            f"Anti-target screening: {len(df)} compounds profiled\n"
            f"Clean (0 anti-targets): {n_clean}/{len(df)}"
        )

    return {
        "summary": summary,
        "n_screened": len(tsg_present | heme_present | terat_present | crbn_present),
        "antitarget_counts": {
            "tumor_suppressors": len(tsg_present),
            "heme_tfs": len(heme_present),
            "teratogenic": len(terat_present),
            "crbn_substrates": len(crbn_present),
        },
        "profiles": results if compound_id != "all" else df.to_dict("records"),
    }


@registry.register(
    name="safety.classify",
    description="Classify compound safety as SAFE/CAUTION/DANGEROUS based on multi-modal profiling",
    category="safety",
    parameters={
        "compound_id": "Compound to classify (or 'all')",
    },
    requires_data=["proteomics", "prism"],
    usage_guide="You need a quick safety verdict (SAFE/CAUTION/DANGEROUS) before advancing a compound. Combines anti-target profile with viability breadth. Run after antitarget_profile for full context.",
)
def classify(compound_id: str = "all", **kwargs) -> dict:
    """Multi-modal safety classification.

    Classification rules:
    - DANGEROUS: degrades any teratogenic substrate OR safety_penalty >= 15
    - CAUTION: degrades tumor suppressors OR heme TFs OR safety_penalty >= 5
    - SAFE: no anti-target degradation AND safety_penalty < 5

    Also considers viability breadth (% cell lines killed) as a toxicity signal.
    """
    # Get anti-target profile (handles missing proteomics internally)
    at_result = antitarget_profile(compound_id=compound_id)
    if "error" in at_result:
        return at_result

    profiles = at_result["profiles"]

    # Get viability breadth from PRISM
    try:
        from ct.data.loaders import load_prism
        prism = load_prism()
    except FileNotFoundError:
        return {
            "error": "PRISM data not available.",
            "summary": "PRISM data not available — skipping. Run: ct data pull prism",
        }

    results = []
    for profile in profiles:
        cpd = profile["compound"]
        penalty = profile["safety_penalty"]

        # Viability breadth
        cpd_data = prism[prism["pert_name"] == cpd]
        breadth = 0.0
        if len(cpd_data) > 0:
            max_dose = cpd_data["pert_dose"].max()
            cpd_hd = cpd_data[cpd_data["pert_dose"] == max_dose]
            per_cell = cpd_hd.groupby("ccle_name")["LFC"].mean()
            breadth = float((per_cell < -0.5).mean())

        # Classification
        if profile["n_teratogenic"] > 0 or penalty >= 15:
            classification = "DANGEROUS"
        elif profile["n_tumor_suppressors"] > 0 or profile["n_heme_tfs"] > 0 or penalty >= 5:
            classification = "CAUTION"
        elif breadth > 0.8:
            classification = "CAUTION"  # kills too many cell lines = nonspecific toxicity
        else:
            classification = "SAFE"

        # Safety score (0-100, higher = safer)
        safety_score = max(0, 100 - penalty * 5 - breadth * 30)

        results.append({
            "compound": cpd,
            "classification": classification,
            "safety_score": round(safety_score, 1),
            "safety_penalty": penalty,
            "viability_breadth": round(breadth, 3),
            "n_antitargets": profile["n_antitargets"],
            "n_tumor_suppressors": profile["n_tumor_suppressors"],
            "n_heme_tfs": profile["n_heme_tfs"],
            "n_teratogenic": profile["n_teratogenic"],
        })

    df = pd.DataFrame(results)

    if len(df) > 0:
        counts = df["classification"].value_counts().to_dict()
        safe = counts.get("SAFE", 0)
        caution = counts.get("CAUTION", 0)
        dangerous = counts.get("DANGEROUS", 0)
    else:
        safe = caution = dangerous = 0

    if compound_id != "all" and results:
        r = results[0]
        summary = (
            f"Safety classification for {compound_id}: {r['classification']}\n"
            f"Score: {r['safety_score']}/100, Penalty: {r['safety_penalty']}, "
            f"Viability breadth: {r['viability_breadth']:.1%}"
        )
    else:
        summary = (
            f"Safety classification: {len(df)} compounds\n"
            f"SAFE: {safe}, CAUTION: {caution}, DANGEROUS: {dangerous}"
        )

    return {
        "summary": summary,
        "classifications": results,
        "distribution": {"SAFE": safe, "CAUTION": caution, "DANGEROUS": dangerous},
    }


@registry.register(
    name="safety.sall4_risk",
    description="Assess SALL4 degradation risk for IMiD-type molecular glue compounds (teratogenicity marker)",
    category="safety",
    parameters={
        "compound_id": "Compound to check (or 'all')",
    },
    requires_data=["proteomics"],
    usage_guide="You are working with CRBN-based molecular glues and need to assess teratogenicity risk. SALL4 degradation was the molecular cause of thalidomide birth defects — critical safety check for any IMiD-type compound.",
)
def sall4_risk(compound_id: str = "all", **kwargs) -> dict:
    """Check for SALL4 degradation -- the key teratogenicity signal for IMiD-type compounds.

    SALL4 is a zinc finger TF essential for limb development. Its degradation by
    thalidomide via CRBN was the molecular cause of thalidomide teratogenicity.
    Any CRBN-based molecular glue that degrades SALL4 is a teratogenicity risk.
    """
    from ct.tools._compound_resolver import resolve_compound
    if compound_id != "all":
        compound_id = resolve_compound(compound_id, dataset="proteomics")

    try:
        from ct.data.loaders import load_proteomics
        prot = load_proteomics()
    except FileNotFoundError:
        return {
            "error": "Proteomics data not available.",
            "summary": "Proteomics data not available — skipping. Provide proteomics data for full analysis.",
        }

    # Check for SALL family proteins (handles both gene symbols and UniProt IDs)
    sall_uniprot = {uid: gene for uid, gene in UNIPROT_TO_GENE.items() if gene.startswith("SALL")}
    sall_proteins = []  # list of (index_id, gene_symbol)
    for p in prot.index:
        if p.startswith("SALL"):
            sall_proteins.append((p, p))
        elif p in sall_uniprot:
            sall_proteins.append((p, sall_uniprot[p]))

    if not sall_proteins:
        return {
            "summary": "SALL proteins not detected in proteomics data -- cannot assess teratogenicity risk",
            "sall_proteins_in_data": [],
            "risk_assessment": "UNKNOWN",
        }

    compounds = [compound_id] if compound_id != "all" else prot.columns.tolist()
    results = []

    for cpd in compounds:
        if cpd not in prot.columns:
            continue

        sall_values = {}  # gene_symbol -> LFC
        for idx_id, gene in sall_proteins:
            val = prot.loc[idx_id, cpd]
            if pd.notna(val):
                sall_values[gene] = float(val)

        # Risk assessment
        sall4_lfc = sall_values.get("SALL4")
        any_sall_degraded = any(v < -0.5 for v in sall_values.values())
        sall4_degraded = sall4_lfc is not None and sall4_lfc < -0.5

        if sall4_degraded:
            risk = "HIGH"
            risk_detail = f"SALL4 degraded (LFC={sall4_lfc:.2f}) -- thalidomide-like teratogenicity risk"
        elif any_sall_degraded:
            risk = "MODERATE"
            degraded_salls = {k: round(v, 3) for k, v in sall_values.items() if v < -0.5}
            risk_detail = f"SALL family member(s) degraded: {degraded_salls} -- potential teratogenicity"
        elif sall4_lfc is not None and sall4_lfc < -0.3:
            risk = "LOW"
            risk_detail = f"SALL4 mildly reduced (LFC={sall4_lfc:.2f}) -- monitor in follow-up"
        else:
            risk = "MINIMAL"
            risk_detail = "No SALL degradation detected"

        results.append({
            "compound": cpd,
            "risk_level": risk,
            "risk_detail": risk_detail,
            "sall_values": sall_values,
        })

    sall_names = [gene for _, gene in sall_proteins]

    if compound_id != "all" and results:
        r = results[0]
        summary = f"SALL4 risk for {compound_id}: {r['risk_level']} -- {r['risk_detail']}"
    else:
        risk_counts = {}
        for r in results:
            risk_counts[r["risk_level"]] = risk_counts.get(r["risk_level"], 0) + 1
        summary = f"SALL4 risk assessment: {len(results)} compounds -- {risk_counts}"

    return {
        "summary": summary,
        "sall_proteins_in_data": sall_names,
        "assessments": results,
    }


@registry.register(
    name="safety.faers_signal_scan",
    description="Scan openFDA FAERS adverse-event reports for disproportionality signals (PRR/ROR) for a drug",
    category="safety",
    parameters={
        "drug_name": "Drug name to scan (generic or brand name)",
        "event": "Optional specific MedDRA preferred term to evaluate",
        "top_n": "If event not provided, evaluate top N reported events for this drug (default 5)",
        "min_case_count": "Minimum A-count threshold for signal flagging (default 3)",
    },
    usage_guide=(
        "Use for post-marketing pharmacovigilance triage. Computes disproportionality metrics "
        "(PRR/ROR/chi-square) from openFDA FAERS counts and flags candidate safety signals."
    ),
)
def faers_signal_scan(
    drug_name: str,
    event: str = "",
    top_n: int = 5,
    min_case_count: int = 3,
    **kwargs,
) -> dict:
    """Run a disproportionality safety scan using openFDA FAERS."""
    if not drug_name or not drug_name.strip():
        return {"error": "drug_name is required", "summary": "No drug name provided"}

    drug_term = _openfda_escape(drug_name)
    if not drug_term:
        return {"error": "drug_name is required", "summary": "No drug name provided"}

    top_n = max(1, min(int(top_n or 5), 20))
    min_case_count = max(1, int(min_case_count or 3))

    drug_search = f'patient.drug.medicinalproduct.exact:"{drug_term}"'

    all_total, error = _openfda_total("")
    if error:
        return {"error": f"openFDA total lookup failed: {error}", "summary": f"FAERS scan failed: {error}"}
    drug_total, error = _openfda_total(drug_search)
    if error:
        return {"error": f"openFDA drug lookup failed: {error}", "summary": f"FAERS scan failed for {drug_name}: {error}"}

    if drug_total <= 0:
        return {
            "drug_name": drug_name,
            "total_reports_for_drug": 0,
            "signals": [],
            "summary": f"No FAERS reports found for '{drug_name}'",
        }

    events_to_scan = []
    if event and event.strip():
        events_to_scan = [event.strip()]
    else:
        data, count_error = request_json(
            "GET",
            _OPENFDA_DRUG_EVENT_URL,
            params={
                "search": drug_search,
                "count": "patient.reaction.reactionmeddrapt.exact",
                "limit": str(top_n),
            },
            timeout=20,
            retries=2,
        )
        if count_error:
            return {
                "error": f"openFDA event aggregation failed: {count_error}",
                "summary": f"FAERS scan failed for {drug_name}: {count_error}",
            }
        events_to_scan = [r.get("term", "") for r in data.get("results", []) if r.get("term")]
        if not events_to_scan:
            return {
                "drug_name": drug_name,
                "total_reports_for_drug": int(drug_total),
                "signals": [],
                "summary": f"FAERS reports found for '{drug_name}', but no reaction terms were returned",
            }

    signals = []
    for ev in events_to_scan:
        ev_term = _openfda_escape(ev)
        if not ev_term:
            continue

        event_search = f'patient.reaction.reactionmeddrapt.exact:"{ev_term}"'
        both_search = f"{drug_search}+AND+{event_search}"

        event_total, event_err = _openfda_total(event_search)
        both_total, both_err = _openfda_total(both_search)
        if event_err or both_err:
            signals.append({
                "event": ev,
                "error": event_err or both_err,
            })
            continue

        a = int(both_total)
        b = int(drug_total) - a
        c = int(event_total) - a
        d = int(all_total) - (a + b + c)
        if d < 0:
            d = 0

        metrics = _faers_signal_metrics(a, b, c, d, min_case_count=min_case_count)
        signals.append({
            "event": ev,
            "a_drug_and_event": a,
            "b_drug_no_event": b,
            "c_no_drug_event": c,
            "d_no_drug_no_event": d,
            **metrics,
        })

    clean_signals = [s for s in signals if "error" not in s]
    clean_signals.sort(
        key=lambda x: (x.get("meets_signal_criteria", False), x.get("prr", 0.0), x.get("ror", 0.0)),
        reverse=True,
    )
    n_flagged = sum(1 for s in clean_signals if s.get("meets_signal_criteria"))
    error_count = len(signals) - len(clean_signals)

    if clean_signals:
        top = clean_signals[0]
        summary = (
            f"FAERS signal scan for {drug_name}: {len(clean_signals)} event(s) analyzed, "
            f"{n_flagged} flagged by PRR/ROR criteria. Top event: {top['event']} "
            f"(PRR={top['prr']}, ROR={top['ror']})."
        )
    else:
        summary = (
            f"FAERS signal scan for {drug_name}: no analyzable events returned"
            + (f" ({error_count} event lookup error(s))." if error_count else ".")
        )

    return {
        "summary": summary,
        "drug_name": drug_name,
        "event_filter": event.strip(),
        "total_reports_all_faers": int(all_total),
        "total_reports_for_drug": int(drug_total),
        "criteria": {
            "min_case_count": min_case_count,
            "prr_threshold": 2.0,
            "chi_square_threshold": 4.0,
        },
        "n_events_analyzed": len(clean_signals),
        "n_events_flagged": n_flagged,
        "n_event_lookup_errors": error_count,
        "signals": clean_signals,
        "errors": [s for s in signals if "error" in s],
    }


@registry.register(
    name="safety.label_risk_extract",
    description="Extract boxed warnings, contraindications, and key risk sections from openFDA drug labels",
    category="safety",
    parameters={
        "drug_name": "Drug name (generic or brand)",
        "max_labels": "Maximum label records to inspect (default 3)",
        "section_max_chars": "Max characters per extracted section (default 500)",
    },
    usage_guide=(
        "Use for rapid regulatory risk triage. Pulls key safety sections from FDA labels "
        "(boxed warning, contraindications, warnings, interactions, special populations)."
    ),
)
def label_risk_extract(
    drug_name: str,
    max_labels: int = 3,
    section_max_chars: int = 500,
    **kwargs,
) -> dict:
    """Extract key risk sections from openFDA drug label endpoint."""
    import re

    if not drug_name or not drug_name.strip():
        return {"error": "drug_name is required", "summary": "No drug name provided"}

    max_labels = max(1, min(int(max_labels or 3), 10))
    section_max_chars = max(120, min(int(section_max_chars or 500), 4000))
    drug_term = _openfda_escape(drug_name)

    search = (
        f'openfda.generic_name.exact:"{drug_term}"'
        f'+OR+openfda.brand_name.exact:"{drug_term}"'
        f'+OR+openfda.substance_name.exact:"{drug_term}"'
    )
    data, error = request_json(
        "GET",
        _OPENFDA_DRUG_LABEL_URL,
        params={"search": search, "limit": str(max_labels)},
        timeout=20,
        retries=2,
    )
    if error:
        return {"error": f"openFDA label query failed: {error}", "summary": f"Label risk extraction failed: {error}"}

    results = data.get("results", [])
    if not results:
        return {
            "drug_name": drug_name,
            "labels_found": 0,
            "risk_level": "UNKNOWN",
            "summary": f"No openFDA label records found for '{drug_name}'",
            "labels": [],
        }

    def _extract_section(entry: dict, key: str) -> str:
        value = entry.get(key, [])
        if isinstance(value, list):
            text = " ".join(str(v).strip() for v in value if str(v).strip())
        elif isinstance(value, str):
            text = value.strip()
        else:
            text = ""
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > section_max_chars:
            text = text[: section_max_chars - 3] + "..."
        return text

    label_summaries = []
    for entry in results:
        openfda = entry.get("openfda", {})
        brand = ", ".join(openfda.get("brand_name", [])[:3]) if isinstance(openfda.get("brand_name"), list) else ""
        generic = ", ".join(openfda.get("generic_name", [])[:3]) if isinstance(openfda.get("generic_name"), list) else ""
        application = ", ".join(openfda.get("application_number", [])[:3]) if isinstance(openfda.get("application_number"), list) else ""
        manufacturer = ", ".join(openfda.get("manufacturer_name", [])[:2]) if isinstance(openfda.get("manufacturer_name"), list) else ""

        sections = {
            "boxed_warning": _extract_section(entry, "boxed_warning"),
            "contraindications": _extract_section(entry, "contraindications"),
            "warnings_and_cautions": _extract_section(entry, "warnings_and_cautions"),
            "warnings": _extract_section(entry, "warnings"),
            "adverse_reactions": _extract_section(entry, "adverse_reactions"),
            "drug_interactions": _extract_section(entry, "drug_interactions"),
            "use_in_specific_populations": _extract_section(entry, "use_in_specific_populations"),
        }

        has_boxed = bool(sections["boxed_warning"])
        has_contra = bool(sections["contraindications"])
        has_warn = bool(sections["warnings"] or sections["warnings_and_cautions"])

        if has_boxed:
            risk_level = "HIGH"
        elif has_contra or has_warn:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        flags = []
        if has_boxed:
            flags.append("boxed_warning")
        if has_contra:
            flags.append("contraindications")
        if has_warn:
            flags.append("warnings")

        label_summaries.append({
            "brand_name": brand,
            "generic_name": generic,
            "application_number": application,
            "manufacturer": manufacturer,
            "risk_level": risk_level,
            "risk_flags": flags,
            "sections": sections,
        })

    rank = {"HIGH": 3, "MODERATE": 2, "LOW": 1}
    overall_risk = max(label_summaries, key=lambda x: rank.get(x["risk_level"], 0))["risk_level"]
    boxed_count = sum(1 for l in label_summaries if "boxed_warning" in l.get("risk_flags", []))
    contra_count = sum(1 for l in label_summaries if "contraindications" in l.get("risk_flags", []))

    summary = (
        f"Label risk extraction for {drug_name}: {len(label_summaries)} label record(s), "
        f"overall risk={overall_risk}. Boxed warning present in {boxed_count} label(s); "
        f"contraindications present in {contra_count} label(s)."
    )

    return {
        "summary": summary,
        "drug_name": drug_name,
        "labels_found": len(label_summaries),
        "risk_level": overall_risk,
        "n_boxed_warning_labels": boxed_count,
        "n_contraindication_labels": contra_count,
        "labels": label_summaries,
    }


@registry.register(
    name="safety.admet_predict",
    description="Predict ADMET properties for a compound from SMILES using RDKit descriptors and heuristic rules",
    category="safety",
    parameters={
        "smiles": "SMILES string for the compound to profile",
    },
    usage_guide="You need a comprehensive ADMET (absorption, distribution, metabolism, excretion, toxicity) profile for a compound. Use early in lead optimization to flag liabilities before synthesis. Covers Lipinski, Veber, Ghose, lead-likeness, oral absorption, BBB, hERG, CYP, and solubility.",
)
def admet_predict(smiles: str, **kwargs) -> dict:
    """Predict ADMET properties from SMILES using RDKit descriptors and heuristic rules.

    Computes physicochemical properties and applies established medicinal chemistry
    filters (Lipinski Ro5, Veber, Ghose, lead-likeness) plus heuristic predictions
    for oral absorption, BBB penetration, hERG risk, CYP liability, and solubility.
    """
    from ct.tools.chemistry import _extract_smiles
    smiles = _extract_smiles(smiles)

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
    except ImportError:
        return {"error": "RDKit is required for ADMET prediction. Install with: pip install rdkit", "summary": "RDKit is required for ADMET prediction. Install with: pip install rdkit"}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}", "summary": f"Could not parse SMILES: {smiles}"}

    # --- Physicochemical descriptors ---
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hba = Lipinski.NumHAcceptors(mol)
    hbd = Lipinski.NumHDonors(mol)
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    aromatic_rings = Descriptors.NumAromaticRings(mol)
    num_rings = Descriptors.RingCount(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    formula = rdMolDescriptors.CalcMolFormula(mol)

    properties = {
        "smiles": smiles,
        "formula": formula,
        "molecular_weight": round(mw, 2),
        "logp": round(logp, 2),
        "tpsa": round(tpsa, 2),
        "hba": hba,
        "hbd": hbd,
        "rotatable_bonds": rotatable_bonds,
        "aromatic_rings": aromatic_rings,
        "num_rings": num_rings,
        "heavy_atoms": heavy_atoms,
    }

    # --- Drug-likeness filters ---
    filters = {}

    # Lipinski Rule of Five
    lipinski_violations = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10,
    ])
    filters["lipinski_ro5"] = {
        "pass": lipinski_violations <= 1,
        "violations": lipinski_violations,
        "details": {
            "MW<=500": mw <= 500,
            "LogP<=5": logp <= 5,
            "HBD<=5": hbd <= 5,
            "HBA<=10": hba <= 10,
        },
    }

    # Veber rule (oral bioavailability)
    veber_pass = tpsa <= 140 and rotatable_bonds <= 10
    filters["veber"] = {
        "pass": veber_pass,
        "details": {
            "TPSA<=140": tpsa <= 140,
            "RotBonds<=10": rotatable_bonds <= 10,
        },
    }

    # Lead-likeness (Teague/Oprea)
    lead_like = mw <= 350 and logp <= 3.5 and rotatable_bonds <= 7
    filters["lead_likeness"] = {
        "pass": lead_like,
        "details": {
            "MW<=350": mw <= 350,
            "LogP<=3.5": logp <= 3.5,
            "RotBonds<=7": rotatable_bonds <= 7,
        },
    }

    # Ghose filter
    ghose_pass = (
        160 <= mw <= 480
        and -0.4 <= logp <= 5.6
        and 40 <= heavy_atoms <= 130  # using heavy atoms as proxy for atom count
        and 20 <= Descriptors.MolMR(mol) <= 130
    )
    filters["ghose"] = {
        "pass": ghose_pass,
        "details": {
            "160<=MW<=480": 160 <= mw <= 480,
            "-0.4<=LogP<=5.6": -0.4 <= logp <= 5.6,
            "20<=MR<=130": 20 <= Descriptors.MolMR(mol) <= 130,
        },
    }

    # --- ADMET predictions (heuristic) ---
    predictions = {}

    # Oral absorption
    oral_absorption = tpsa < 140 and rotatable_bonds <= 10
    oral_score = max(0, 100 - (max(0, tpsa - 60) * 0.8) - (max(0, rotatable_bonds - 5) * 5))
    predictions["oral_absorption"] = {
        "prediction": "likely" if oral_absorption else "poor",
        "score": round(min(100, oral_score), 1),
        "rationale": f"TPSA={tpsa:.0f} ({'<' if tpsa < 140 else '>='} 140), "
                     f"RotBonds={rotatable_bonds} ({'<=' if rotatable_bonds <= 10 else '>'} 10)",
    }

    # BBB penetration
    bbb = tpsa < 90 and mw < 450 and 1 <= logp <= 3
    bbb_score = max(0, 100 - max(0, tpsa - 40) * 1.2 - max(0, mw - 300) * 0.3 - abs(logp - 2) * 15)
    predictions["bbb_penetration"] = {
        "prediction": "likely" if bbb else "unlikely",
        "score": round(min(100, bbb_score), 1),
        "rationale": f"TPSA={tpsa:.0f} ({'<' if tpsa < 90 else '>='} 90), "
                     f"MW={mw:.0f} ({'<' if mw < 450 else '>='} 450), "
                     f"LogP={logp:.1f} ({'in' if 1 <= logp <= 3 else 'outside'} 1-3)",
    }

    # hERG risk (rough heuristic)
    herg_risk = logp > 3.7 and mw > 400
    herg_concern = "elevated" if herg_risk else "low"
    predictions["herg_risk"] = {
        "prediction": herg_concern,
        "flag": herg_risk,
        "rationale": f"LogP={logp:.1f} ({'>' if logp > 3.7 else '<='} 3.7), "
                     f"MW={mw:.0f} ({'>' if mw > 400 else '<='} 400). "
                     f"Lipophilic, large molecules more likely to block hERG channel.",
    }

    # CYP liability
    cyp_risk_factors = 0
    cyp_details = []
    if aromatic_rings >= 3:
        cyp_risk_factors += 1
        cyp_details.append(f"{aromatic_rings} aromatic rings (>=3)")
    if logp > 3:
        cyp_risk_factors += 1
        cyp_details.append(f"LogP={logp:.1f} (>3)")
    if mw > 500:
        cyp_risk_factors += 1
        cyp_details.append(f"MW={mw:.0f} (>500)")

    cyp_level = "high" if cyp_risk_factors >= 2 else "moderate" if cyp_risk_factors == 1 else "low"
    predictions["cyp_liability"] = {
        "prediction": cyp_level,
        "risk_factors": cyp_risk_factors,
        "details": cyp_details if cyp_details else ["No major CYP liability flags"],
    }

    # Solubility class (simplified Yalkowsky-based heuristic: logS ~ 0.5 - 0.01*(MP) - logP)
    # Without melting point, use MW as rough proxy: logS ~ 0.5 - 0.01*MW - logP
    log_s_est = 0.5 - 0.01 * mw - logp
    if log_s_est > -1:
        sol_class = "highly soluble"
    elif log_s_est > -3:
        sol_class = "soluble"
    elif log_s_est > -5:
        sol_class = "moderately soluble"
    elif log_s_est > -7:
        sol_class = "poorly soluble"
    else:
        sol_class = "insoluble"

    predictions["solubility"] = {
        "class": sol_class,
        "estimated_logS": round(log_s_est, 2),
        "rationale": f"Estimated logS={log_s_est:.2f} (Yalkowsky-type heuristic from MW and LogP)",
    }

    # --- Overall ADMET verdict ---
    flags = []
    if not filters["lipinski_ro5"]["pass"]:
        flags.append(f"Lipinski: {lipinski_violations} violations")
    if not veber_pass:
        flags.append("Fails Veber (oral bioavailability concern)")
    if herg_risk:
        flags.append("Elevated hERG risk")
    if cyp_level == "high":
        flags.append("High CYP liability")
    if sol_class in ("poorly soluble", "insoluble"):
        flags.append(f"Solubility: {sol_class}")

    if not flags:
        verdict = "FAVORABLE"
    elif len(flags) <= 2:
        verdict = "ACCEPTABLE"
    else:
        verdict = "UNFAVORABLE"

    summary_parts = [
        f"ADMET profile for {formula} (MW={mw:.0f}, LogP={logp:.1f}): {verdict}",
        f"Lipinski: {'PASS' if filters['lipinski_ro5']['pass'] else 'FAIL'} ({lipinski_violations} violations)",
        f"Oral absorption: {predictions['oral_absorption']['prediction']} (score {predictions['oral_absorption']['score']})",
        f"BBB: {predictions['bbb_penetration']['prediction']} (score {predictions['bbb_penetration']['score']})",
        f"hERG: {predictions['herg_risk']['prediction']}, CYP: {predictions['cyp_liability']['prediction']}",
        f"Solubility: {predictions['solubility']['class']} (logS~{log_s_est:.1f})",
    ]
    if flags:
        summary_parts.append(f"Flags: {'; '.join(flags)}")

    return {
        "summary": "\n".join(summary_parts),
        "verdict": verdict,
        "properties": properties,
        "filters": filters,
        "predictions": predictions,
        "flags": flags,
    }


@registry.register(
    name="safety.ddi_predict",
    description="Predict drug-drug interaction potential based on CYP metabolism profile and molecular features",
    category="safety",
    parameters={
        "smiles": "SMILES string for the primary compound",
        "comedication_smiles": "SMILES string for a co-administered drug (optional)",
    },
    usage_guide="You need to assess drug-drug interaction risk for a compound, especially CYP-mediated interactions. Use when evaluating combination therapies or compounds likely to be co-prescribed. Identifies CYP inhibition/induction risk from structural features.",
)
def ddi_predict(smiles: str, comedication_smiles: str = None, **kwargs) -> dict:
    """Predict drug-drug interaction potential based on CYP metabolism profile.

    Uses structural features to estimate CYP inhibition risk for major isoforms
    (3A4, 2D6, 2C9, 2C19, 1A2). Optionally compares with a co-medication.
    """
    from ct.tools.chemistry import _extract_smiles
    smiles = _extract_smiles(smiles)

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
    except ImportError:
        return {"error": "RDKit is required for DDI prediction. Install with: pip install rdkit", "summary": "RDKit is required for DDI prediction. Install with: pip install rdkit"}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}", "summary": f"Could not parse SMILES: {smiles}"}

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    aromatic_rings = Descriptors.NumAromaticRings(mol)
    num_rings = Descriptors.RingCount(mol)
    hba = Descriptors.NumHAcceptors(mol)

    # --- Detect structural motifs associated with CYP inhibition ---
    motif_flags = {}

    # Nitrogen heterocycles (CYP3A4 inhibition)
    n_heterocycle_pattern = Chem.MolFromSmarts("[nR]")  # ring nitrogen
    n_heterocycles = len(mol.GetSubstructMatches(n_heterocycle_pattern)) if n_heterocycle_pattern else 0
    motif_flags["nitrogen_heterocycles"] = n_heterocycles

    # Imidazole motif (strong CYP inhibition — azole antifungals)
    # Multiple SMARTS to catch both NH and N-substituted forms
    imidazole_patterns = [
        Chem.MolFromSmarts("c1cnc[nH]1"),   # unsubstituted
        Chem.MolFromSmarts("c1nccn1"),       # N-substituted imidazole
        Chem.MolFromSmarts("c1cncn1"),       # alternative numbering
    ]
    has_imidazole = any(
        pat is not None and bool(mol.GetSubstructMatches(pat))
        for pat in imidazole_patterns
    )

    # Triazole motif
    triazole_1 = Chem.MolFromSmarts("c1nncn1")
    triazole_2 = Chem.MolFromSmarts("c1nnn[nH]1")
    has_triazole = (
        (bool(mol.GetSubstructMatches(triazole_1)) if triazole_1 else False)
        or (bool(mol.GetSubstructMatches(triazole_2)) if triazole_2 else False)
    )
    motif_flags["has_imidazole"] = has_imidazole
    motif_flags["has_triazole"] = has_triazole
    motif_flags["has_azole"] = has_imidazole or has_triazole

    # Furanyl groups (mechanism-based CYP inhibition)
    furan = Chem.MolFromSmarts("c1ccoc1")
    has_furan = bool(mol.GetSubstructMatches(furan)) if furan else False
    motif_flags["has_furan"] = has_furan

    # Amine groups (CYP2D6 substrates/inhibitors)
    basic_amine = Chem.MolFromSmarts("[NX3;!$(NC=O);!$(NS=O)]")
    n_basic_amines = len(mol.GetSubstructMatches(basic_amine)) if basic_amine else 0
    motif_flags["basic_amines"] = n_basic_amines

    # --- CYP isoform risk assessment ---
    cyp_profile = {}

    # CYP3A4 — the major drug-metabolizing enzyme
    cyp3a4_score = 0
    cyp3a4_reasons = []
    if has_imidazole or has_triazole:
        cyp3a4_score += 3
        cyp3a4_reasons.append("Azole motif (strong CYP3A4 inhibition)")
    if n_heterocycles >= 2:
        cyp3a4_score += 1
        cyp3a4_reasons.append(f"{n_heterocycles} nitrogen heterocycles")
    if mw > 400 and logp > 3:
        cyp3a4_score += 1
        cyp3a4_reasons.append(f"Large lipophilic molecule (MW={mw:.0f}, LogP={logp:.1f})")
    cyp_profile["CYP3A4"] = {
        "inhibition_risk": "high" if cyp3a4_score >= 3 else "moderate" if cyp3a4_score >= 1 else "low",
        "score": cyp3a4_score,
        "reasons": cyp3a4_reasons if cyp3a4_reasons else ["No major CYP3A4 inhibition flags"],
    }

    # CYP2D6
    cyp2d6_score = 0
    cyp2d6_reasons = []
    if n_basic_amines >= 1:
        cyp2d6_score += 1
        cyp2d6_reasons.append(f"{n_basic_amines} basic amine(s) — CYP2D6 substrate/inhibitor feature")
    if aromatic_rings >= 2 and n_basic_amines >= 1:
        cyp2d6_score += 1
        cyp2d6_reasons.append("Lipophilic amine — classic CYP2D6 inhibitor pharmacophore")
    cyp_profile["CYP2D6"] = {
        "inhibition_risk": "high" if cyp2d6_score >= 2 else "moderate" if cyp2d6_score >= 1 else "low",
        "score": cyp2d6_score,
        "reasons": cyp2d6_reasons if cyp2d6_reasons else ["No major CYP2D6 inhibition flags"],
    }

    # CYP2C9
    cyp2c9_score = 0
    cyp2c9_reasons = []
    if logp > 3 and aromatic_rings >= 2:
        cyp2c9_score += 1
        cyp2c9_reasons.append("Lipophilic aromatic compound")
    # Acidic groups — CYP2C9 substrates tend to be weak acids
    carboxylic = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
    has_acid = bool(mol.GetSubstructMatches(carboxylic)) if carboxylic else False
    if has_acid:
        cyp2c9_score += 1
        cyp2c9_reasons.append("Carboxylic acid group — CYP2C9 substrate feature")
    cyp_profile["CYP2C9"] = {
        "inhibition_risk": "moderate" if cyp2c9_score >= 1 else "low",
        "score": cyp2c9_score,
        "reasons": cyp2c9_reasons if cyp2c9_reasons else ["No major CYP2C9 inhibition flags"],
    }

    # CYP2C19
    cyp2c19_score = 0
    cyp2c19_reasons = []
    if has_imidazole:
        cyp2c19_score += 2
        cyp2c19_reasons.append("Imidazole motif (CYP2C19 inhibitor feature)")
    if n_heterocycles >= 2 and mw < 500:
        cyp2c19_score += 1
        cyp2c19_reasons.append("Multiple nitrogen heterocycles")
    cyp_profile["CYP2C19"] = {
        "inhibition_risk": "high" if cyp2c19_score >= 2 else "moderate" if cyp2c19_score >= 1 else "low",
        "score": cyp2c19_score,
        "reasons": cyp2c19_reasons if cyp2c19_reasons else ["No major CYP2C19 inhibition flags"],
    }

    # CYP1A2
    cyp1a2_score = 0
    cyp1a2_reasons = []
    if aromatic_rings >= 3:
        cyp1a2_score += 1
        cyp1a2_reasons.append(f"{aromatic_rings} aromatic rings — planar aromatic CYP1A2 substrate")
    # Fused ring systems
    if num_rings >= 3 and aromatic_rings >= 2:
        cyp1a2_score += 1
        cyp1a2_reasons.append("Polycyclic aromatic system")
    cyp_profile["CYP1A2"] = {
        "inhibition_risk": "moderate" if cyp1a2_score >= 1 else "low",
        "score": cyp1a2_score,
        "reasons": cyp1a2_reasons if cyp1a2_reasons else ["No major CYP1A2 inhibition flags"],
    }

    # --- Mechanism-based inhibition (MBI) risk ---
    mbi_risk = False
    mbi_reasons = []
    if has_furan:
        mbi_risk = True
        mbi_reasons.append("Furan ring — known MBI risk (bioactivated to reactive epoxide)")
    # Terminal alkyne
    alkyne = Chem.MolFromSmarts("[CX2]#[CX2H1]")
    if alkyne and mol.GetSubstructMatches(alkyne):
        mbi_risk = True
        mbi_reasons.append("Terminal alkyne — potential MBI via ketene intermediate")
    # Methylenedioxy
    mdp = Chem.MolFromSmarts("c1cc2OCOc2cc1")
    if mdp and mol.GetSubstructMatches(mdp):
        mbi_risk = True
        mbi_reasons.append("Methylenedioxy group — known CYP MBI risk (carbene formation)")

    # --- Overall DDI risk ---
    high_risk_cyps = [k for k, v in cyp_profile.items() if v["inhibition_risk"] == "high"]
    moderate_risk_cyps = [k for k, v in cyp_profile.items() if v["inhibition_risk"] == "moderate"]

    if high_risk_cyps or mbi_risk:
        overall_risk = "HIGH"
    elif len(moderate_risk_cyps) >= 2:
        overall_risk = "MODERATE"
    elif moderate_risk_cyps:
        overall_risk = "LOW-MODERATE"
    else:
        overall_risk = "LOW"

    # --- Co-medication analysis ---
    comedication_analysis = None
    if comedication_smiles:
        comol = Chem.MolFromSmiles(comedication_smiles)
        if comol is not None:
            co_mw = Descriptors.MolWt(comol)
            co_logp = Crippen.MolLogP(comol)
            co_aromatic = Descriptors.NumAromaticRings(comol)

            # Check if comedication shares metabolic pathway features
            co_n_het = Chem.MolFromSmarts("[nR]")
            co_n_heterocycles = len(comol.GetSubstructMatches(co_n_het)) if co_n_het else 0
            co_basic_amine = Chem.MolFromSmarts("[NX3;!$(NC=O);!$(NS=O)]")
            co_amines = len(comol.GetSubstructMatches(co_basic_amine)) if co_basic_amine else 0

            shared_pathways = []
            if (n_heterocycles >= 2 or has_imidazole) and co_n_heterocycles >= 2:
                shared_pathways.append("CYP3A4 (both contain N-heterocycles)")
            if n_basic_amines >= 1 and co_amines >= 1:
                shared_pathways.append("CYP2D6 (both contain basic amines)")
            if logp > 3 and co_logp > 3:
                shared_pathways.append("General CYP competition (both lipophilic)")

            interaction_risk = "high" if shared_pathways else "low"

            comedication_analysis = {
                "comedication_smiles": comedication_smiles,
                "comedication_mw": round(co_mw, 1),
                "comedication_logp": round(co_logp, 2),
                "shared_metabolic_pathways": shared_pathways,
                "interaction_risk": interaction_risk,
                "recommendation": (
                    f"Monitor for interactions via {', '.join(shared_pathways)}"
                    if shared_pathways
                    else "Low structural overlap in CYP-relevant features"
                ),
            }
        else:
            comedication_analysis = {"error": f"Invalid co-medication SMILES: {comedication_smiles}"}

    # --- Summary ---
    summary_lines = [
        f"DDI risk assessment: {overall_risk}",
    ]
    if high_risk_cyps:
        summary_lines.append(f"High CYP inhibition risk: {', '.join(high_risk_cyps)}")
    if moderate_risk_cyps:
        summary_lines.append(f"Moderate CYP inhibition risk: {', '.join(moderate_risk_cyps)}")
    if mbi_risk:
        summary_lines.append(f"Mechanism-based inhibition risk: {'; '.join(mbi_reasons)}")
    if motif_flags["has_azole"]:
        summary_lines.append("Contains azole motif — strong CYP inhibitor pharmacophore")
    if comedication_analysis and isinstance(comedication_analysis, dict) and "shared_metabolic_pathways" in comedication_analysis:
        if comedication_analysis["shared_metabolic_pathways"]:
            summary_lines.append(f"Co-medication interaction via: {', '.join(comedication_analysis['shared_metabolic_pathways'])}")
        else:
            summary_lines.append("Low metabolic pathway overlap with co-medication")

    result = {
        "summary": "\n".join(summary_lines),
        "overall_risk": overall_risk,
        "cyp_profile": cyp_profile,
        "motif_flags": motif_flags,
        "mechanism_based_inhibition": {
            "risk": mbi_risk,
            "reasons": mbi_reasons if mbi_reasons else ["No MBI structural alerts"],
        },
    }

    if comedication_analysis:
        result["comedication_analysis"] = comedication_analysis

    return result


@registry.register(
    name="safety.admetlab3_predict",
    description="Predict 119 ADMET endpoints for a compound using the ADMETlab 3.0 web API",
    category="safety",
    parameters={
        "smiles": "SMILES string or compound name (will be resolved)",
        "endpoints": "Specific ADMET endpoint categories to include (optional list, e.g. ['absorption', 'toxicity'])",
    },
    requires_data=[],
    usage_guide="You need ML-predicted ADMET properties from ADMETlab 3.0 — covers 119 endpoints across absorption, distribution, metabolism, excretion, and toxicity. More comprehensive than the local heuristic safety.admet_predict. Use for detailed ADMET profiling when API access is available.",
)
def admetlab3_predict(smiles: str, endpoints: list = None, **kwargs) -> dict:
    """Query ADMETlab 3.0 API for comprehensive ADMET predictions (119 endpoints)."""
    from ct.tools.chemistry import _extract_smiles
    smiles = _extract_smiles(smiles)

    if not smiles or not smiles.strip():
        return {"error": "SMILES string is required", "summary": "ADMETlab3 requires a SMILES string"}

    api_url = "https://admetlab3.scbdd.com/server/api/aio"

    data, error = request_json(
        "POST", api_url,
        json={"smiles": smiles},
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=60, retries=2,
    )
    if error:
        return {"error": f"ADMETlab3 API failed: {error}", "summary": f"ADMETlab3 prediction failed: {error}"}

    if not isinstance(data, dict):
        return {"error": "Unexpected ADMETlab3 response format", "summary": "ADMETlab3 returned unexpected format"}

    # ADMETlab3 returns predictions grouped by category
    # Categories: Absorption, Distribution, Metabolism, Excretion, Toxicity
    category_map = {
        "absorption": ["Caco-2", "HIA", "Pgp-inh", "Pgp-sub", "MDCK", "F(20%)", "F(30%)"],
        "distribution": ["PPB", "BBB", "VDss", "Fu"],
        "metabolism": ["CYP1A2-inh", "CYP1A2-sub", "CYP2C19-inh", "CYP2C19-sub",
                       "CYP2C9-inh", "CYP2C9-sub", "CYP2D6-inh", "CYP2D6-sub",
                       "CYP3A4-inh", "CYP3A4-sub"],
        "excretion": ["CL", "T12"],
        "toxicity": ["hERG", "AMES", "DILI", "Carcinogenicity", "SkinSen",
                      "Respiratory", "LD50", "BCF", "IGC50", "LC50FM", "LC50DM"],
    }

    # Parse and organize predictions
    predictions = {}
    flagged_liabilities = []
    n_endpoints = 0

    for key, value in data.items():
        if key in ("smiles", "status", "error", "msg"):
            continue
        n_endpoints += 1

        # Determine category
        category = "other"
        key_lower = key.lower()
        for cat, keywords in category_map.items():
            if any(kw.lower() in key_lower for kw in keywords):
                category = cat
                break

        # Filter by requested endpoints
        if endpoints:
            endpoint_lower = [e.lower() for e in endpoints]
            if category not in endpoint_lower and key_lower not in endpoint_lower:
                continue

        if category not in predictions:
            predictions[category] = {}

        # Parse value
        pred_value = value
        if isinstance(value, dict):
            pred_value = value.get("value", value.get("prediction", value))
        predictions[category][key] = pred_value

        # Flag liabilities
        if isinstance(pred_value, (int, float)):
            # Common liability thresholds
            if "herg" in key_lower and pred_value > 0.5:
                flagged_liabilities.append(f"hERG inhibition risk (score={pred_value:.2f})")
            elif "ames" in key_lower and pred_value > 0.5:
                flagged_liabilities.append(f"AMES mutagenicity risk (score={pred_value:.2f})")
            elif "dili" in key_lower and pred_value > 0.5:
                flagged_liabilities.append(f"DILI risk (score={pred_value:.2f})")
        elif isinstance(pred_value, str):
            if pred_value.lower() in ("positive", "active", "toxic", "yes", "1"):
                flagged_liabilities.append(f"{key}: {pred_value}")

    # Overall verdict
    if len(flagged_liabilities) == 0:
        verdict = "FAVORABLE"
    elif len(flagged_liabilities) <= 2:
        verdict = "ACCEPTABLE"
    else:
        verdict = "UNFAVORABLE"

    flag_str = "; ".join(flagged_liabilities[:5]) if flagged_liabilities else "None"
    cat_counts = ", ".join(f"{k}={len(v)}" for k, v in sorted(predictions.items()))

    return {
        "summary": (
            f"ADMETlab3 for {smiles}: {n_endpoints} endpoints predicted, "
            f"verdict={verdict}. Liabilities: {flag_str}. Categories: {cat_counts}"
        ),
        "smiles": smiles,
        "verdict": verdict,
        "n_endpoints": n_endpoints,
        "predictions": predictions,
        "flagged_liabilities": flagged_liabilities,
    }


@registry.register(
    name="safety.admetai_predict",
    description="Predict ADMET properties using ADMET-AI (top of TDC leaderboard, 41 endpoints)",
    category="safety",
    parameters={
        "smiles": "SMILES string for the compound",
        "endpoints": "Optional list of specific endpoint names to return",
    },
    requires_data=[],
    usage_guide="You need fast local ADMET predictions using ADMET-AI, the #1 model on the TDC benchmark (41 endpoints). Complements ADMETlab 3.0 (web API) with local predictions. No API key or internet needed once installed.",
)
def admetai_predict(smiles: str, endpoints: list = None, **kwargs) -> dict:
    """Predict ADMET properties locally using the ADMET-AI package."""
    from ct.tools.chemistry import _extract_smiles
    smiles = _extract_smiles(smiles)

    if not smiles or not smiles.strip():
        return {"error": "SMILES string is required", "summary": "ADMET-AI requires a SMILES string"}

    try:
        from admet_ai import ADMETModel
    except ImportError:
        return {
            "error": "admet_ai package not installed",
            "summary": "ADMET-AI not installed. Install with: pip install admet-ai",
            "install_instructions": {"pip": "pip install admet-ai"},
        }

    try:
        model = ADMETModel()
        preds = model.predict(smiles=smiles)
    except Exception as e:
        return {"error": f"ADMET-AI prediction failed: {e}", "summary": f"ADMET-AI error: {e}"}

    # Parse predictions
    if isinstance(preds, dict):
        all_preds = preds
    elif hasattr(preds, "to_dict"):
        all_preds = preds.to_dict()
        if all_preds and isinstance(next(iter(all_preds.values()), None), dict):
            first_key = next(iter(next(iter(all_preds.values())).keys()), None)
            all_preds = {k: v.get(first_key, v) for k, v in all_preds.items()}
    else:
        all_preds = {}

    # Filter endpoints if requested
    if endpoints:
        endpoint_lower = {e.lower() for e in endpoints}
        filtered = {k: v for k, v in all_preds.items() if k.lower() in endpoint_lower}
        if filtered:
            all_preds = filtered

    # Flag liabilities
    flagged = []
    for key, value in all_preds.items():
        if not isinstance(value, (int, float)):
            continue
        k = key.lower()
        if "herg" in k and value > 0.5:
            flagged.append(f"hERG inhibition ({key}={value:.2f})")
        elif "ames" in k and value > 0.5:
            flagged.append(f"AMES mutagenicity ({key}={value:.2f})")
        elif "dili" in k and value > 0.5:
            flagged.append(f"DILI risk ({key}={value:.2f})")
        elif "ld50" in k and isinstance(value, (int, float)) and value < 500:
            flagged.append(f"Low LD50 ({key}={value:.1f} mg/kg)")

    verdict = "FAVORABLE" if not flagged else "ACCEPTABLE" if len(flagged) <= 2 else "UNFAVORABLE"
    flag_str = "; ".join(flagged[:5]) if flagged else "None"

    return {
        "summary": f"ADMET-AI for {smiles}: {len(all_preds)} endpoints, verdict={verdict}. Flags: {flag_str}",
        "smiles": smiles,
        "verdict": verdict,
        "n_endpoints": len(all_preds),
        "predictions": all_preds,
        "flagged_liabilities": flagged,
    }


@registry.register(
    name="safety.chemprop_predict",
    description="Predict molecular properties using Chemprop message-passing neural networks",
    category="safety",
    parameters={
        "smiles": "SMILES string (or comma-separated list for batch)",
        "model_path": "Path to trained Chemprop model directory",
        "task": "Prediction task type: 'classification' or 'regression' (default inferred from model)",
    },
    requires_data=[],
    usage_guide="You have a trained Chemprop model and want to predict molecular properties. Chemprop v2 is the backbone of ADMET-AI and many pharma internal models. Use for custom endpoints when pre-trained models are available.",
)
def chemprop_predict(smiles: str, model_path: str = None, task: str = None, **kwargs) -> dict:
    """Predict properties using Chemprop message-passing neural networks."""
    if not smiles or not smiles.strip():
        return {"error": "SMILES string is required", "summary": "Chemprop requires SMILES input"}

    if not model_path:
        return {
            "error": "model_path is required — path to a trained Chemprop model directory",
            "summary": "Chemprop requires a model_path to a trained model",
        }

    try:
        import chemprop
    except ImportError:
        return {
            "error": "chemprop package not installed",
            "summary": "Chemprop not installed. Install with: pip install chemprop",
            "install_instructions": {"pip": "pip install chemprop"},
        }

    import os
    if not os.path.exists(model_path):
        return {"error": f"Model path not found: {model_path}", "summary": f"Model not found: {model_path}"}

    # Parse SMILES list
    smiles_list = [s.strip() for s in smiles.split(",") if s.strip()]

    try:
        # Chemprop v2 API
        preds = chemprop.predict(
            smiles=smiles_list,
            model_path=model_path,
        )

        results = []
        for smi, pred in zip(smiles_list, preds):
            if isinstance(pred, (list, tuple)):
                pred_values = [float(p) for p in pred]
            else:
                pred_values = [float(pred)]

            results.append({
                "smiles": smi,
                "predictions": pred_values,
                "prediction": pred_values[0] if len(pred_values) == 1 else pred_values,
            })

        return {
            "summary": f"Chemprop predictions for {len(smiles_list)} molecule(s) from {os.path.basename(model_path)}",
            "model_path": model_path,
            "n_molecules": len(smiles_list),
            "results": results,
        }
    except Exception as e:
        return {"error": f"Chemprop prediction failed: {e}", "summary": f"Chemprop error: {e}"}
