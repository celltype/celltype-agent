"""Drug repurposing tools: connectivity map queries, signature matching."""

import numpy as np
from ct.tools import registry
from ct.tools.http_client import request_json


def _to_float(value):
    """Best-effort float conversion for heterogeneous API payloads."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_l1000fwd_hits(payload: dict | list, mode: str) -> list:
    """Extract mode-specific hits from L1000FWD response payload."""
    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        return []

    if mode == "reverse":
        primary_keys = ("opposite", "reverse", "discordant", "anti")
    else:
        primary_keys = ("similar", "mimic", "concordant")

    secondary_keys = ("results", "topn", "data")

    for key in primary_keys:
        hits = payload.get(key)
        if isinstance(hits, list):
            return hits

    for section in secondary_keys:
        nested = payload.get(section)
        if not isinstance(nested, dict):
            continue
        for key in primary_keys:
            hits = nested.get(key)
            if isinstance(hits, list):
                return hits

    return []


def _normalize_l1000fwd_hit(hit, rank: int) -> dict:
    """Normalize one L1000FWD hit into a stable ct-friendly shape."""
    if not isinstance(hit, dict):
        return {
            "rank": rank,
            "compound": str(hit),
            "connectivity_score": None,
            "raw": hit,
        }

    compound = (
        hit.get("pert_iname")
        or hit.get("name")
        or hit.get("drug")
        or hit.get("sig_id")
        or hit.get("id")
        or f"hit_{rank}"
    )
    score = (
        _to_float(hit.get("score"))
        or _to_float(hit.get("combined_score"))
        or _to_float(hit.get("combined_scores"))
        or _to_float(hit.get("tau"))
        or _to_float(hit.get("zscore"))
        or _to_float(hit.get("zscores"))
    )
    p_value = _to_float(hit.get("p_value")) or _to_float(hit.get("pval")) or _to_float(hit.get("pvals"))
    q_value = _to_float(hit.get("q_value")) or _to_float(hit.get("qval")) or _to_float(hit.get("qvals"))

    normalized = {
        "rank": rank,
        "compound": compound,
        "connectivity_score": score,
    }
    if p_value is not None:
        normalized["p_value"] = p_value
    if q_value is not None:
        normalized["q_value"] = q_value
    normalized["raw"] = hit
    return normalized


def _query_l1000fwd(up_genes: list[str], down_genes: list[str], mode: str, top_n: int) -> tuple[list, str | None]:
    """Query free L1000FWD API and return normalized hits."""
    if not up_genes and not down_genes:
        return [], "Signature is empty after separating up/down genes"

    search_payload = {
        "up_genes": up_genes,
        "down_genes": down_genes,
    }
    search_data, search_error = request_json(
        "POST",
        "https://maayanlab.cloud/L1000FWD/sig_search",
        json=search_payload,
        timeout=45,
        retries=1,
    )
    if search_error:
        return [], f"L1000FWD sig_search failed: {search_error}"
    if not isinstance(search_data, dict):
        return [], "L1000FWD sig_search returned unexpected payload"

    result_id = search_data.get("result_id") or search_data.get("id")
    if not result_id:
        return [], "L1000FWD did not return a result_id"

    result_data, result_error = request_json(
        "GET",
        f"https://maayanlab.cloud/L1000FWD/result/topn/{result_id}",
        timeout=45,
        retries=1,
    )
    if result_error:
        return [], f"L1000FWD topn lookup failed for {result_id}: {result_error}"

    raw_hits = _extract_l1000fwd_hits(result_data, mode=mode)
    if not raw_hits:
        return [], f"L1000FWD returned no {mode} hits"

    hits = [_normalize_l1000fwd_hit(hit, rank=i + 1) for i, hit in enumerate(raw_hits[:top_n])]
    return hits, None


@registry.register(
    name="repurposing.cmap_query",
    description="Query for drug repurposing opportunities using L1000 connectivity map signature matching",
    category="repurposing",
    parameters={
        "gene_signature": "Dict of gene:value (expression changes), used as query signature",
        "compound_id": "Compound ID to use as query (pulls L1000 signature; alternative to gene_signature)",
        "mode": "'similar' (same mechanism) or 'reverse' (opposing signature, e.g. disease reversal)",
        "top_n": "Number of top hits to return (default 20)",
        "allow_remote": "If true and local L1000 is unavailable, try free L1000FWD API fallback",
    },
    requires_data=[],
    usage_guide="You want to find drug repurposing opportunities — compounds with similar signatures "
                "(shared mechanism) or reverse signatures (disease-reversing). Provide a gene signature "
                "dict or compound_id. Use mode='reverse' to find compounds that reverse a disease signature. "
                "Works best with local L1000 data; if unavailable, can query the free L1000FWD API.",
)
def cmap_query(gene_signature: dict = None, compound_id: str = None,
               mode: str = "similar", top_n: int = 20,
               allow_remote: bool = True, **kwargs) -> dict:
    """Query for drug repurposing via connectivity map signature matching.

    Strategy:
    1. If compound_id provided + L1000 loaded: correlate that compound's signature
       with all other compounds in the L1000 matrix.
    2. If gene_signature provided + L1000 loaded: correlate the query signature
       against all compounds in L1000.
    3. If L1000 not available: return guidance for external CMap queries.

    Parameters
    ----------
    gene_signature : dict, optional
        Query signature as {gene_name: expression_value}. Positive values = upregulated,
        negative = downregulated.
    compound_id : str, optional
        Compound ID to pull signature from L1000 data.
    mode : str
        "similar" to find compounds with correlated signatures (shared mechanism).
        "reverse" to find compounds with anti-correlated signatures (disease reversal).
    top_n : int
        Number of top hits to return.
    """
    from scipy import stats as sp_stats

    if gene_signature is None and compound_id is None:
        return {"error": "Provide either gene_signature (dict) or compound_id", "summary": "Provide either gene_signature (dict) or compound_id"}
    if mode not in ("similar", "reverse"):
        return {"error": f"Unknown mode '{mode}'. Use 'similar' or 'reverse'", "summary": f"Unknown mode '{mode}'. Use 'similar' or 'reverse'"}
    # Try to load L1000 data
    l1000 = None
    try:
        from ct.data.loaders import load_l1000
        l1000 = load_l1000()
    except (FileNotFoundError, ImportError):
        pass

    # Build query vector
    if compound_id is not None and l1000 is not None:
        if compound_id not in l1000.index:
            return {"error": f"Compound {compound_id} not found in L1000 data", "summary": f"Compound {compound_id} not found in L1000 data"}
        query_series = l1000.loc[compound_id]
        query_genes = set(query_series.index)
        source = f"compound {compound_id} (L1000)"
    elif gene_signature is not None:
        query_genes = set(gene_signature.keys())
        source = f"provided signature ({len(query_genes)} genes)"
        query_series = None  # handled below
    else:
        return {"error": "compound_id not found in L1000 and no gene_signature provided", "summary": "compound_id not found in L1000 and no gene_signature provided"}
    # If L1000 available, do full matrix correlation
    results = None
    if l1000 is not None:
        if query_series is not None:
            # compound_id mode: correlate against all other compounds
            query_vec = query_series.values
            compounds = l1000.index.tolist()
            results = []

            for cpd in compounds:
                if cpd == compound_id:
                    continue
                other_vec = l1000.loc[cpd].values
                # Pearson correlation
                valid = ~(np.isnan(query_vec) | np.isnan(other_vec))
                if valid.sum() < 10:
                    continue
                r, p = sp_stats.pearsonr(query_vec[valid], other_vec[valid])
                results.append({
                    "compound": cpd,
                    "correlation": round(float(r), 4),
                    "p_value": float(p),
                    "n_genes": int(valid.sum()),
                })
        else:
            # gene_signature mode: correlate provided signature against L1000 compounds
            common_genes = sorted(query_genes & set(l1000.columns))
            if len(common_genes) >= 10:
                query_vec = np.array([gene_signature[g] for g in common_genes])
                compounds = l1000.index.tolist()
                results = []

                for cpd in compounds:
                    other_vec = l1000.loc[cpd, common_genes].values.astype(float)
                    valid = ~(np.isnan(query_vec) | np.isnan(other_vec))
                    if valid.sum() < 10:
                        continue
                    r, p = sp_stats.pearsonr(query_vec[valid], other_vec[valid])
                    results.append({
                        "compound": cpd,
                        "correlation": round(float(r), 4),
                        "p_value": float(p),
                        "n_genes": int(valid.sum()),
                    })
            # else: results stays None, fall through to external guidance

    if results is not None:
        if not results:
            return {
                "summary": "No correlations computed -- insufficient overlapping data",
                "source": source,
                "mode": mode,
                "hits": [],
            }

        # Sort by correlation
        if mode == "similar":
            results.sort(key=lambda x: x["correlation"], reverse=True)
        else:  # reverse
            results.sort(key=lambda x: x["correlation"])

        top_hits = results[:top_n]

        # Classify hits
        for hit in top_hits:
            r = hit["correlation"]
            if abs(r) > 0.5:
                hit["strength"] = "strong"
            elif abs(r) > 0.3:
                hit["strength"] = "moderate"
            else:
                hit["strength"] = "weak"

        # Summary
        if mode == "similar":
            desc = "similar mechanism (positively correlated)"
            best_r = top_hits[0]["correlation"] if top_hits else 0
        else:
            desc = "signature-reversing (negatively correlated)"
            best_r = top_hits[0]["correlation"] if top_hits else 0

        strong = sum(1 for h in top_hits if h["strength"] == "strong")
        moderate = sum(1 for h in top_hits if h["strength"] == "moderate")

        top3_str = ", ".join(
            f"{h['compound']}(r={h['correlation']:.3f})" for h in top_hits[:3]
        )

        summary = (
            f"CMap query ({source}, mode={mode}):\n"
            f"Searching for: {desc}\n"
            f"Top {len(top_hits)} hits: {strong} strong, {moderate} moderate\n"
            f"Best matches: {top3_str}\n"
            f"Best correlation: {best_r:.4f}"
        )

        return {
            "summary": summary,
            "source": source,
            "mode": mode,
            "n_compounds_screened": len(results),
            "hits": top_hits,
        }

    # L1000 not available -- remote fallback for gene signatures
    if gene_signature is not None:
        # Separate into up/down gene lists for external tools
        up_genes = sorted([g for g, v in gene_signature.items() if v > 0])
        down_genes = sorted([g for g, v in gene_signature.items() if v < 0])
        remote_error = None

        if allow_remote:
            remote_hits, remote_error = _query_l1000fwd(
                up_genes=up_genes,
                down_genes=down_genes,
                mode=mode,
                top_n=top_n,
            )
            if remote_hits:
                top3_str = ", ".join(
                    f"{h['compound']}(score={h['connectivity_score']:.3f})"
                    if isinstance(h.get("connectivity_score"), float)
                    else h["compound"]
                    for h in remote_hits[:3]
                )
                summary = (
                    f"Remote CMap query via L1000FWD ({source}, mode={mode}): "
                    f"{len(remote_hits)} hit(s) returned. "
                    f"Top hits: {top3_str}"
                )
                return {
                    "summary": summary,
                    "source": source,
                    "mode": mode,
                    "remote_source": "L1000FWD",
                    "local_data_unavailable": True,
                    "remote_used": True,
                    "hits": remote_hits,
                    "up_genes": up_genes[:100],
                    "down_genes": down_genes[:100],
                }

        remote_note = f"Remote fallback attempt failed: {remote_error}\n" if (allow_remote and remote_error) else ""
        summary = (
            f"L1000 data not loaded -- cannot compute correlations locally.\n"
            f"Query signature: {len(up_genes)} up, {len(down_genes)} down genes.\n"
            f"{remote_note}"
            f"For external CMap query, use these gene lists at:\n"
            f"  - CLUE (https://clue.io/query)\n"
            f"  - SigCom LINCS (https://maayanlab.cloud/sigcom-lincs/)\n"
            f"  - L1000FWD (https://maayanlab.cloud/L1000FWD/)\n"
            f"Upload up-genes and down-genes separately."
        )

        return {
            "summary": summary,
            "source": source,
            "mode": mode,
            "up_genes": up_genes[:100],  # cap for readability
            "down_genes": down_genes[:100],
            "data_unavailable": True,
            "remote_used": False,
            "remote_error": remote_error if allow_remote else None,
            "external_resources": [
                {"name": "CLUE", "url": "https://clue.io/query"},
                {"name": "SigCom LINCS", "url": "https://maayanlab.cloud/sigcom-lincs/"},
                {"name": "L1000FWD", "url": "https://maayanlab.cloud/L1000FWD/"},
            ],
            "hits": [],
        }

    return {
        "summary": "L1000 data not available and no gene_signature provided for external query",
        "error": "Load L1000 data (ct data pull l1000) or provide a gene_signature dict",
        "hits": [],
    }


@registry.register(
    name="repurposing.txgnn_predict",
    description="Zero-shot drug repurposing prediction using TxGNN biomedical knowledge graph model",
    category="repurposing",
    parameters={
        "disease": "Disease name or MONDO ontology ID (e.g. 'Alzheimer disease' or 'MONDO:0004975')",
        "top_k": "Number of top drug predictions to return (default 20)",
        "drug_filter": "Optional drug name or list to filter results (e.g. 'metformin')",
    },
    requires_data=[],
    usage_guide="You want zero-shot drug repurposing predictions for a disease. TxGNN (Nature Medicine 2024, Harvard/Zitnik lab) achieves 49.2% improvement on indication prediction over prior SOTA. Uses biomedical knowledge graph with 17,080 diseases and 7,957 drugs. First run downloads ~2GB model.",
)
def txgnn_predict(disease: str, top_k: int = 20, drug_filter: str = None, **kwargs) -> dict:
    """Zero-shot drug repurposing via TxGNN knowledge graph model."""
    disease = (disease or "").strip()
    if not disease:
        return {"error": "Disease name or MONDO ID is required", "summary": "TxGNN requires a disease input"}

    top_k = max(1, min(int(top_k or 20), 100))

    try:
        from txgnn import TxGNN, TxData
    except ImportError:
        return {
            "error": "txgnn package not installed",
            "summary": "TxGNN not installed. Install with: pip install txgnn",
            "install_instructions": {
                "pip": "pip install txgnn",
                "github": "git clone https://github.com/mims-harvard/TxGNN && pip install -e .",
                "note": "First run downloads ~2GB pretrained model to ~/.celltype/models/txgnn/",
            },
        }

    import os
    cache_dir = os.path.expanduser("~/.celltype/models/txgnn")
    os.makedirs(cache_dir, exist_ok=True)

    try:
        # Initialize TxGNN with pretrained model
        txdata = TxData(data_folder_path=cache_dir)
        txdata.prepare_split(split="complex_disease", seed=42)

        model = TxGNN(data=txdata, weight_bias_track=False, proj_name="txgnn_predict", device="cpu")
        model.load_pretrained(path=os.path.join(cache_dir, "model.ckpt"))

        # Query for disease
        predictions = model.predict_drug_disease(disease_name=disease, top_k=top_k)

        if isinstance(predictions, list):
            results = []
            for i, pred in enumerate(predictions[:top_k]):
                if isinstance(pred, dict):
                    drug_name = pred.get("drug_name", pred.get("name", f"drug_{i}"))
                    score = pred.get("score", pred.get("probability", 0.0))
                else:
                    drug_name = str(pred)
                    score = 0.0

                # Apply drug filter
                if drug_filter:
                    filter_terms = [f.strip().lower() for f in drug_filter.split(",")]
                    if not any(term in drug_name.lower() for term in filter_terms):
                        continue

                results.append({
                    "rank": i + 1,
                    "drug": drug_name,
                    "score": round(float(score), 4) if isinstance(score, (int, float)) else score,
                })
        elif isinstance(predictions, dict):
            results = [{"rank": i+1, "drug": k, "score": round(float(v), 4)} for i, (k, v) in enumerate(predictions.items())]
        else:
            results = []

        filter_str = f" (filtered by '{drug_filter}')" if drug_filter else ""
        top_drugs = ", ".join(r["drug"] for r in results[:5])

        return {
            "summary": (
                f"TxGNN predictions for {disease}{filter_str}: "
                f"{len(results)} drug(s). Top: {top_drugs}"
            ),
            "disease": disease,
            "n_predictions": len(results),
            "predictions": results[:top_k],
            "model": "TxGNN (Nature Medicine 2024)",
        }
    except Exception as e:
        return {"error": f"TxGNN prediction failed: {e}", "summary": f"TxGNN error: {e}"}
