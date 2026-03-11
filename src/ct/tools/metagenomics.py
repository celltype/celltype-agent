"""
Metagenomics tools: antimicrobial resistance gene classification.

Provides WHO priority tier classification for antimicrobial resistance genes
based on the WHO Bacterial Priority Pathogen List and ARG taxonomies.
"""

from ct.tools import registry


# WHO Priority Antimicrobial Resistance Genes
# Source: WHO Bacterial Priority Pathogens List 2024 + comprehensive ARG databases
# Classification: Critical > High > Medium priority pathogens and their resistance genes
_WHO_PRIORITY_ARGS = {
    # CRITICAL priority — Carbapenem-resistant and 3rd-gen cephalosporin-resistant
    # Acinetobacter baumannii (carbapenem-resistant)
    "blaOXA-23": {"tier": "critical", "pathogen": "Acinetobacter baumannii", "drug_class": "carbapenem", "mechanism": "enzymatic inactivation"},
    "blaOXA-24": {"tier": "critical", "pathogen": "Acinetobacter baumannii", "drug_class": "carbapenem", "mechanism": "enzymatic inactivation"},
    "blaOXA-48": {"tier": "critical", "pathogen": "Enterobacterales", "drug_class": "carbapenem", "mechanism": "enzymatic inactivation"},
    "blaOXA-58": {"tier": "critical", "pathogen": "Acinetobacter baumannii", "drug_class": "carbapenem", "mechanism": "enzymatic inactivation"},
    "blaOXA-51": {"tier": "critical", "pathogen": "Acinetobacter baumannii", "drug_class": "carbapenem", "mechanism": "enzymatic inactivation"},
    "blaOXA-143": {"tier": "critical", "pathogen": "Acinetobacter baumannii", "drug_class": "carbapenem", "mechanism": "enzymatic inactivation"},
    # Pseudomonas aeruginosa (carbapenem-resistant)
    "blaVIM": {"tier": "critical", "pathogen": "Pseudomonas aeruginosa", "drug_class": "carbapenem", "mechanism": "enzymatic inactivation"},
    "blaVIM-1": {"tier": "critical", "pathogen": "Pseudomonas aeruginosa", "drug_class": "carbapenem", "mechanism": "enzymatic inactivation"},
    "blaVIM-2": {"tier": "critical", "pathogen": "Pseudomonas aeruginosa", "drug_class": "carbapenem", "mechanism": "enzymatic inactivation"},
    "blaIMP": {"tier": "critical", "pathogen": "Pseudomonas aeruginosa", "drug_class": "carbapenem", "mechanism": "enzymatic inactivation"},
    "blaIMP-1": {"tier": "critical", "pathogen": "Pseudomonas aeruginosa", "drug_class": "carbapenem", "mechanism": "enzymatic inactivation"},
    "blaNDM": {"tier": "critical", "pathogen": "Enterobacterales", "drug_class": "carbapenem", "mechanism": "enzymatic inactivation"},
    "blaNDM-1": {"tier": "critical", "pathogen": "Enterobacterales", "drug_class": "carbapenem", "mechanism": "enzymatic inactivation"},
    "blaNDM-5": {"tier": "critical", "pathogen": "Enterobacterales", "drug_class": "carbapenem", "mechanism": "enzymatic inactivation"},
    "blaKPC": {"tier": "critical", "pathogen": "Enterobacterales", "drug_class": "carbapenem", "mechanism": "enzymatic inactivation"},
    "blaKPC-2": {"tier": "critical", "pathogen": "Enterobacterales", "drug_class": "carbapenem", "mechanism": "enzymatic inactivation"},
    "blaKPC-3": {"tier": "critical", "pathogen": "Enterobacterales", "drug_class": "carbapenem", "mechanism": "enzymatic inactivation"},
    # Enterobacterales 3rd-gen cephalosporin-resistant
    "blaCTX-M": {"tier": "critical", "pathogen": "Enterobacterales", "drug_class": "3rd-gen cephalosporin", "mechanism": "enzymatic inactivation"},
    "blaCTX-M-1": {"tier": "critical", "pathogen": "Enterobacterales", "drug_class": "3rd-gen cephalosporin", "mechanism": "enzymatic inactivation"},
    "blaCTX-M-14": {"tier": "critical", "pathogen": "Enterobacterales", "drug_class": "3rd-gen cephalosporin", "mechanism": "enzymatic inactivation"},
    "blaCTX-M-15": {"tier": "critical", "pathogen": "Enterobacterales", "drug_class": "3rd-gen cephalosporin", "mechanism": "enzymatic inactivation"},
    "blaCTX-M-27": {"tier": "critical", "pathogen": "Enterobacterales", "drug_class": "3rd-gen cephalosporin", "mechanism": "enzymatic inactivation"},
    "blaSHV": {"tier": "critical", "pathogen": "Enterobacterales", "drug_class": "3rd-gen cephalosporin", "mechanism": "enzymatic inactivation"},
    "blaSHV-12": {"tier": "critical", "pathogen": "Enterobacterales", "drug_class": "3rd-gen cephalosporin", "mechanism": "enzymatic inactivation"},
    "blaTEM": {"tier": "critical", "pathogen": "Enterobacterales", "drug_class": "3rd-gen cephalosporin", "mechanism": "enzymatic inactivation"},
    "blaCMY-2": {"tier": "critical", "pathogen": "Enterobacterales", "drug_class": "3rd-gen cephalosporin", "mechanism": "enzymatic inactivation"},
    "blaGES": {"tier": "critical", "pathogen": "Enterobacterales", "drug_class": "carbapenem", "mechanism": "enzymatic inactivation"},
    # Mycobacterium tuberculosis (rifampicin-resistant)
    "rpoB": {"tier": "critical", "pathogen": "Mycobacterium tuberculosis", "drug_class": "rifampicin", "mechanism": "target alteration"},
    "katG": {"tier": "critical", "pathogen": "Mycobacterium tuberculosis", "drug_class": "isoniazid", "mechanism": "target alteration"},
    "inhA": {"tier": "critical", "pathogen": "Mycobacterium tuberculosis", "drug_class": "isoniazid", "mechanism": "target alteration"},
    "embB": {"tier": "critical", "pathogen": "Mycobacterium tuberculosis", "drug_class": "ethambutol", "mechanism": "target alteration"},
    "gyrA": {"tier": "critical", "pathogen": "Mycobacterium tuberculosis", "drug_class": "fluoroquinolone", "mechanism": "target alteration"},
    "gyrB": {"tier": "critical", "pathogen": "Mycobacterium tuberculosis", "drug_class": "fluoroquinolone", "mechanism": "target alteration"},

    # HIGH priority
    # Salmonella / E. coli (fluoroquinolone-resistant)
    "qnrA": {"tier": "high", "pathogen": "Enterobacterales", "drug_class": "fluoroquinolone", "mechanism": "target protection"},
    "qnrB": {"tier": "high", "pathogen": "Enterobacterales", "drug_class": "fluoroquinolone", "mechanism": "target protection"},
    "qnrS": {"tier": "high", "pathogen": "Enterobacterales", "drug_class": "fluoroquinolone", "mechanism": "target protection"},
    "aac(6')-Ib-cr": {"tier": "high", "pathogen": "Enterobacterales", "drug_class": "fluoroquinolone", "mechanism": "enzymatic inactivation"},
    "oqxA": {"tier": "high", "pathogen": "Enterobacterales", "drug_class": "fluoroquinolone", "mechanism": "efflux pump"},
    "oqxB": {"tier": "high", "pathogen": "Enterobacterales", "drug_class": "fluoroquinolone", "mechanism": "efflux pump"},
    # Staphylococcus aureus (MRSA)
    "mecA": {"tier": "high", "pathogen": "Staphylococcus aureus", "drug_class": "methicillin/oxacillin", "mechanism": "target alteration"},
    "mecC": {"tier": "high", "pathogen": "Staphylococcus aureus", "drug_class": "methicillin/oxacillin", "mechanism": "target alteration"},
    # Enterococcus faecium (vancomycin-resistant)
    "vanA": {"tier": "high", "pathogen": "Enterococcus faecium", "drug_class": "vancomycin", "mechanism": "target alteration"},
    "vanB": {"tier": "high", "pathogen": "Enterococcus faecium", "drug_class": "vancomycin", "mechanism": "target alteration"},
    "vanC": {"tier": "high", "pathogen": "Enterococcus", "drug_class": "vancomycin", "mechanism": "target alteration"},
    "vanD": {"tier": "high", "pathogen": "Enterococcus", "drug_class": "vancomycin", "mechanism": "target alteration"},
    # Neisseria gonorrhoeae (3rd-gen cephalosporin-resistant)
    "penA": {"tier": "high", "pathogen": "Neisseria gonorrhoeae", "drug_class": "3rd-gen cephalosporin", "mechanism": "target alteration"},
    "mtrR": {"tier": "high", "pathogen": "Neisseria gonorrhoeae", "drug_class": "multiple", "mechanism": "efflux pump"},
    # Helicobacter pylori (clarithromycin-resistant)
    "23S_rRNA_A2142G": {"tier": "high", "pathogen": "Helicobacter pylori", "drug_class": "macrolide", "mechanism": "target alteration"},
    "23S_rRNA_A2143G": {"tier": "high", "pathogen": "Helicobacter pylori", "drug_class": "macrolide", "mechanism": "target alteration"},
    # Campylobacter (fluoroquinolone-resistant)
    "gyrA_T86I": {"tier": "high", "pathogen": "Campylobacter", "drug_class": "fluoroquinolone", "mechanism": "target alteration"},
    # Clostridioides difficile
    "tcdA": {"tier": "high", "pathogen": "Clostridioides difficile", "drug_class": "toxin", "mechanism": "virulence factor"},
    "tcdB": {"tier": "high", "pathogen": "Clostridioides difficile", "drug_class": "toxin", "mechanism": "virulence factor"},
    # Colistin resistance (MCR family)
    "mcr-1": {"tier": "high", "pathogen": "Enterobacterales", "drug_class": "colistin", "mechanism": "target alteration"},
    "mcr-2": {"tier": "high", "pathogen": "Enterobacterales", "drug_class": "colistin", "mechanism": "target alteration"},
    "mcr-3": {"tier": "high", "pathogen": "Enterobacterales", "drug_class": "colistin", "mechanism": "target alteration"},
    "mcr-4": {"tier": "high", "pathogen": "Enterobacterales", "drug_class": "colistin", "mechanism": "target alteration"},
    "mcr-5": {"tier": "high", "pathogen": "Enterobacterales", "drug_class": "colistin", "mechanism": "target alteration"},

    # MEDIUM priority
    # Streptococcus pneumoniae (penicillin-non-susceptible)
    "pbp1a": {"tier": "medium", "pathogen": "Streptococcus pneumoniae", "drug_class": "penicillin", "mechanism": "target alteration"},
    "pbp2b": {"tier": "medium", "pathogen": "Streptococcus pneumoniae", "drug_class": "penicillin", "mechanism": "target alteration"},
    "pbp2x": {"tier": "medium", "pathogen": "Streptococcus pneumoniae", "drug_class": "penicillin", "mechanism": "target alteration"},
    # Haemophilus influenzae (ampicillin-resistant)
    "blaTEM-1": {"tier": "medium", "pathogen": "Haemophilus influenzae", "drug_class": "ampicillin", "mechanism": "enzymatic inactivation"},
    "blaROB-1": {"tier": "medium", "pathogen": "Haemophilus influenzae", "drug_class": "ampicillin", "mechanism": "enzymatic inactivation"},
    # Additional aminoglycoside resistance
    "aph(3')-III": {"tier": "medium", "pathogen": "Gram-positive", "drug_class": "aminoglycoside", "mechanism": "enzymatic inactivation"},
    "aac(3)-II": {"tier": "medium", "pathogen": "Enterobacterales", "drug_class": "aminoglycoside", "mechanism": "enzymatic inactivation"},
    "armA": {"tier": "medium", "pathogen": "Enterobacterales", "drug_class": "aminoglycoside", "mechanism": "target alteration"},
    # Tetracycline resistance
    "tetA": {"tier": "medium", "pathogen": "Enterobacterales", "drug_class": "tetracycline", "mechanism": "efflux pump"},
    "tetB": {"tier": "medium", "pathogen": "Enterobacterales", "drug_class": "tetracycline", "mechanism": "efflux pump"},
    "tetM": {"tier": "medium", "pathogen": "multiple", "drug_class": "tetracycline", "mechanism": "target protection"},
    "tetO": {"tier": "medium", "pathogen": "multiple", "drug_class": "tetracycline", "mechanism": "target protection"},
    "tetW": {"tier": "medium", "pathogen": "multiple", "drug_class": "tetracycline", "mechanism": "target protection"},
    # Sulfonamide/trimethoprim resistance
    "sul1": {"tier": "medium", "pathogen": "multiple", "drug_class": "sulfonamide", "mechanism": "target replacement"},
    "sul2": {"tier": "medium", "pathogen": "multiple", "drug_class": "sulfonamide", "mechanism": "target replacement"},
    "dfrA1": {"tier": "medium", "pathogen": "multiple", "drug_class": "trimethoprim", "mechanism": "target replacement"},
    "dfrA12": {"tier": "medium", "pathogen": "multiple", "drug_class": "trimethoprim", "mechanism": "target replacement"},
    # Chloramphenicol resistance
    "catA1": {"tier": "medium", "pathogen": "multiple", "drug_class": "chloramphenicol", "mechanism": "enzymatic inactivation"},
    "cmlA": {"tier": "medium", "pathogen": "multiple", "drug_class": "chloramphenicol", "mechanism": "efflux pump"},
    "floR": {"tier": "medium", "pathogen": "multiple", "drug_class": "chloramphenicol", "mechanism": "efflux pump"},
    # Macrolide resistance
    "ermA": {"tier": "medium", "pathogen": "Gram-positive", "drug_class": "macrolide", "mechanism": "target alteration"},
    "ermB": {"tier": "medium", "pathogen": "Gram-positive", "drug_class": "macrolide", "mechanism": "target alteration"},
    "ermC": {"tier": "medium", "pathogen": "Staphylococcus", "drug_class": "macrolide", "mechanism": "target alteration"},
    "mefA": {"tier": "medium", "pathogen": "Streptococcus", "drug_class": "macrolide", "mechanism": "efflux pump"},
}


@registry.register(
    name="metagenomics.who_arg_classify",
    description="Classify antimicrobial resistance genes by WHO priority tier and associated pathogen/drug class",
    category="metagenomics",
    parameters={
        "genes": "Comma-separated list of ARG names (e.g. 'mecA,blaNDM-1,tetM')",
        "include_unclassified": "Include genes not in WHO priority list (default True)",
    },
    requires_data=[],
    usage_guide="You want to classify antimicrobial resistance genes detected in metagenomics data by clinical importance. "
                "Maps genes to WHO priority tiers (critical/high/medium), associated pathogens, drug classes, and resistance mechanisms. "
                "Use after running ARG detection tools (e.g. RGI, AMRFinderPlus) to prioritize findings.",
)
def who_arg_classify(genes: str, include_unclassified: bool = True, **kwargs) -> dict:
    """Classify antimicrobial resistance genes by WHO priority tier."""
    genes = (genes or "").strip()
    if not genes:
        return {"error": "Gene list is required (comma-separated ARG names)", "summary": "No genes provided for WHO ARG classification"}

    gene_list = [g.strip() for g in genes.split(",") if g.strip()]
    if not gene_list:
        return {"error": "Gene list is required (comma-separated ARG names)", "summary": "No genes provided for WHO ARG classification"}

    classified = []
    buckets = {"critical": [], "high": [], "medium": [], "unclassified": []}

    for gene in gene_list:
        # Case-sensitive lookup first
        info = _WHO_PRIORITY_ARGS.get(gene)
        if info:
            entry = {
                "gene": gene,
                "tier": info["tier"],
                "pathogen": info["pathogen"],
                "drug_class": info["drug_class"],
                "mechanism": info["mechanism"],
            }
            classified.append(entry)
            buckets[info["tier"]].append(gene)
        else:
            if include_unclassified:
                entry = {
                    "gene": gene,
                    "tier": "unclassified",
                    "pathogen": None,
                    "drug_class": None,
                    "mechanism": None,
                }
                classified.append(entry)
            buckets["unclassified"].append(gene)

    n_critical = len(buckets["critical"])
    n_high = len(buckets["high"])
    n_medium = len(buckets["medium"])
    n_unclassified = len(buckets["unclassified"])

    summary = (
        f"Classified {len(gene_list)} ARGs: "
        f"{n_critical} critical, {n_high} high, {n_medium} medium, "
        f"{n_unclassified} unclassified"
    )

    return {
        "summary": summary,
        "total_genes": len(gene_list),
        "counts": {
            "critical": n_critical,
            "high": n_high,
            "medium": n_medium,
            "unclassified": n_unclassified,
        },
        "classified_genes": classified,
        "by_tier": {
            "critical": buckets["critical"],
            "high": buckets["high"],
            "medium": buckets["medium"],
            "unclassified": buckets["unclassified"],
        },
    }
