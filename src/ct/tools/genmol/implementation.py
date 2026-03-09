"""GenMol — molecule generation using RDKit BRICS decomposition.

Uses RDKit's BRICS (Breaking of Retrosynthetically Interesting Chemical Substructures)
algorithm to decompose a scaffold and recombine fragments into novel molecules.
This is a real computational chemistry method. CPU-only.
"""

import random
import time


def run(scaffold_smiles="", target_properties=None, num_molecules=10, session_id="", **kwargs):
    """Generate novel molecules from a scaffold using BRICS decomposition."""
    t0 = time.time()
    target_properties = target_properties or {}

    try:
        from rdkit import Chem
        from rdkit.Chem import BRICS, Descriptors, QED
    except ImportError:
        return {"summary": "Error: RDKit not installed.", "error": "missing_dep"}

    seed_smiles = [
        "c1ccc(NC(=O)c2ccccc2)cc1",
        "CC(=O)Oc1ccccc1C(=O)O",
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "c1ccc2[nH]c(-c3ccccc3)nc2c1",
        "O=C(O)c1ccccc1O",
        "c1ccc(Oc2ccccc2)cc1",
        "CC(=O)Nc1ccc(O)cc1",
        "c1cnc2ccccc2n1",
    ]

    if scaffold_smiles:
        seed_smiles.insert(0, scaffold_smiles)

    all_fragments = set()
    for smi in seed_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            frags = BRICS.BRICSDecompose(mol)
            all_fragments.update(frags)

    if len(all_fragments) < 2:
        return {"summary": "Error: Could not decompose scaffold.", "error": "decomposition_failed"}

    frag_mols = [Chem.MolFromSmiles(f) for f in all_fragments if Chem.MolFromSmiles(f)]
    random.seed(42)

    generated = set()
    max_attempts = num_molecules * 50

    builder = BRICS.BRICSBuild(frag_mols)
    for i, product in enumerate(builder):
        if len(generated) >= num_molecules * 3 or i > max_attempts:
            break
        smi = Chem.MolToSmiles(product)
        if smi:
            generated.add(smi)

    molecules = []
    mw_max = target_properties.get("molecular_weight_max", 800)
    logp_max = target_properties.get("logp_max", 6)

    for smi in generated:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            continue
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        qed = QED.qed(mol)
        if mw <= mw_max and logp <= logp_max:
            molecules.append({
                "smiles": smi,
                "molecular_weight": round(mw, 1),
                "logp": round(logp, 2),
                "qed": round(qed, 3),
            })

    molecules.sort(key=lambda m: m["qed"], reverse=True)
    molecules = molecules[:num_molecules]

    if not molecules:
        return {"summary": f"No molecules passed filters.", "error": "no_valid_molecules"}

    scaffold_str = f" from scaffold {scaffold_smiles[:30]}" if scaffold_smiles else ""

    return {
        "summary": (
            f"GenMol: generated {len(molecules)} drug-like molecules{scaffold_str} "
            f"via BRICS decomposition. Best QED: {molecules[0]['qed']:.3f}."
        ),
        "molecules": molecules,
        "num_molecules": len(molecules),
        "metrics": {
            "vram_before_mb": 0,
            "vram_peak_mb": 0,
            "time_total_s": round(time.time() - t0, 2),
            "hardware": "CPU-only",
        },
    }
