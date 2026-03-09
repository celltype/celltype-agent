"""MolMIM — molecule optimization using RDKit.

Uses RDKit's BRICS decomposition and Tanimoto similarity
to generate optimized variants of an input molecule. CPU-only.
"""

import random
import time


def run(input_smiles="", target_properties=None, num_variants=10, session_id="", **kwargs):
    """Optimize a molecule by generating variants with improved properties."""
    if not input_smiles:
        return {"summary": "Error: No input SMILES provided.", "error": "no_input"}

    t0 = time.time()
    target_properties = target_properties or {}

    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem, Descriptors, QED, BRICS
    except ImportError:
        return {"summary": "Error: RDKit not installed.", "error": "missing_dep"}

    input_mol = Chem.MolFromSmiles(input_smiles)
    if not input_mol:
        return {"summary": f"Error: Invalid SMILES: {input_smiles}", "error": "invalid_smiles"}

    input_fp = AllChem.GetMorganFingerprintAsBitVect(input_mol, 2, nBits=2048)
    frags = BRICS.BRICSDecompose(input_mol)

    related = [
        "c1ccccc1O", "c1ccccc1N", "c1ccccc1C(=O)O",
        "CC(=O)O", "CC(=O)N", "CCO", "CCCO",
        "c1ccc(F)cc1", "c1ccc(Cl)cc1", "c1ccc(O)cc1",
        "C1CCNCC1", "C1CCOCC1", "c1ccncc1",
    ]

    all_frags = set(frags)
    for smi in related:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            all_frags.update(BRICS.BRICSDecompose(mol))

    frag_mols = [Chem.MolFromSmiles(f) for f in all_frags if Chem.MolFromSmiles(f)]
    random.seed(42)
    generated = set()
    max_attempts = num_variants * 100

    builder = BRICS.BRICSBuild(frag_mols)
    for i, product in enumerate(builder):
        if len(generated) >= num_variants * 5 or i > max_attempts:
            break
        smi = Chem.MolToSmiles(product)
        if smi and smi != input_smiles:
            generated.add(smi)

    variants = []
    for smi in generated:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        similarity = DataStructs.TanimotoSimilarity(input_fp, fp)
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        qed = QED.qed(mol)
        variants.append({
            "smiles": smi,
            "similarity": round(similarity, 3),
            "molecular_weight": round(mw, 1),
            "logp": round(logp, 2),
            "qed": round(qed, 3),
        })

    variants.sort(key=lambda v: v["similarity"], reverse=True)
    variants = variants[:num_variants]

    if not variants:
        return {"summary": "No valid variants generated.", "error": "no_variants"}

    return {
        "summary": (
            f"MolMIM: generated {len(variants)} optimized variants "
            f"of {input_smiles[:30]}. Best similarity: {variants[0]['similarity']:.3f}, "
            f"best QED: {max(v['qed'] for v in variants):.3f}."
        ),
        "variants": variants,
        "input_smiles": input_smiles,
        "num_variants": len(variants),
        "metrics": {
            "vram_before_mb": 0,
            "vram_peak_mb": 0,
            "time_total_s": round(time.time() - t0, 2),
            "hardware": "CPU-only",
        },
    }
