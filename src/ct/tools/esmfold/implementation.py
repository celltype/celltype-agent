"""ESMFold structure prediction implementation.

Uses Meta's ESMFold model to predict protein 3D structure from sequence.
Hardware: A10G GPU, 16GB VRAM.
"""
import time


def run(sequence: str = "", session_id: str = "", **kwargs) -> dict:
    if not sequence:
        return {"summary": "Error: No sequence provided.", "error": "no_sequence"}

    clean_seq = sequence.strip().upper().replace(" ", "").replace("\n", "")
    if clean_seq.startswith(">"):
        lines = clean_seq.split("\n")
        clean_seq = "".join(l for l in lines if not l.startswith(">"))
    seq_len = len(clean_seq)

    t0 = time.time()

    try:
        import torch
        from transformers import AutoTokenizer, EsmForProteinFolding
    except (ImportError, OSError, Exception):
        return {
            "summary": f"ESMFold: {seq_len}-residue protein (no GPU libraries).",
            "pdb_content": f"REMARK placeholder\nEND\n",
            "confidence": 0, "num_residues": seq_len,
        }

    from transformers.models.esm.openfold_utils import protein as protein_utils
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- METRICS: before model load ---
    vram_before = torch.cuda.memory_allocated() // (1024*1024) if torch.cuda.is_available() else 0

    t_load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    model = model.to(device)
    model.eval()
    t_load = time.time() - t_load_start

    # --- METRICS: after model load ---
    vram_after_load = torch.cuda.memory_allocated() // (1024*1024) if torch.cuda.is_available() else 0

    t_infer_start = time.time()
    with torch.no_grad():
        tokenized = tokenizer([clean_seq], return_tensors="pt", add_special_tokens=False)
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        output = model(**tokenized)
    t_infer = time.time() - t_infer_start

    # --- METRICS: peak after inference ---
    vram_peak = torch.cuda.max_memory_allocated() // (1024*1024) if torch.cuda.is_available() else 0

    plddt_tensor = output["plddt"][0].cpu()
    avg_plddt = plddt_tensor.mean().item() * 100

    try:
        pdb_string = model.output_to_pdb(output)[0]
    except (AttributeError, Exception):
        aatype = output["aatype"][0].cpu()
        positions = output["positions"][-1][0].cpu()
        atom_mask = output["atom37_atom_exists"][0].cpu()
        def _to_np(t):
            return t.numpy() if hasattr(t, 'numpy') else np.array(t)
        pdb_string = protein_utils.to_pdb(protein_utils.Protein(
            aatype=_to_np(aatype), atom_positions=_to_np(positions),
            atom_mask=_to_np(atom_mask), residue_index=np.arange(seq_len)+1,
            b_factors=_to_np(plddt_tensor*100), chain_index=np.zeros(seq_len, dtype=np.int32),
        ))

    t_total = time.time() - t0

    if session_id:
        import os
        workspace_dir = f"/vol/workspace/{session_id}"
        os.makedirs(workspace_dir, exist_ok=True)
        with open(f"{workspace_dir}/predicted_structure.pdb", "w") as f:
            f.write(pdb_string)

    return {
        "summary": f"ESMFold structure prediction for {seq_len}-residue protein. Average pLDDT confidence: {avg_plddt:.1f}/100.",
        "pdb_content": pdb_string,
        "confidence": avg_plddt,
        "num_residues": seq_len,
        "metrics": {
            "vram_before_model_load_mb": vram_before,
            "vram_after_model_load_mb": vram_after_load,
            "vram_peak_mb": vram_peak,
            "time_model_load_s": round(t_load, 2),
            "time_inference_s": round(t_infer, 2),
            "time_total_s": round(t_total, 2),
        },
    }
