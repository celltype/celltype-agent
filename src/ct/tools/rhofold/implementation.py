"""RhoFold+ — RNA 3D structure prediction from sequence.

Outperforms human experts on RNA-Puzzles benchmark.
Hardware: A10G GPU, 16GB VRAM.
"""
import time
import os


def run(sequence: str = "", name: str = "rna_pred", session_id: str = "", **kwargs) -> dict:
    if not sequence:
        return {"summary": "Error: No RNA sequence provided.", "error": "no_sequence"}

    # Check if sequence is a file path
    if os.path.isfile(sequence):
        with open(sequence) as f:
            lines = f.readlines()
            seq_lines = [l.strip() for l in lines if not l.startswith(">")]
            sequence = "".join(seq_lines)

    clean_seq = sequence.strip().upper().replace(" ", "").replace("\n", "")
    valid_bases = set("AUGCTN")
    invalid = set(clean_seq) - valid_bases
    if invalid:
        return {"summary": f"Error: Invalid RNA bases: {invalid}", "error": f"invalid_bases: {invalid}"}

    seq_len = len(clean_seq)

    t0 = time.time()

    try:
        import torch
        import sys
        sys.path.insert(0, "/opt/RhoFold")
        from rhofold.rhofold import RhoFoldModel
        from rhofold.config import rhofold_config
    except ImportError:
        return {
            "summary": f"RhoFold+ not available for {seq_len}-nt RNA.",
            "error": "rhofold not installed",
        }

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = RhoFoldModel(rhofold_config)
        model = model.to(device)
        model.eval()

        output_dir = "/workspace/output"
        os.makedirs(output_dir, exist_ok=True)

        # Write temporary FASTA
        fasta_path = os.path.join(output_dir, f"{name}.fasta")
        with open(fasta_path, "w") as f:
            f.write(f">{name}\n{clean_seq}\n")

        with torch.no_grad():
            output = model.predict(fasta_path, output_dir)

        t_total = time.time() - t0

        pdb_path = os.path.join(output_dir, f"{name}.pdb")
        has_pdb = os.path.isfile(pdb_path)

        pdb_content = ""
        if has_pdb:
            with open(pdb_path) as f:
                pdb_content = f.read()

        confidence = None
        if hasattr(output, "plddt"):
            import numpy as np
            confidence = round(float(np.mean(output.plddt)), 2)

        if session_id and has_pdb:
            workspace = f"/vol/workspace/{session_id}"
            os.makedirs(workspace, exist_ok=True)
            import shutil
            shutil.copy2(pdb_path, workspace)

        return {
            "summary": (
                f"RhoFold+ prediction for {seq_len}-nt RNA. "
                f"{'Confidence: ' + str(confidence) if confidence else ''} "
                f"Generated in {t_total:.1f}s."
            ),
            "sequence_length": seq_len,
            "pdb_content": pdb_content[:5000] if pdb_content else None,
            "confidence": confidence,
            "output_pdb": pdb_path if has_pdb else None,
            "metrics": {"time_total_s": round(t_total, 2)},
        }
    except Exception as e:
        return {"summary": f"RhoFold+ prediction failed: {e}", "error": str(e)}
