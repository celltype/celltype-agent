"""Chai-1 biomolecular structure prediction.

Predicts 3D structures for proteins, DNA, RNA, and small molecule complexes.
Hardware: A100 GPU, 40GB VRAM.
"""
import time
import os


def run(fasta_path: str = "", ligand_smiles: str = "", num_trunk_recycles: int = 3,
        num_diffusion_timesteps: int = 200, session_id: str = "", **kwargs) -> dict:
    if not fasta_path:
        return {"summary": "Error: No FASTA path provided.", "error": "no_fasta_path"}

    if not os.path.isfile(fasta_path):
        return {"summary": f"Error: FASTA file not found: {fasta_path}", "error": "file_not_found"}

    # Read FASTA to determine input composition
    sequences = []
    with open(fasta_path) as f:
        current_header = ""
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    sequences.append({"header": current_header, "sequence": "".join(current_seq)})
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            sequences.append({"header": current_header, "sequence": "".join(current_seq)})

    n_sequences = len(sequences)
    total_residues = sum(len(s["sequence"]) for s in sequences)

    t0 = time.time()

    try:
        import torch
        from chai_lab.chai1 import run_inference
    except ImportError:
        return {
            "summary": f"Chai-1: {n_sequences} sequence(s), {total_residues} residues (chai_lab not available).",
            "n_sequences": n_sequences,
            "total_residues": total_residues,
            "error": "chai_lab not installed",
        }

    try:
        candidates = run_inference(
            fasta_file=fasta_path,
            output_dir="/workspace/output",
            num_trunk_recycles=num_trunk_recycles,
            num_diffn_timesteps=num_diffusion_timesteps,
            seed=42,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        t_total = time.time() - t0

        # Extract results
        output_pdbs = []
        confidences = []
        output_dir = "/workspace/output"
        for f in sorted(os.listdir(output_dir)):
            if f.endswith(".pdb"):
                output_pdbs.append(os.path.join(output_dir, f))

        if hasattr(candidates, "aggregate_score"):
            confidences = [round(float(c.aggregate_score), 2) for c in candidates]

        # Save to session workspace
        if session_id and output_pdbs:
            workspace = f"/vol/workspace/{session_id}"
            os.makedirs(workspace, exist_ok=True)
            import shutil
            for pdb in output_pdbs:
                shutil.copy2(pdb, workspace)

        return {
            "summary": (
                f"Chai-1 prediction: {n_sequences} chain(s), {total_residues} residues. "
                f"{len(output_pdbs)} model(s) generated in {t_total:.1f}s."
            ),
            "n_sequences": n_sequences,
            "total_residues": total_residues,
            "n_models": len(output_pdbs),
            "output_pdbs": output_pdbs,
            "confidences": confidences,
            "has_ligand": bool(ligand_smiles),
            "metrics": {"time_total_s": round(t_total, 2)},
        }
    except Exception as e:
        return {"summary": f"Chai-1 prediction failed: {e}", "error": str(e)}
