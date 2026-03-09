"""LigandMPNN — ligand-aware protein sequence design.

Nature Methods 2025. Designs protein sequences considering bound ligands,
nucleotides, and metal ions. Hardware: T4 GPU, 8GB VRAM.
"""
import time
import os
import json
import subprocess


def run(pdb_path: str = "", chains_to_design: str = "", num_sequences: int = 10,
        temperature: float = 0.1, fixed_residues: str = "", session_id: str = "", **kwargs) -> dict:
    if not pdb_path:
        return {"summary": "Error: No PDB path provided.", "error": "no_pdb_path"}
    if not os.path.isfile(pdb_path):
        return {"summary": f"Error: PDB not found: {pdb_path}", "error": "file_not_found"}

    num_sequences = max(1, min(int(num_sequences or 10), 100))
    temperature = max(0.01, min(float(temperature or 0.1), 2.0))

    t0 = time.time()
    output_dir = "/workspace/output"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python3", "/opt/LigandMPNN/run.py",
        "--pdb_path", pdb_path,
        "--out_folder", output_dir,
        "--num_seq_per_target", str(num_sequences),
        "--sampling_temp", str(temperature),
        "--model_type", "ligand_mpnn",
        "--checkpoint_ligand_mpnn", "/opt/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt",
    ]

    if chains_to_design:
        cmd.extend(["--chains_to_design", chains_to_design])
    if fixed_residues:
        cmd.extend(["--fixed_residues", fixed_residues])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        t_total = time.time() - t0

        if proc.returncode != 0:
            return {"summary": f"LigandMPNN failed: {proc.stderr[:500]}", "error": proc.stderr[:500]}

        # Parse output FASTA
        sequences = []
        fasta_dir = os.path.join(output_dir, "seqs")
        if os.path.isdir(fasta_dir):
            for fname in sorted(os.listdir(fasta_dir)):
                if fname.endswith(".fa") or fname.endswith(".fasta"):
                    with open(os.path.join(fasta_dir, fname)) as f:
                        header = ""
                        seq = []
                        for line in f:
                            line = line.strip()
                            if line.startswith(">"):
                                if seq:
                                    sequences.append({"header": header, "sequence": "".join(seq)})
                                header = line[1:]
                                seq = []
                            else:
                                seq.append(line)
                        if seq:
                            sequences.append({"header": header, "sequence": "".join(seq)})

        if session_id:
            workspace = f"/vol/workspace/{session_id}"
            os.makedirs(workspace, exist_ok=True)
            import shutil
            if os.path.isdir(fasta_dir):
                shutil.copytree(fasta_dir, os.path.join(workspace, "seqs"), dirs_exist_ok=True)

        return {
            "summary": (
                f"LigandMPNN: {len(sequences)} sequence(s) designed for {os.path.basename(pdb_path)} "
                f"(T={temperature}) in {t_total:.1f}s."
            ),
            "pdb_path": pdb_path,
            "n_sequences": len(sequences),
            "sequences": sequences[:num_sequences],
            "temperature": temperature,
            "metrics": {"time_total_s": round(t_total, 2)},
        }
    except subprocess.TimeoutExpired:
        return {"summary": "LigandMPNN timed out (>240s).", "error": "timeout"}
    except Exception as e:
        return {"summary": f"LigandMPNN error: {e}", "error": str(e)}
