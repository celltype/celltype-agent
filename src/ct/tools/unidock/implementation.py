"""Uni-Dock — GPU-accelerated molecular docking.

Ultra-large virtual screening, orders of magnitude faster than Vina.
Hardware: T4/A10G GPU, 16GB VRAM.
"""
import time
import os
import subprocess


def run(receptor_pdb: str = "", ligand_sdf: str = "", center_x: float = 0, center_y: float = 0,
        center_z: float = 0, size_x: float = 25, size_y: float = 25, size_z: float = 25,
        exhaustiveness: int = 128, session_id: str = "", **kwargs) -> dict:
    if not receptor_pdb:
        return {"summary": "Error: No receptor file provided.", "error": "no_receptor"}
    if not ligand_sdf:
        return {"summary": "Error: No ligand file provided.", "error": "no_ligand"}
    if not os.path.isfile(receptor_pdb):
        return {"summary": f"Error: Receptor not found: {receptor_pdb}", "error": "file_not_found"}
    if not os.path.isfile(ligand_sdf):
        return {"summary": f"Error: Ligand not found: {ligand_sdf}", "error": "file_not_found"}

    t0 = time.time()
    output_dir = "/workspace/output"
    os.makedirs(output_dir, exist_ok=True)
    output_sdf = os.path.join(output_dir, "docked.sdf")

    cmd = [
        "unidock",
        "--receptor", receptor_pdb,
        "--gpu_batch", ligand_sdf,
        "--center_x", str(center_x),
        "--center_y", str(center_y),
        "--center_z", str(center_z),
        "--size_x", str(size_x),
        "--size_y", str(size_y),
        "--size_z", str(size_z),
        "--exhaustiveness", str(exhaustiveness),
        "--dir", output_dir,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=540)
        t_total = time.time() - t0

        if proc.returncode != 0:
            return {"summary": f"Uni-Dock failed: {proc.stderr[:500]}", "error": proc.stderr[:500]}

        # Parse docked results
        docked_files = [f for f in os.listdir(output_dir) if f.endswith(".sdf") or f.endswith(".pdbqt")]
        n_docked = len(docked_files)

        # Extract scores from output
        scores = []
        for line in proc.stdout.split("\n"):
            if "affinity" in line.lower() or "score" in line.lower():
                parts = line.split()
                for p in parts:
                    try:
                        scores.append(float(p))
                    except ValueError:
                        pass

        if session_id:
            workspace = f"/vol/workspace/{session_id}"
            os.makedirs(workspace, exist_ok=True)
            import shutil
            for f in docked_files:
                shutil.copy2(os.path.join(output_dir, f), workspace)

        best_score = min(scores) if scores else None

        return {
            "summary": (
                f"Uni-Dock: {n_docked} pose(s) generated in {t_total:.1f}s. "
                f"Best affinity: {best_score:.2f} kcal/mol" if best_score else
                f"Uni-Dock: {n_docked} pose(s) generated in {t_total:.1f}s."
            ),
            "n_docked": n_docked,
            "best_score": best_score,
            "scores": scores[:20],
            "output_files": docked_files,
            "metrics": {"time_total_s": round(t_total, 2)},
        }
    except subprocess.TimeoutExpired:
        return {"summary": "Uni-Dock timed out.", "error": "timeout"}
    except Exception as e:
        return {"summary": f"Uni-Dock error: {e}", "error": str(e)}
