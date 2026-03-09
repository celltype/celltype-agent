"""GNINA — deep learning molecular docking with CNN scoring.

CNN-based scoring outperforms Vina scoring and matches commercial tools.
Hardware: T4 GPU, 8GB VRAM.
"""
import time
import os
import subprocess


def run(receptor_pdb: str = "", ligand_sdf: str = "", autobox_ligand: str = "",
        center_x: float = 0, center_y: float = 0, center_z: float = 0,
        size_x: float = 25, size_y: float = 25, size_z: float = 25,
        exhaustiveness: int = 16, cnn_scoring: str = "default",
        session_id: str = "", **kwargs) -> dict:
    if not receptor_pdb:
        return {"summary": "Error: No receptor file provided.", "error": "no_receptor"}
    if not ligand_sdf:
        return {"summary": "Error: No ligand file provided.", "error": "no_ligand"}
    if not os.path.isfile(receptor_pdb):
        return {"summary": f"Error: Receptor not found: {receptor_pdb}", "error": "file_not_found"}
    if not os.path.isfile(ligand_sdf):
        return {"summary": f"Error: Ligand not found: {ligand_sdf}", "error": "file_not_found"}

    t0 = time.time()
    output_sdf = "/workspace/output/docked.sdf"
    os.makedirs("/workspace/output", exist_ok=True)

    cmd = [
        "gnina",
        "--receptor", receptor_pdb,
        "--ligand", ligand_sdf,
        "--out", output_sdf,
        "--exhaustiveness", str(exhaustiveness),
    ]

    if autobox_ligand and os.path.isfile(autobox_ligand):
        cmd.extend(["--autobox_ligand", autobox_ligand])
    else:
        cmd.extend([
            "--center_x", str(center_x), "--center_y", str(center_y), "--center_z", str(center_z),
            "--size_x", str(size_x), "--size_y", str(size_y), "--size_z", str(size_z),
        ])

    if cnn_scoring and cnn_scoring != "default":
        cmd.extend(["--cnn", cnn_scoring])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=540)
        t_total = time.time() - t0

        if proc.returncode != 0:
            return {"summary": f"GNINA failed: {proc.stderr[:500]}", "error": proc.stderr[:500]}

        # Parse output for scores
        poses = []
        for line in proc.stdout.split("\n"):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    mode = int(parts[0])
                    affinity = float(parts[1])
                    cnn_score = float(parts[2]) if len(parts) > 2 else None
                    cnn_affinity = float(parts[3]) if len(parts) > 3 else None
                    poses.append({
                        "mode": mode,
                        "vina_affinity": affinity,
                        "cnn_score": cnn_score,
                        "cnn_affinity": cnn_affinity,
                    })
                except (ValueError, IndexError):
                    pass

        has_output = os.path.isfile(output_sdf)
        best_vina = min((p["vina_affinity"] for p in poses), default=None)
        best_cnn = max((p["cnn_score"] for p in poses if p["cnn_score"]), default=None)

        if session_id and has_output:
            workspace = f"/vol/workspace/{session_id}"
            os.makedirs(workspace, exist_ok=True)
            import shutil
            shutil.copy2(output_sdf, workspace)

        return {
            "summary": (
                f"GNINA docking: {len(poses)} pose(s) in {t_total:.1f}s. "
                f"Best Vina: {best_vina:.2f} kcal/mol, CNN score: {best_cnn:.3f}" if best_vina and best_cnn else
                f"GNINA docking: {len(poses)} pose(s) in {t_total:.1f}s."
            ),
            "n_poses": len(poses),
            "poses": poses[:20],
            "best_vina_affinity": best_vina,
            "best_cnn_score": best_cnn,
            "output_sdf": output_sdf if has_output else None,
            "metrics": {"time_total_s": round(t_total, 2)},
        }
    except subprocess.TimeoutExpired:
        return {"summary": "GNINA timed out.", "error": "timeout"}
    except Exception as e:
        return {"summary": f"GNINA error: {e}", "error": str(e)}
