"""DiffDock molecular docking implementation.

Uses DiffDock for GPU-accelerated molecular docking.
Runs inference via the DiffDock CLI inside the container.
Hardware: T4/A10G GPU, 8-16GB VRAM.
"""

import os
import subprocess
import sys
import tempfile
import threading
import time


DIFFDOCK_DIR = "/opt/DiffDock"


def _get_gpu_vram_mb():
    """Get current GPU VRAM usage in MB via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return 0


def _monitor_vram(stop_event, results):
    """Background thread to track peak VRAM usage."""
    peak = 0
    while not stop_event.is_set():
        vram = _get_gpu_vram_mb()
        if vram > peak:
            peak = vram
        results["peak"] = peak
        stop_event.wait(0.5)
    # Final check
    vram = _get_gpu_vram_mb()
    if vram > peak:
        results["peak"] = vram


def run(
    protein_pdb: str = "",
    ligand_smiles: str = "",
    session_id: str = "",
    num_poses: int = 5,
    **kwargs,
) -> dict:
    """Molecular docking using DiffDock.

    Args:
        protein_pdb: PDB content string.
        ligand_smiles: SMILES string of the ligand.
        session_id: For reading protein from shared workspace.
        num_poses: Number of binding poses to generate.

    Returns:
        dict with 'summary', 'poses', 'best_score', 'ligand_smiles', 'metrics'.
    """
    t0 = time.time()

    # Try to read PDB from shared volume if not provided directly
    if not protein_pdb and session_id:
        pdb_path = f"/vol/workspace/{session_id}/predicted_structure.pdb"
        if os.path.exists(pdb_path):
            with open(pdb_path) as f:
                protein_pdb = f.read()

    if not protein_pdb:
        return {
            "summary": "Error: No protein structure provided. Run structure prediction first.",
            "error": "no_protein",
        }
    if not ligand_smiles:
        return {
            "summary": "Error: No ligand SMILES provided.",
            "error": "no_ligand",
        }

    # Check if we're inside the DiffDock container
    if not os.path.isdir(DIFFDOCK_DIR):
        return {
            "summary": "Error: DiffDock not installed at /opt/DiffDock.",
            "error": "not_installed",
        }

    # VRAM baseline before inference
    vram_before = _get_gpu_vram_mb()

    # Start VRAM monitoring
    stop_event = threading.Event()
    vram_results = {"peak": vram_before}
    monitor_thread = threading.Thread(target=_monitor_vram, args=(stop_event, vram_results), daemon=True)
    monitor_thread.start()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write protein PDB
        protein_path = os.path.join(tmpdir, "protein.pdb")
        with open(protein_path, "w") as f:
            f.write(protein_pdb)

        out_dir = os.path.join(tmpdir, "results")
        os.makedirs(out_dir, exist_ok=True)

        # Count CA atoms for validation
        n_ca = sum(1 for l in protein_pdb.split("\n") if l.startswith("ATOM") and " CA " in l)

        # DiffDock downloads score/confidence models from GitHub releases on first run.
        # Symlink workdir → cache so downloads persist across container restarts.
        cache_dir = os.environ.get("DIFFDOCK_CACHE", "/root/.cache/diffdock")
        workdir = os.path.join(DIFFDOCK_DIR, "workdir")
        if os.path.isdir(cache_dir) and not os.path.exists(workdir):
            os.symlink(cache_dir, workdir)
        elif not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            if not os.path.exists(workdir):
                os.symlink(cache_dir, workdir)

        cmd = [
            sys.executable, "-m", "inference",
            "--config", os.path.join(DIFFDOCK_DIR, "default_inference_args.yaml"),
            "--protein_path", protein_path,
            "--ligand_description", ligand_smiles,
            "--out_dir", out_dir,
            "--samples_per_complex", str(num_poses),
            "--no_final_step_noise",
        ]

        t_inference = time.time()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=600, cwd=DIFFDOCK_DIR,
            )
        except subprocess.TimeoutExpired:
            stop_event.set()
            monitor_thread.join(timeout=2)
            return {"summary": "Error: DiffDock timed out after 600s.", "error": "timeout"}
        t_inference = time.time() - t_inference

        # Stop VRAM monitoring
        stop_event.set()
        monitor_thread.join(timeout=2)
        vram_after = _get_gpu_vram_mb()
        vram_peak = vram_results["peak"]

        if result.returncode != 0:
            return {
                "summary": f"Error: DiffDock failed: {result.stderr[-500:]}",
                "error": result.stderr[-500:],
                "metrics": {
                    "vram_before_mb": vram_before,
                    "vram_after_mb": vram_after,
                    "vram_peak_mb": vram_peak,
                    "time_inference_s": round(t_inference, 2),
                    "time_total_s": round(time.time() - t0, 2),
                },
            }

        # Parse output — DiffDock creates SDF files with confidence scores
        poses = []
        for root, dirs, files in os.walk(out_dir):
            for f in sorted(files):
                if f.endswith(".sdf"):
                    sdf_path = os.path.join(root, f)
                    with open(sdf_path) as sf:
                        sdf_content = sf.read()

                    # Extract confidence from filename (rank1_confidence-0.85.sdf)
                    confidence = 0.0
                    if "confidence" in f:
                        try:
                            conf_str = f.split("confidence")[1].replace(".sdf", "")
                            confidence = float(conf_str)
                        except (ValueError, IndexError):
                            pass

                    poses.append({
                        "pose_id": len(poses) + 1,
                        "confidence_score": confidence,
                        "sdf_content": sdf_content[:1000],  # Truncate for JSON
                    })

        if not poses:
            return {
                "summary": f"DiffDock ran but produced no poses. stdout: {result.stdout[-300:]}",
                "stderr": result.stderr[-500:] if result.stderr else "",
                "error": "no_output",
                "metrics": {
                    "vram_before_mb": vram_before,
                    "vram_after_mb": vram_after,
                    "vram_peak_mb": vram_peak,
                    "time_inference_s": round(t_inference, 2),
                    "time_total_s": round(time.time() - t0, 2),
                },
            }

        # Sort by confidence (higher is better)
        poses.sort(key=lambda p: p["confidence_score"], reverse=True)
        best_score = poses[0]["confidence_score"] if poses else 0.0

        t_total = time.time() - t0

        return {
            "summary": (
                f"DiffDock docking: generated {len(poses)} binding poses for ligand "
                f"({ligand_smiles[:30]}). "
                f"Best confidence: {best_score:.2f}."
            ),
            "poses": poses[:num_poses],
            "best_score": best_score,
            "ligand_smiles": ligand_smiles,
            "num_ca_atoms": n_ca,
            "metrics": {
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after,
                "vram_peak_mb": vram_peak,
                "time_inference_s": round(t_inference, 2),
                "time_total_s": round(t_total, 2),
            },
        }
