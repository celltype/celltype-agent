"""Boltz-1 structure prediction implementation.

Uses Boltz-1 for protein structure and protein-ligand complex prediction.
Boltz-1 natively co-folds protein and ligand in one step — no separate docking needed.
Hardware: A10G GPU, 16GB+ VRAM.

Source: https://github.com/jwohlwend/boltz
"""
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time


def _get_vram_mb() -> int:
    """Read current GPU VRAM usage via nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        return int(r.stdout.strip().split("\n")[0])
    except Exception:
        return 0


def _monitor_vram(results: dict, stop_event: threading.Event) -> None:
    """Background thread: poll VRAM every 0.5 s and record peak."""
    peak = 0
    while not stop_event.is_set():
        val = _get_vram_mb()
        if val > peak:
            peak = val
        stop_event.wait(0.5)
    # Final sample after stop
    val = _get_vram_mb()
    results["peak"] = max(peak, val)


def run(sequence: str = "", ligand_smiles: str = "", session_id: str = "", **kwargs) -> dict:
    """Run Boltz-1 structure prediction.

    Args:
        sequence:      Protein amino acid sequence (single-letter code or FASTA).
        ligand_smiles: Optional SMILES for protein-ligand complex prediction.
        session_id:    Optional session ID — output PDB is copied to /vol/workspace/<session_id>.

    Returns:
        dict with keys: summary, pdb_content, confidence, num_residues, metrics.
    """
    # Clean sequence — strip FASTA header if present
    raw = (sequence or "").strip()
    if raw.startswith(">"):
        # Multi-line FASTA: drop header lines, join sequence lines
        clean_seq = "".join(
            line.strip() for line in raw.splitlines() if not line.startswith(">")
        ).upper().replace(" ", "")
    else:
        clean_seq = raw.upper().replace(" ", "").replace("\n", "")

    if not clean_seq:
        return {"summary": "Error: No sequence provided.", "error": "no_sequence"}

    seq_len = len(clean_seq)

    t0 = time.time()
    vram_before = _get_vram_mb()

    with tempfile.TemporaryDirectory() as tmpdir:
        # --- Write Boltz-1 YAML input ---
        yaml_path = os.path.join(tmpdir, "input.yaml")
        out_dir = os.path.join(tmpdir, "output")
        os.makedirs(out_dir, exist_ok=True)

        yaml_lines = [
            "version: 1",
            "sequences:",
            "  - protein:",
            "      id: A",
            f"      sequence: {clean_seq}",
        ]
        if ligand_smiles:
            yaml_lines += [
                "  - ligand:",
                "      id: B",
                f'      smiles: "{ligand_smiles}"',
            ]
        with open(yaml_path, "w") as fh:
            fh.write("\n".join(yaml_lines) + "\n")

        # --- Build boltz predict command ---
        boltz_bin = shutil.which("boltz") or "boltz"
        cache_dir = os.environ.get("BOLTZ_CACHE", "/root/.boltz")

        cmd = [
            boltz_bin, "predict", yaml_path,
            "--out_dir", out_dir,
            "--recycling_steps", "3",
            "--sampling_steps", "200",
            "--diffusion_samples", "1",
            "--output_format", "pdb",
            "--devices", "1",
            "--accelerator", "gpu",
            "--num_workers", "0",
            "--override",
            "--use_msa_server",
            "--cache", cache_dir,
        ]

        # --- Start VRAM monitor ---
        vram_results: dict = {"peak": 0}
        stop_event = threading.Event()
        monitor = threading.Thread(
            target=_monitor_vram, args=(vram_results, stop_event), daemon=True
        )
        monitor.start()

        t_exec_start = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            stop_event.set()
            monitor.join(timeout=2)
            return {"summary": "Error: Boltz-1 timed out after 600s.", "error": "timeout"}

        t_exec = time.time() - t_exec_start
        stop_event.set()
        monitor.join(timeout=2)
        vram_peak = vram_results["peak"]

        # --- Retry with fewer sampling steps on failure ---
        if result.returncode != 0:
            cmd_retry = [
                c if c != "200" else "50" for c in cmd
            ]
            try:
                result = subprocess.run(cmd_retry, capture_output=True, text=True, timeout=600)
            except Exception:
                pass
            if result.returncode != 0:
                return {
                    "summary": f"Error: Boltz-1 failed: {result.stderr[-500:]}",
                    "error": result.stderr[-500:],
                }

        # --- Collect output ---
        pdb_content = ""
        confidence = 0.0

        for root, _, files in os.walk(out_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                if fname.endswith(".pdb") and not pdb_content:
                    with open(fpath) as pf:
                        pdb_content = pf.read()
                elif fname.endswith(".json") and "confidence" in fname.lower():
                    try:
                        with open(fpath) as jf:
                            scores = json.load(jf)
                        confidence = float(
                            scores.get("confidence", scores.get("ptm", scores.get("plddt", 0)))
                        )
                    except Exception:
                        pass

        # Fallback: accept CIF if no PDB found
        if not pdb_content:
            for root, _, files in os.walk(out_dir):
                for fname in files:
                    if fname.endswith(".cif"):
                        with open(os.path.join(root, fname)) as cf:
                            pdb_content = cf.read()
                        break

        if not pdb_content:
            all_files = [
                os.path.relpath(os.path.join(r, f), out_dir)
                for r, _, fs in os.walk(out_dir)
                for f in fs
            ]
            return {
                "summary": f"Boltz-1 ran but produced no structure. Files: {all_files}",
                "error": "no_output",
            }

        # --- Optional: persist to session workspace ---
        if session_id:
            workspace_dir = f"/vol/workspace/{session_id}"
            os.makedirs(workspace_dir, exist_ok=True)
            with open(f"{workspace_dir}/predicted_structure.pdb", "w") as fh:
                fh.write(pdb_content)

        complex_label = f" + ligand ({ligand_smiles[:30]})" if ligand_smiles else ""
        return {
            "summary": (
                f"Boltz-1 predicted structure for {seq_len}-residue protein"
                f"{complex_label}. Confidence: {confidence:.2f}."
            ),
            "pdb_content": pdb_content[:5000],
            "confidence": confidence,
            "num_residues": seq_len,
            "ligand_included": bool(ligand_smiles),
            "metrics": {
                "vram_before_mb": vram_before,
                "vram_peak_mb": vram_peak,
                "vram_delta_mb": vram_peak - vram_before,
                "time_execution_s": round(t_exec, 2),
                "time_total_s": round(time.time() - t0, 2),
            },
        }
