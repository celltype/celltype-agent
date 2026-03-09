"""OpenFold3 structure prediction implementation.

Uses openfold3 for structure prediction.
Built from https://github.com/aqlaboratory/openfold-3 official Docker image.
Hardware: A100 GPU, 80GB VRAM.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time


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
    vram = _get_gpu_vram_mb()
    if vram > peak:
        results["peak"] = vram


def run(sequence: str = "", run_msa: bool = False, session_id: str = "", **kwargs) -> dict:
    """Predict protein structure using OpenFold3.

    Args:
        sequence: Amino acid sequence.
        run_msa: Whether to run MSA search (default: False for speed).
        session_id: Optional session ID for shared workspace.

    Returns:
        dict with 'summary', 'pdb_content', 'confidence', 'num_residues', 'metrics'.
    """
    if not sequence:
        return {"summary": "Error: No sequence provided.", "error": "no_sequence"}

    t0 = time.time()
    clean_seq = sequence.strip().upper().replace(" ", "").replace("\n", "")
    if clean_seq.startswith(">"):
        lines = clean_seq.split("\n")
        clean_seq = "".join(l for l in lines if not l.startswith(">"))

    seq_len = len(clean_seq)

    vram_before = _get_gpu_vram_mb()
    stop_event = threading.Event()
    vram_results = {"peak": vram_before}
    monitor = threading.Thread(target=_monitor_vram, args=(stop_event, vram_results), daemon=True)
    monitor.start()

    with tempfile.TemporaryDirectory() as tmpdir:
        query_json = os.path.join(tmpdir, "query.json")
        out_dir = os.path.join(tmpdir, "output")
        os.makedirs(out_dir, exist_ok=True)

        query = {
            "queries": {
                "query_1": {
                    "chains": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": "A",
                            "sequence": clean_seq,
                        }
                    ]
                }
            }
        }

        with open(query_json, "w") as f:
            json.dump(query, f)

        # Download weights on first run (try aws s3, fall back to openfold3's own script)
        ckpt_dir = os.environ.get("OPENFOLD3_CACHE", "/root/.openfold3")
        ckpt_file = os.path.join(ckpt_dir, "of3_ft3_v1.pt")
        if not os.path.isfile(ckpt_file):
            os.makedirs(ckpt_dir, exist_ok=True)
            try:
                subprocess.run([
                    "aws", "s3", "cp",
                    "s3://openfold/openfold3_params/of3_ft3_v1.pt",
                    ckpt_dir + "/", "--no-sign-request",
                ], check=True, timeout=600)
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback: use the bundled download script if available
                dl_script = "/opt/openfold3/openfold3/scripts/download_openfold3_params.sh"
                if os.path.isfile(dl_script):
                    subprocess.run(["bash", dl_script, f"--download_dir={ckpt_dir}"],
                                   check=True, timeout=600)
                else:
                    return {"summary": "Error: Cannot download OpenFold3 weights. Install awscli.",
                            "error": "weight_download_failed"}

        # Check if DeepSpeed evoformer attention can be JIT compiled
        # If not (e.g. Modal), fall back to low_mem preset with LMA attention
        use_low_mem = os.environ.get("OPENFOLD3_LOW_MEM", "")
        runner_yaml_path = os.path.join(tmpdir, "runner.yaml")
        if use_low_mem:
            with open(runner_yaml_path, "w") as yf:
                yf.write("model_update:\n  presets:\n    - predict\n    - low_mem\n    - pae_enabled\n")

        cmd = [
            "run_openfold", "predict",
            f"--query_json={query_json}",
            f"--output_dir={out_dir}",
            f"--inference_ckpt_path={ckpt_file}",
            "--num_diffusion_samples=1",
            "--num_model_seeds=1",
        ]
        if use_low_mem:
            cmd.append(f"--runner_yaml={runner_yaml_path}")

        if run_msa:
            cmd.extend(["--use_msa_server", "True"])
        else:
            cmd.extend(["--use_templates", "False"])

        t_inference = time.time()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )
        except subprocess.TimeoutExpired:
            stop_event.set()
            monitor.join(timeout=2)
            return {"summary": "Error: OpenFold3 timed out after 600s.", "error": "timeout"}
        t_inference = time.time() - t_inference

        stop_event.set()
        monitor.join(timeout=2)
        vram_after = _get_gpu_vram_mb()
        vram_peak = vram_results["peak"]

        metrics = {
            "vram_before_mb": vram_before,
            "vram_after_mb": vram_after,
            "vram_peak_mb": vram_peak,
            "time_inference_s": round(t_inference, 2),
            "time_total_s": round(time.time() - t0, 2),
        }

        if result.returncode != 0:
            return {
                "summary": f"Error: OpenFold3 failed: {result.stderr[-500:]}",
                "error": result.stderr[-500:],
                "metrics": metrics,
            }

        # Find output CIF/PDB
        pdb_content = ""
        confidence = 0.0

        for root, dirs, files in os.walk(out_dir):
            for f in files:
                fpath = os.path.join(root, f)
                if f.endswith(".cif") or f.endswith(".pdb"):
                    with open(fpath) as pf:
                        pdb_content = pf.read()
                elif f.endswith(".json") and ("score" in f.lower() or "metric" in f.lower()):
                    try:
                        scores = json.load(open(fpath))
                        if isinstance(scores, dict):
                            confidence = float(scores.get("ptm", scores.get("plddt", scores.get("confidence", 0))))
                    except Exception:
                        pass

        if not pdb_content:
            all_files = []
            log_content = ""
            for root, dirs, files in os.walk(out_dir):
                for f in files:
                    all_files.append(os.path.relpath(os.path.join(root, f), out_dir))
                    if f.endswith(".log") or f == "summary.txt":
                        try:
                            with open(os.path.join(root, f)) as lf:
                                log_content += f"\n--- {f} ---\n" + lf.read()[-500:]
                        except Exception:
                            pass
            return {
                "summary": f"OpenFold3 ran but no structure found. Files: {all_files}. Logs: {log_content[-800:]}",
                "error": "no_output",
                "metrics": metrics,
            }

        msa_str = " with MSA" if run_msa else " without MSA"

        if session_id:
            workspace_dir = f"/vol/workspace/{session_id}"
            os.makedirs(workspace_dir, exist_ok=True)
            with open(f"{workspace_dir}/predicted_structure.pdb", "w") as pf:
                pf.write(pdb_content)

        return {
            "summary": (
                f"OpenFold3 structure prediction for {seq_len}-residue protein{msa_str}. "
                f"Confidence: {confidence:.2f}."
            ),
            "pdb_content": pdb_content[:5000],
            "confidence": confidence,
            "num_residues": seq_len,
            "metrics": metrics,
        }
