"""OpenFold2 structure prediction.

Uses OpenFold (aqlaboratory/openfold) — a PyTorch reimplementation of AlphaFold2.
Runs inference via run_pretrained_openfold.py in single-sequence mode.
Hardware: A100/H100 GPU.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time


def _get_gpu_vram_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return 0


def _monitor_vram(stop_event, results):
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


OPENFOLD_DIR = "/opt/openfold"


def run(sequence="", session_id="", **kwargs):
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
        # Write FASTA
        fasta_dir = os.path.join(tmpdir, "fasta")
        os.makedirs(fasta_dir)
        with open(os.path.join(fasta_dir, "query.fasta"), "w") as f:
            f.write(f">query\n{clean_seq}\n")

        out_dir = os.path.join(tmpdir, "output")
        os.makedirs(out_dir)

        # Create template dir with a dummy CIF (required by template featurizer init)
        template_dir = os.path.join(tmpdir, "templates")
        os.makedirs(template_dir)
        with open(os.path.join(template_dir, "dummy.cif"), "w") as f:
            f.write("data_dummy\n")

        # Create alignments dir with empty per-sequence subdirectory
        align_dir = os.path.join(tmpdir, "alignments")
        os.makedirs(os.path.join(align_dir, "query"))

        # Download AF2 weights on first run
        params_dir = os.environ.get("OPENFOLD_PARAMS_DIR", "/root/.cache/openfold/params")
        params_file = os.path.join(params_dir, "params_model_1.npz")
        if not os.path.isfile(params_file):
            os.makedirs(params_dir, exist_ok=True)
            dl_script = os.path.join(OPENFOLD_DIR, "scripts", "download_alphafold_params.sh")
            # download_alphafold_params.sh creates a params/ subdir
            parent = os.path.dirname(params_dir)
            subprocess.run(["bash", dl_script, parent], check=True, timeout=600)

        # Check if TensorRT + cuda-python are both available
        trt_available = False
        try:
            import tensorrt  # noqa: F401
            import cuda.cudart  # noqa: F401
            trt_available = True
        except (ImportError, ModuleNotFoundError):
            pass

        trt_engine_dir = os.path.join(OPENFOLD_DIR, "trt_engines")

        cmd = [
            sys.executable, os.path.join(OPENFOLD_DIR, "run_pretrained_openfold.py"),
            fasta_dir,
            template_dir,
            "--use_single_seq_mode",
            "--use_precomputed_alignments", align_dir,
            "--output_dir", out_dir,
            "--model_device", "cuda:0",
            "--config_preset", "model_1",
            "--skip_relaxation",
            "--jax_param_path", params_file,
        ]

        # Use TensorRT if available (builds engine on first run, reuses after)
        if trt_available:
            if os.path.isdir(trt_engine_dir) and os.listdir(trt_engine_dir):
                cmd.extend(["--trt_mode", "run", "--trt_engine_dir", trt_engine_dir])
            else:
                os.makedirs(trt_engine_dir, exist_ok=True)
                cmd.extend(["--trt_mode", "build", "--trt_engine_dir", trt_engine_dir])

        t_inference = time.time()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
                cwd=OPENFOLD_DIR,
            )
        except subprocess.TimeoutExpired:
            stop_event.set()
            monitor.join(timeout=2)
            return {"summary": "Error: OpenFold2 timed out.", "error": "timeout"}
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
                "summary": f"Error: OpenFold2 failed: {result.stderr[-500:]}",
                "error": result.stderr[-500:],
                "metrics": metrics,
            }

        # Find output PDB (may be in predictions/ subdirectory)
        pdb_content = ""
        confidence = 0.0

        for root, dirs, files in os.walk(out_dir):
            for f_name in sorted(files):
                if f_name.endswith(".pdb"):
                    fpath = os.path.join(root, f_name)
                    with open(fpath) as pf:
                        pdb_content = pf.read()
                    # Extract pLDDT from B-factor column
                    bfactors = []
                    for line in pdb_content.split("\n"):
                        if line.startswith("ATOM") and " CA " in line:
                            try:
                                bfactors.append(float(line[60:66].strip()))
                            except (ValueError, IndexError):
                                pass
                    if bfactors:
                        confidence = sum(bfactors) / len(bfactors)
                    break
            if pdb_content:
                break

        if not pdb_content:
            all_files = os.listdir(out_dir) if os.path.isdir(out_dir) else []
            return {
                "summary": f"OpenFold2 ran but no structure. Files: {all_files}. stderr: {result.stderr[-300:]}",
                "error": "no_output",
                "metrics": metrics,
            }

        if session_id:
            workspace_dir = f"/vol/workspace/{session_id}"
            os.makedirs(workspace_dir, exist_ok=True)
            with open(f"{workspace_dir}/predicted_structure.pdb", "w") as pf:
                pf.write(pdb_content)

        return {
            "summary": f"OpenFold2 prediction for {seq_len}-residue protein. pLDDT: {confidence:.1f}/100.",
            "pdb_content": pdb_content[:5000],
            "confidence": confidence,
            "num_residues": seq_len,
            "metrics": metrics,
        }
