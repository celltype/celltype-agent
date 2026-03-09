"""RFDiffusion — thin wrapper around their run_inference.py CLI.
Hardware: A10G/H100 GPU, ~8-16GB VRAM.
"""
import json, os, subprocess, sys, tempfile, threading, time


RFDIFFUSION_DIR = "/app/RFdiffusion"


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


def run(target_pdb="", hotspot_residues="", num_designs=3, session_id="", **kwargs):
    if not target_pdb:
        return {"summary": "Error: No target PDB provided.", "error": "no_target"}

    t0 = time.time()
    vram_before = _get_gpu_vram_mb()

    # Start VRAM monitoring
    stop_event = threading.Event()
    vram_results = {"peak": vram_before}
    monitor = threading.Thread(target=_monitor_vram, args=(stop_event, vram_results), daemon=True)
    monitor.start()

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "target.pdb")
        out_dir = os.path.join(tmpdir, "output")
        os.makedirs(out_dir)
        with open(pdb_path, "w") as f:
            f.write(target_pdb)

        n_res = sum(1 for l in target_pdb.split("\n") if l.startswith("ATOM") and " CA " in l)
        if n_res < 1:
            n_res = 5

        # Download weights to cache on first run
        model_dir = os.environ.get("RFDIFFUSION_MODEL_DIR", "/root/.cache/rfdiffusion")
        ckpt = os.path.join(model_dir, "Base_ckpt.pt")
        if not os.path.isfile(ckpt):
            os.makedirs(model_dir, exist_ok=True)
            subprocess.run([
                "wget", "-q",
                "http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt",
                "-O", ckpt,
            ], check=True, timeout=300)

        cmd = [
            sys.executable, os.path.join(RFDIFFUSION_DIR, "scripts", "run_inference.py"),
            f"inference.output_prefix={out_dir}/design",
            f"inference.input_pdb={pdb_path}",
            f"inference.num_designs={num_designs}",
            f"contigmap.contigs=[{n_res}-{n_res}]",
            f"inference.model_directory_path={model_dir}",
            "diffuser.T=25",
        ]

        if hotspot_residues:
            cmd.append(f"ppi.hotspot_res=[{hotspot_residues}]")

        t_inference = time.time()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
                cwd=RFDIFFUSION_DIR, env={**os.environ, "DGLBACKEND": "pytorch"},
            )
        except subprocess.TimeoutExpired:
            stop_event.set()
            monitor.join(timeout=2)
            return {"summary": "Error: RFDiffusion timed out.", "error": "timeout"}
        t_inference = time.time() - t_inference

        stop_event.set()
        monitor.join(timeout=2)
        vram_after = _get_gpu_vram_mb()
        vram_peak = vram_results["peak"]

        if result.returncode != 0:
            return {
                "summary": f"Error: RFDiffusion failed: {result.stderr[-500:]}",
                "error": result.stderr[-500:],
                "metrics": {
                    "vram_before_mb": vram_before,
                    "vram_after_mb": vram_after,
                    "vram_peak_mb": vram_peak,
                    "time_inference_s": round(t_inference, 2),
                    "time_total_s": round(time.time() - t0, 2),
                },
            }

        designs = []
        scores = []
        for f_name in sorted(os.listdir(out_dir)):
            if f_name.endswith(".pdb"):
                with open(os.path.join(out_dir, f_name)) as pf:
                    content = pf.read()
                    designs.append(content[:3000])
                    n_atoms = sum(1 for l in content.split("\n") if l.startswith("ATOM"))
                    scores.append(n_atoms)

        if not designs:
            return {
                "summary": f"RFDiffusion ran but no designs. stdout: {result.stdout[-300:]}",
                "error": "no_output",
                "metrics": {
                    "vram_before_mb": vram_before,
                    "vram_after_mb": vram_after,
                    "vram_peak_mb": vram_peak,
                    "time_inference_s": round(t_inference, 2),
                    "time_total_s": round(time.time() - t0, 2),
                },
            }

        t_total = time.time() - t0

        return {
            "summary": f"RFDiffusion: generated {len(designs)} backbone designs with {scores[0]} atoms each.",
            "designs": designs,
            "scores": scores,
            "num_designs": len(designs),
            "metrics": {
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after,
                "vram_peak_mb": vram_peak,
                "time_inference_s": round(t_inference, 2),
                "time_total_s": round(t_total, 2),
            },
        }
