"""ProteinMPNN sequence design. Hardware: T4 GPU, 8GB VRAM."""
import json, os, subprocess, sys, tempfile, time, threading

PROTEINMPNN_DIR = "/opt/ProteinMPNN"

def _vram_peak():
    try:
        r = subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=3)
        return int(r.stdout.strip().split('\n')[0])
    except: return 0

def run(backbone_pdb="", num_sequences=10, session_id="", **kwargs):
    if not backbone_pdb:
        return {"summary": "Error: No backbone PDB provided.", "error": "no_backbone"}
    if not os.path.isdir(PROTEINMPNN_DIR):
        return {"summary": "Error: ProteinMPNN not installed.", "error": "not_installed"}

    t0 = time.time()
    vram_before = _vram_peak()

    # Ensure weights are in cache (copy from repo clone on first run)
    cache_weights = os.environ.get("PROTEINMPNN_WEIGHTS", "/root/.cache/proteinmpnn")
    vanilla_cache = os.path.join(cache_weights, "vanilla_model_weights")
    if not os.path.isdir(vanilla_cache) or not os.listdir(vanilla_cache):
        import shutil
        src = os.path.join(PROTEINMPNN_DIR, "vanilla_model_weights")
        if os.path.isdir(src):
            os.makedirs(cache_weights, exist_ok=True)
            shutil.copytree(src, vanilla_cache, dirs_exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "input.pdb")
        with open(pdb_path, "w") as f: f.write(backbone_pdb)
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir)

        cmd = [sys.executable, os.path.join(PROTEINMPNN_DIR, "protein_mpnn_run.py"),
               "--pdb_path", pdb_path, "--out_folder", output_dir,
               "--num_seq_per_target", str(num_sequences), "--sampling_temp", "0.1",
               "--seed", "42", "--batch_size", str(min(num_sequences, 8)),
               "--path_to_model_weights", vanilla_cache + "/"]

        # Monitor VRAM during execution
        peak_vram = [vram_before]
        stop = threading.Event()
        def _mon():
            while not stop.is_set():
                v = _vram_peak()
                if v > peak_vram[0]: peak_vram[0] = v
                stop.wait(0.5)
        mon = threading.Thread(target=_mon, daemon=True); mon.start()

        t_exec = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=PROTEINMPNN_DIR)
        t_exec = time.time() - t_exec

        stop.set(); mon.join(timeout=2)
        vram_peak = peak_vram[0]

        if result.returncode != 0:
            return {"summary": f"Error: ProteinMPNN failed: {result.stderr[:500]}", "error": result.stderr[:500]}

        sequences, scores = [], []
        seqs_dir = os.path.join(output_dir, "seqs")
        if os.path.isdir(seqs_dir):
            for fasta_file in sorted(os.listdir(seqs_dir)):
                if fasta_file.endswith(".fa"):
                    with open(os.path.join(seqs_dir, fasta_file)) as f:
                        lines = f.readlines()
                    for i in range(0, len(lines), 2):
                        header = lines[i].strip()
                        if i+1 < len(lines):
                            seq = lines[i+1].strip()
                            score = 0.0
                            if "score=" in header:
                                try: score = float(header.split("score=")[1].split(",")[0])
                                except: pass
                            if seq and "sample=0" not in header:
                                sequences.append(seq); scores.append(score)

        if not sequences:
            return {"summary": f"ProteinMPNN ran but no sequences.", "error": "no_output"}

        return {
            "summary": f"ProteinMPNN: designed {len(sequences)} sequences. Best score: {min(scores):.3f} (lower is better).",
            "sequences": sequences[:num_sequences],
            "scores": scores[:num_sequences],
            "num_sequences": len(sequences[:num_sequences]),
            "metrics": {
                "vram_before_mb": vram_before,
                "vram_peak_mb": vram_peak,
                "vram_delta_mb": vram_peak - vram_before,
                "time_execution_s": round(t_exec, 2),
                "time_total_s": round(time.time()-t0, 2),
            },
        }
