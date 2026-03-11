"""Boltz-2 structure prediction. pip install boltz. Hardware: A100 GPU, 40GB VRAM."""
import json, os, subprocess, sys, tempfile, time, threading

def _monitor_vram(results, stop_event):
    """Background thread to monitor GPU VRAM via nvidia-smi."""
    peak = 0
    while not stop_event.is_set():
        try:
            r = subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=3)
            val = int(r.stdout.strip().split('\n')[0])
            if val > peak: peak = val
        except: pass
        stop_event.wait(0.5)
    results["peak"] = peak

def run(sequence="", ligand_smiles="", session_id="", **kwargs):
    if not sequence:
        return {"summary": "Error: No sequence provided.", "error": "no_sequence"}
    clean_seq = sequence.strip().upper().replace(" ","").replace("\n","")
    if clean_seq.startswith(">"): clean_seq = "".join(l for l in clean_seq.split("\n") if not l.startswith(">"))
    seq_len = len(clean_seq)
    t0 = time.time()

    # Get baseline VRAM
    try:
        r = subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=3)
        vram_before = int(r.stdout.strip().split('\n')[0])
    except: vram_before = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = os.path.join(tmpdir, "input.yaml")
        out_dir = os.path.join(tmpdir, "output")
        os.makedirs(out_dir)
        yaml_content = f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {clean_seq}\n"
        if ligand_smiles: yaml_content += f'  - ligand:\n      id: B\n      smiles: "{ligand_smiles}"\n'
        with open(yaml_path, "w") as f: f.write(yaml_content)

        import shutil
        boltz_cmd = shutil.which("boltz") or "boltz"
        cache_dir = os.environ.get("BOLTZ_CACHE", "/root/.boltz")
        cmd = [boltz_cmd,"predict",yaml_path,"--out_dir",out_dir,"--recycling_steps","3","--sampling_steps","50",
               "--diffusion_samples","1","--output_format","pdb","--devices","1","--accelerator","gpu",
               "--num_workers","0","--override","--use_msa_server","--no_kernels",
               "--cache", cache_dir]

        # Start VRAM monitoring
        vram_results = {"peak": 0}
        stop = threading.Event()
        mon = threading.Thread(target=_monitor_vram, args=(vram_results, stop), daemon=True)
        mon.start()

        t_exec_start = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            stop.set()
            return {"summary": "Error: Boltz-2 timed out.", "error": "timeout"}

        t_exec = time.time() - t_exec_start
        stop.set(); mon.join(timeout=2)
        vram_peak = vram_results["peak"]

        if result.returncode != 0:
            cmd[cmd.index("50")] = "20"
            try: result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            except: pass
            if result.returncode != 0:
                return {"summary": f"Error: Boltz-2 failed: {result.stderr[-500:]}", "error": result.stderr[-500:]}

        pdb_content = ""
        confidence = 0.0
        for root, dirs, files in os.walk(out_dir):
            for f in files:
                fpath = os.path.join(root, f)
                if f.endswith(".pdb"):
                    with open(fpath) as pf: pdb_content = pf.read()
                elif f.endswith(".json") and "confidence" in f.lower():
                    try: scores = json.load(open(fpath)); confidence = float(scores.get("confidence", scores.get("ptm", 0)))
                    except: pass
        if not pdb_content:
            for root, dirs, files in os.walk(out_dir):
                for f in files:
                    if f.endswith(".cif"):
                        with open(os.path.join(root, f)) as cf: pdb_content = cf.read()
                        break
        if not pdb_content:
            all_files = [os.path.relpath(os.path.join(r,f),out_dir) for r,_,fs in os.walk(out_dir) for f in fs]
            return {"summary": f"Boltz-2 ran but no structure. Files: {all_files}", "error": "no_output"}

        complex_str = f" with ligand {ligand_smiles[:20]}" if ligand_smiles else ""
        return {
            "summary": f"Boltz-2 structure prediction for {seq_len}-residue protein{complex_str}. Confidence: {confidence:.2f}.",
            "pdb_content": pdb_content,
            "confidence": confidence,
            "num_residues": seq_len,
            "metrics": {
                "vram_before_mb": vram_before,
                "vram_peak_mb": vram_peak,
                "vram_delta_mb": vram_peak - vram_before,
                "time_execution_s": round(t_exec, 2),
                "time_total_s": round(time.time()-t0, 2),
            },
        }
