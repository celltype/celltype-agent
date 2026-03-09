"""Protenix structure prediction — outperforms AlphaFold 3.

ByteDance open-source model (Apache 2.0).
Hardware: A100 GPU, 40-80GB VRAM.
"""
import time
import os


def run(input_json: str = "", fasta_path: str = "", num_samples: int = 5,
        session_id: str = "", **kwargs) -> dict:
    if not input_json and not fasta_path:
        return {"summary": "Error: Provide input_json or fasta_path.", "error": "no_input"}

    input_file = input_json or fasta_path
    if not os.path.isfile(input_file):
        return {"summary": f"Error: Input not found: {input_file}", "error": "file_not_found"}

    num_samples = max(1, min(int(num_samples or 5), 20))
    t0 = time.time()

    try:
        from protenix.predict import predict as protenix_predict
    except ImportError:
        return {
            "summary": "Protenix not available. Install from: https://github.com/bytedance/Protenix",
            "error": "protenix not installed",
        }

    try:
        output_dir = "/workspace/output"
        os.makedirs(output_dir, exist_ok=True)

        results = protenix_predict(
            input_path=input_file,
            output_dir=output_dir,
            num_samples=num_samples,
        )

        t_total = time.time() - t0

        output_pdbs = [
            os.path.join(output_dir, f) for f in sorted(os.listdir(output_dir)) if f.endswith(".pdb")
        ]

        if session_id and output_pdbs:
            workspace = f"/vol/workspace/{session_id}"
            os.makedirs(workspace, exist_ok=True)
            import shutil
            for pdb in output_pdbs:
                shutil.copy2(pdb, workspace)

        return {
            "summary": f"Protenix: {len(output_pdbs)} structure(s) generated in {t_total:.1f}s.",
            "n_models": len(output_pdbs),
            "output_pdbs": output_pdbs,
            "metrics": {"time_total_s": round(t_total, 2)},
        }
    except Exception as e:
        return {"summary": f"Protenix prediction failed: {e}", "error": str(e)}
