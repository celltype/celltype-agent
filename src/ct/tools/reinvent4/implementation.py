"""REINVENT 4 — de novo molecular design with reinforcement learning.

AstraZeneca's industry-standard tool for goal-directed molecular generation.
Hardware: A10G GPU, 16GB VRAM.
"""
import time
import os
import json


def run(config_path: str = "", mode: str = "denovo", scoring_smiles: str = "",
        scaffold_smiles: str = "", num_molecules: int = 100, num_epochs: int = 50,
        session_id: str = "", **kwargs) -> dict:

    t0 = time.time()
    output_dir = "/workspace/output"
    os.makedirs(output_dir, exist_ok=True)

    try:
        import reinvent
    except ImportError:
        return {
            "summary": "REINVENT 4 not available. Install from: https://github.com/MolecularAI/REINVENT4",
            "error": "reinvent not installed",
        }

    mode = (mode or "denovo").lower()
    valid_modes = ("denovo", "scaffold", "linker", "rgroup")
    if mode not in valid_modes:
        return {"summary": f"Invalid mode '{mode}'. Use: {valid_modes}", "error": "invalid_mode"}

    if mode in ("scaffold", "linker") and not scaffold_smiles:
        return {"summary": f"scaffold_smiles required for {mode} mode.", "error": "no_scaffold"}

    try:
        # Use config file if provided, otherwise build one
        if config_path and os.path.isfile(config_path):
            import toml
            config = toml.load(config_path)
        else:
            config = {
                "run_type": mode,
                "output_dir": output_dir,
                "parameters": {
                    "num_molecules": num_molecules,
                    "num_epochs": num_epochs,
                },
            }
            if scoring_smiles:
                config["scoring"] = {"reference_smiles": scoring_smiles}
            if scaffold_smiles:
                config["scaffold"] = {"smiles": scaffold_smiles}

            config_path = os.path.join(output_dir, "config.toml")
            import toml
            with open(config_path, "w") as f:
                toml.dump(config, f)

        # Run REINVENT
        from reinvent.runmodes import create_adapter
        adapter = create_adapter(config_path)
        adapter.run()

        t_total = time.time() - t0

        # Collect generated molecules
        generated = []
        results_file = os.path.join(output_dir, "results.csv")
        if os.path.isfile(results_file):
            import csv
            with open(results_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    generated.append({
                        "smiles": row.get("SMILES", row.get("smiles", "")),
                        "score": float(row.get("score", row.get("total_score", 0))),
                    })

        generated.sort(key=lambda x: x["score"], reverse=True)

        if session_id:
            workspace = f"/vol/workspace/{session_id}"
            os.makedirs(workspace, exist_ok=True)
            import shutil
            if os.path.isfile(results_file):
                shutil.copy2(results_file, workspace)

        return {
            "summary": (
                f"REINVENT 4 ({mode}): {len(generated)} molecule(s) generated in {t_total:.1f}s. "
                f"Best score: {generated[0]['score']:.3f}" if generated else
                f"REINVENT 4 ({mode}): completed in {t_total:.1f}s."
            ),
            "mode": mode,
            "n_generated": len(generated),
            "molecules": generated[:num_molecules],
            "metrics": {"time_total_s": round(t_total, 2)},
        }
    except Exception as e:
        return {"summary": f"REINVENT 4 error: {e}", "error": str(e)}
