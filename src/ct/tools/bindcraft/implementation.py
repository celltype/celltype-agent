"""BindCraft — de novo protein binder design pipeline.
Hardware: A100/H100 GPU, 32-80GB VRAM.
Uses AlphaFold2 backpropagation + ProteinMPNN + PyRosetta scoring.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time


BINDCRAFT_DIR = "/app/BindCraft"
PROTEINMPNN_DIR = "/app/ProteinMPNN"
ALLOWED_PATH_SUFFIXES = {".pdb", ".cif", ".mmcif", ".ent"}
ALLOWED_DESIGN_MODES = {"default", "betasheet", "peptide"}
ALLOWED_ARGS = {
    "target_pdb",
    "pdb_text",
    "chains",
    "hotspot_residues",
    "binder_length_min",
    "binder_length_max",
    "num_designs",
    "design_mode",
    "binder_name",
    "session_id",
}


def _get_gpu_vram_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
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


def _require_inline_pdb_text(pdb_text: str) -> str:
    pdb_text = str(pdb_text or "").strip()
    if not pdb_text:
        raise ValueError("BindCraft requires non-empty inline PDB text in `target_pdb`.")
    lines = [line for line in pdb_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("BindCraft requires non-empty inline PDB text in `target_pdb`.")
    if not any(line.startswith(("ATOM", "HETATM")) for line in lines):
        raise ValueError("BindCraft requires PDB text with ATOM/HETATM records.")
    return pdb_text


def _load_target_pdb_file(target_pdb: str) -> str:
    path = str(target_pdb or "").strip()
    if not path:
        raise ValueError("BindCraft requires either `target_pdb` (inline PDB) or a file path.")
    if not os.path.isfile(path):
        raise ValueError("BindCraft `target_pdb` must point to an existing local structure file.")
    if os.path.splitext(path)[1].lower() not in ALLOWED_PATH_SUFFIXES:
        raise ValueError("BindCraft `target_pdb` must be a .pdb, .cif, .mmcif, or .ent file.")
    with open(path, encoding="utf-8") as fh:
        return _require_inline_pdb_text(fh.read())


def _parse_chain_ids(pdb_text: str) -> set[str]:
    chains = set()
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) > 21:
            chain_id = line[21].strip()
            if chain_id:
                chains.add(chain_id)
    return chains


def _count_residues(pdb_text: str) -> int:
    seen = set()
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and " CA " in line and len(line) > 26:
            chain = line[21]
            resnum = line[22:26].strip()
            seen.add((chain, resnum))
    return len(seen)


def normalize_args(args: dict) -> dict:
    unexpected = sorted(set(args) - ALLOWED_ARGS)
    if unexpected:
        raise ValueError(f"BindCraft received unsupported arguments: {', '.join(unexpected)}.")

    normalized = dict(args)

    target_pdb = str(normalized.get("target_pdb", "")).strip()
    pdb_text = str(normalized.get("pdb_text", "")).strip()
    if pdb_text:
        inline_content = _require_inline_pdb_text(pdb_text)
    elif target_pdb:
        if "\n" in target_pdb or target_pdb.startswith(("ATOM", "HETATM", "HEADER", "MODEL")):
            inline_content = _require_inline_pdb_text(target_pdb)
        else:
            inline_content = _load_target_pdb_file(target_pdb)
    else:
        raise ValueError("BindCraft requires `target_pdb` (PDB content or file path).")

    normalized["target_pdb"] = inline_content

    available_chains = _parse_chain_ids(inline_content)
    chains = str(normalized.get("chains", "")).strip()
    if not chains:
        chains = sorted(available_chains)[0] if available_chains else "A"
    for ch in chains.replace(",", ""):
        if ch.strip() and ch.strip() not in available_chains:
            raise ValueError(
                f"Chain '{ch.strip()}' not found in target PDB. Available: {sorted(available_chains)}"
            )
    normalized["chains"] = chains

    normalized["hotspot_residues"] = str(normalized.get("hotspot_residues", "")).strip()

    try:
        binder_min = int(normalized.get("binder_length_min", 65))
    except (TypeError, ValueError):
        binder_min = 65
    try:
        binder_max = int(normalized.get("binder_length_max", 150))
    except (TypeError, ValueError):
        binder_max = 150
    binder_min = max(4, min(250, binder_min))
    binder_max = max(binder_min, min(250, binder_max))
    normalized["binder_length_min"] = binder_min
    normalized["binder_length_max"] = binder_max

    try:
        num_designs = int(normalized.get("num_designs", 5))
    except (TypeError, ValueError):
        num_designs = 5
    normalized["num_designs"] = max(1, min(100, num_designs))

    design_mode = str(normalized.get("design_mode", "default")).strip().lower()
    if design_mode not in ALLOWED_DESIGN_MODES:
        raise ValueError(f"BindCraft `design_mode` must be one of {sorted(ALLOWED_DESIGN_MODES)}.")
    normalized["design_mode"] = design_mode

    normalized["binder_name"] = str(normalized.get("binder_name", "binder")).strip() or "binder"

    return normalized


def _build_settings_json(normalized: dict, pdb_path: str, output_dir: str) -> dict:
    hotspot = normalized["hotspot_residues"] or None
    return {
        "design_path": output_dir,
        "binder_name": normalized["binder_name"],
        "starting_pdb": pdb_path,
        "chains": normalized["chains"],
        "target_hotspot_residues": hotspot,
        "lengths": [normalized["binder_length_min"], normalized["binder_length_max"]],
        "number_of_final_designs": normalized["num_designs"],
    }


def _get_filter_path(design_mode: str) -> str:
    filters_dir = os.path.join(BINDCRAFT_DIR, "settings_filters")
    if design_mode == "peptide":
        candidate = os.path.join(filters_dir, "peptide_filters.json")
        if os.path.isfile(candidate):
            return candidate
    default = os.path.join(filters_dir, "default_filters.json")
    if os.path.isfile(default):
        return default
    return ""


def _get_advanced_path(design_mode: str) -> str:
    advanced_dir = os.path.join(BINDCRAFT_DIR, "settings_advanced")
    mode_map = {
        "default": "default_4stage_multimer.json",
        "betasheet": "betasheet_4stage_multimer.json",
        "peptide": "peptide_4stage_multimer.json",
    }
    candidate = os.path.join(advanced_dir, mode_map.get(design_mode, "default_4stage_multimer.json"))
    if os.path.isfile(candidate):
        return candidate
    for fname in sorted(os.listdir(advanced_dir)) if os.path.isdir(advanced_dir) else []:
        if fname.startswith(design_mode) and fname.endswith(".json"):
            return os.path.join(advanced_dir, fname)
    return ""


def _collect_designs(output_dir: str) -> tuple[list[dict], list[str]]:
    designs = []
    pdb_contents = []

    accepted_dir = os.path.join(output_dir, "Accepted")
    if not os.path.isdir(accepted_dir):
        for candidate in ["accepted", "Ranked", "ranked"]:
            alt = os.path.join(output_dir, candidate)
            if os.path.isdir(alt):
                accepted_dir = alt
                break

    ranked_dir = os.path.join(accepted_dir, "Ranked")
    search_dir = ranked_dir if os.path.isdir(ranked_dir) else accepted_dir

    if os.path.isdir(search_dir):
        for fname in sorted(os.listdir(search_dir)):
            if fname.endswith(".pdb"):
                pdb_path = os.path.join(search_dir, fname)
                with open(pdb_path, encoding="utf-8") as f:
                    content = f.read()
                pdb_contents.append(content[:5000])
                designs.append({"name": fname, "path": pdb_path})

    stats_file = os.path.join(output_dir, "final_design_stats.csv")
    if os.path.isfile(stats_file):
        try:
            with open(stats_file, encoding="utf-8") as f:
                stats_text = f.read()
            lines = stats_text.strip().split("\n")
            if len(lines) > 1:
                headers = lines[0].split(",")
                for i, design in enumerate(designs):
                    if i + 1 < len(lines):
                        values = lines[i + 1].split(",")
                        for h, v in zip(headers, values):
                            h = h.strip()
                            if h in ("i_pTM", "i_pAE", "pLDDT", "Binder_RMSD"):
                                try:
                                    design[h] = float(v.strip())
                                except ValueError:
                                    pass
        except Exception:
            pass

    return designs, pdb_contents


def run(
    target_pdb="",
    pdb_text="",
    chains="A",
    hotspot_residues="",
    binder_length_min=65,
    binder_length_max=150,
    num_designs=5,
    design_mode="default",
    binder_name="binder",
    session_id="",
    **kwargs,
):
    try:
        normalized = normalize_args({
            "target_pdb": target_pdb,
            "pdb_text": pdb_text,
            "chains": chains,
            "hotspot_residues": hotspot_residues,
            "binder_length_min": binder_length_min,
            "binder_length_max": binder_length_max,
            "num_designs": num_designs,
            "design_mode": design_mode,
            "binder_name": binder_name,
            "session_id": session_id,
        })
    except ValueError as exc:
        return {"summary": f"Error: {exc}", "error": "invalid_args"}

    target_pdb_content = normalized["target_pdb"]
    n_residues = _count_residues(target_pdb_content)
    max_binder = normalized["binder_length_max"]
    total_residues = n_residues + max_binder

    if total_residues > 950:
        return {
            "summary": (
                f"Error: Complex too large ({total_residues} residues). "
                f"Target has {n_residues} residues + max binder {max_binder}. "
                "80GB GPU supports ~950 residues. Trim the target or reduce binder length."
            ),
            "error": "complex_too_large",
        }

    t0 = time.time()
    vram_before = _get_gpu_vram_mb()
    stop_event = threading.Event()
    vram_results = {"peak": vram_before}
    monitor = threading.Thread(target=_monitor_vram, args=(stop_event, vram_results), daemon=True)
    monitor.start()

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "target.pdb")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)
        with open(pdb_path, "w", encoding="utf-8") as f:
            f.write(target_pdb_content)

        settings = _build_settings_json(normalized, pdb_path, output_dir)
        settings_path = os.path.join(tmpdir, "settings.json")
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)

        bindcraft_script = os.path.join(BINDCRAFT_DIR, "bindcraft.py")
        if not os.path.isfile(bindcraft_script):
            stop_event.set()
            monitor.join(timeout=2)
            return {
                "summary": "Error: BindCraft not installed at expected path.",
                "error": "not_installed",
            }

        cmd = [
            sys.executable,
            bindcraft_script,
            "--settings", settings_path,
        ]

        filter_path = _get_filter_path(normalized["design_mode"])
        if filter_path:
            cmd.extend(["--filters", filter_path])

        advanced_path = _get_advanced_path(normalized["design_mode"])
        if advanced_path:
            cmd.extend(["--advanced", advanced_path])

        t_inference = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,
                cwd=BINDCRAFT_DIR,
            )
        except subprocess.TimeoutExpired:
            stop_event.set()
            monitor.join(timeout=2)
            return {"summary": "Error: BindCraft timed out (2h limit).", "error": "timeout"}
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
            "target_residues": n_residues,
            "total_complex_residues": total_residues,
        }

        if result.returncode != 0:
            stderr_tail = result.stderr[-1000:] if result.stderr else ""
            stdout_tail = result.stdout[-500:] if result.stdout else ""
            return {
                "summary": f"Error: BindCraft failed (exit {result.returncode}): {stderr_tail[:500]}",
                "error": stderr_tail,
                "stdout": stdout_tail,
                "metrics": metrics,
            }

        designs, pdb_contents = _collect_designs(output_dir)

        if not designs:
            return {
                "summary": (
                    f"BindCraft completed but no designs passed filters. "
                    f"Target had {n_residues} residues, binder range "
                    f"{normalized['binder_length_min']}-{normalized['binder_length_max']}. "
                    f"Try different hotspots, binder lengths, or target trimming."
                ),
                "error": "no_passing_designs",
                "stdout": result.stdout[-500:],
                "metrics": metrics,
            }

        design_summaries = []
        for d in designs:
            summary = d["name"]
            if "i_pTM" in d:
                summary += f" (i_pTM={d['i_pTM']:.3f})"
            design_summaries.append(summary)

        return {
            "summary": (
                f"BindCraft: generated {len(designs)} binder designs for "
                f"{n_residues}-residue target. "
                f"Binder range: {normalized['binder_length_min']}-"
                f"{normalized['binder_length_max']} aa. "
                f"Top designs: {', '.join(design_summaries[:3])}"
            ),
            "designs": designs,
            "pdb_contents": pdb_contents[:10],
            "num_designs": len(designs),
            "metrics": metrics,
        }
