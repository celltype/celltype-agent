"""RFDiffusion — thin wrapper around their run_inference.py CLI.
Hardware: A10G/H100 GPU, ~8-16GB VRAM.
"""
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request


RFDIFFUSION_DIR = "/app/RFdiffusion"
RFDIFFUSION_CHECKPOINT_URLS = {
    "Base_ckpt.pt": "http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt",
    "Complex_base_ckpt.pt": "http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt",
}
RFDIFFUSION_ALLOWED_MODES = {"monomer", "binder"}
RFDIFFUSION_ALLOWED_ARGS = {
    "target_pdb",
    "pdb_text",
    "mode",
    "num_designs",
    "receptor_chain",
    "binder_length",
    "hotspot_residues",
    "diffusion_steps",
    "session_id",
}
RFDIFFUSION_ALLOWED_PATH_SUFFIXES = {".pdb", ".cif", ".mmcif", ".ent"}


def _get_gpu_vram_mb():
    """Get current GPU VRAM usage in MB via nvidia-smi."""
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


def _require_inline_pdb_text(pdb_text: str) -> str:
    pdb_text = str(pdb_text or "").strip()
    if not pdb_text:
        raise ValueError("RFDiffusion requires non-empty inline PDB text in `pdb_text`.")

    lines = [line for line in pdb_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("RFDiffusion requires non-empty inline PDB text in `pdb_text`.")

    if not any(line.startswith(("ATOM", "HETATM")) for line in lines):
        raise ValueError("RFDiffusion currently supports inline PDB text with ATOM/HETATM records.")

    return pdb_text


def _load_target_pdb_file(target_pdb: str) -> str:
    path = str(target_pdb or "").strip()
    if not path:
        raise ValueError("RFDiffusion requires either `pdb_text` or `target_pdb`.")
    if not os.path.isfile(path):
        raise ValueError("RFDiffusion `target_pdb` must point to an existing local structure file.")
    if os.path.splitext(path)[1].lower() not in RFDIFFUSION_ALLOWED_PATH_SUFFIXES:
        raise ValueError(
            "RFDiffusion `target_pdb` must be a local .pdb, .cif, .mmcif, or .ent file."
        )
    with open(path, encoding="utf-8") as fh:
        return _require_inline_pdb_text(fh.read())


def _parse_chain_ranges(target_pdb: str) -> dict[str, tuple[int, int]]:
    chain_ranges: dict[str, tuple[int, int]] = {}
    for line in target_pdb.splitlines():
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 27:
            continue
        chain_id = (line[21] or "A").strip() or "A"
        try:
            residue_number = int(line[22:26].strip())
        except ValueError:
            continue
        current = chain_ranges.get(chain_id)
        if current is None:
            chain_ranges[chain_id] = (residue_number, residue_number)
        else:
            chain_ranges[chain_id] = (
                min(current[0], residue_number),
                max(current[1], residue_number),
            )
    return chain_ranges


def _parse_hotspot_residues(hotspot_residues: str, receptor_chain: str) -> str:
    canonical = []
    for raw_token in str(hotspot_residues or "").split(","):
        token = raw_token.strip().upper()
        if not token:
            continue
        if not re.fullmatch(r"[A-Z]\d+", token):
            raise ValueError(
                "RFDiffusion binder mode requires `hotspot_residues` in canonical form like `A54,A57`."
            )
        if token[0] != receptor_chain:
            raise ValueError(
                f"RFDiffusion binder mode requires all hotspot residues to use receptor chain `{receptor_chain}`."
            )
        canonical.append(token)

    if not canonical:
        raise ValueError(
            "RFDiffusion binder mode requires non-empty canonical `hotspot_residues` like `A54,A57`."
        )
    return ",".join(canonical)


def _normalize_num_designs(value) -> int:
    try:
        num_designs = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("RFDiffusion requires `num_designs` to be an integer.") from exc
    if num_designs < 1:
        raise ValueError("RFDiffusion requires `num_designs` to be at least 1.")
    return num_designs


def _normalize_binder_length(value) -> int:
    try:
        binder_length = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("RFDiffusion binder mode requires integer `binder_length`.") from exc
    if binder_length < 1:
        raise ValueError("RFDiffusion binder mode requires `binder_length` to be at least 1.")
    return binder_length


def normalize_args(args: dict) -> dict:
    unexpected_args = sorted(set(args) - RFDIFFUSION_ALLOWED_ARGS)
    if unexpected_args:
        raise ValueError(f"RFDiffusion received unsupported arguments: {', '.join(unexpected_args)}.")

    normalized = dict(args)
    target_pdb = str(normalized.get("target_pdb", "")).strip()
    pdb_text = str(normalized.get("pdb_text", "")).strip()
    if target_pdb and pdb_text:
        pdb_text = pdb_text
        target_pdb = ""
    elif not target_pdb and not pdb_text:
        raise ValueError(
            "RFDiffusion requires `pdb_text` (inline PDB content) or `target_pdb` (file path)."
        )
    inline_content = (
        _require_inline_pdb_text(pdb_text)
        if pdb_text
        else _load_target_pdb_file(target_pdb)
    )
    normalized["target_pdb"] = inline_content
    normalized["pdb_text"] = inline_content
    normalized["mode"] = str(normalized.get("mode", "monomer")).strip().lower()
    normalized["num_designs"] = _normalize_num_designs(normalized.get("num_designs", 3))
    raw_steps = normalized.get("diffusion_steps")
    if raw_steps is not None:
        try:
            steps = int(raw_steps)
        except (TypeError, ValueError):
            steps = 50
        normalized["diffusion_steps"] = max(10, min(90, steps))
    else:
        normalized["diffusion_steps"] = 50

    if normalized["mode"] not in RFDIFFUSION_ALLOWED_MODES:
        raise ValueError("RFDiffusion `mode` must be either `monomer` or `binder`.")

    chain_ranges = _parse_chain_ranges(normalized["target_pdb"])
    if not chain_ranges:
        raise ValueError("RFDiffusion could not extract chain ranges from the supplied PDB text.")

    if normalized["mode"] == "monomer":
        receptor_chain_value = str(normalized.get("receptor_chain", "")).strip()
        hotspot_value = str(normalized.get("hotspot_residues", "")).strip()
        binder_length_value = normalized.get("binder_length", 0)
        if receptor_chain_value or hotspot_value or int(binder_length_value or 0) > 0:
            raise ValueError(
                "RFDiffusion monomer mode does not accept `receptor_chain`, `binder_length`, or `hotspot_residues`."
            )
        normalized["receptor_chain"] = ""
        normalized["binder_length"] = 0
        normalized["hotspot_residues"] = ""
        return normalized

    receptor_chain = str(normalized.get("receptor_chain", "")).strip().upper()
    if not receptor_chain:
        raise ValueError("RFDiffusion binder mode requires explicit `receptor_chain`.")
    if receptor_chain not in chain_ranges:
        raise ValueError(
            f"RFDiffusion binder mode could not find receptor chain `{receptor_chain}` in `target_pdb`."
        )

    normalized["receptor_chain"] = receptor_chain
    normalized["binder_length"] = _normalize_binder_length(normalized.get("binder_length"))
    normalized["hotspot_residues"] = _parse_hotspot_residues(
        normalized.get("hotspot_residues", ""),
        receptor_chain,
    )
    return normalized


def _checkpoint_looks_valid(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    if os.path.getsize(path) < 100_000_000:
        return False
    with open(path, "rb") as fh:
        magic = fh.read(4)
    return magic.startswith(b"PK\x03\x04") or magic.startswith(b"\x80")


def _download_checkpoint(ckpt_name: str, ckpt_path: str) -> None:
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    tmp_path = f"{ckpt_path}.tmp"
    urllib.request.urlretrieve(RFDIFFUSION_CHECKPOINT_URLS[ckpt_name], tmp_path)
    os.replace(tmp_path, ckpt_path)


def _ensure_checkpoint(model_dir: str, ckpt_name: str) -> None:
    ckpt = os.path.join(model_dir, ckpt_name)
    if _checkpoint_looks_valid(ckpt):
        return
    if os.path.exists(ckpt):
        os.remove(ckpt)
    _download_checkpoint(ckpt_name, ckpt)


def _required_checkpoint_names(mode: str) -> list[str]:
    if mode == "binder":
        return ["Base_ckpt.pt", "Complex_base_ckpt.pt"]
    return ["Base_ckpt.pt"]


def _build_inference_args(
    target_pdb: str,
    mode: str,
    receptor_chain: str,
    binder_length: int,
    hotspot_residues: str,
) -> list[str]:
    chain_ranges = _parse_chain_ranges(target_pdb)
    if mode == "monomer":
        n_res = sum(1 for line in target_pdb.splitlines() if line.startswith("ATOM") and " CA " in line)
        if n_res < 1:
            raise ValueError("RFDiffusion monomer mode requires at least one CA atom in `target_pdb`.")
        return [f"contigmap.contigs=[{n_res}-{n_res}]"]

    receptor_start, receptor_end = chain_ranges[receptor_chain]
    return [
        f"contigmap.contigs=[{binder_length}-{binder_length}/0 {receptor_chain}{receptor_start}-{receptor_end}]",
        f"ppi.hotspot_res=[{hotspot_residues}]",
    ]


def run(
    target_pdb="",
    pdb_text="",
    mode="monomer",
    num_designs=3,
    receptor_chain="",
    binder_length=0,
    hotspot_residues="",
    diffusion_steps=50,
    session_id="",
    **kwargs,
):
    try:
        normalized = normalize_args(
            {
                "target_pdb": target_pdb,
                "pdb_text": pdb_text,
                "mode": mode,
                "num_designs": num_designs,
                "receptor_chain": receptor_chain,
                "binder_length": binder_length,
                "hotspot_residues": hotspot_residues,
                "diffusion_steps": diffusion_steps,
                "session_id": session_id,
                **kwargs,
            }
        )
    except ValueError as exc:
        return {"summary": f"Error: {exc}", "error": "invalid_args"}

    target_pdb = normalized["target_pdb"]
    mode = normalized["mode"]
    num_designs = normalized["num_designs"]
    receptor_chain = normalized["receptor_chain"]
    binder_length = normalized["binder_length"]
    hotspot_residues = normalized["hotspot_residues"]
    diffusion_steps = normalized["diffusion_steps"]
    session_id = normalized.get("session_id", "")

    t0 = time.time()
    vram_before = _get_gpu_vram_mb()

    stop_event = threading.Event()
    vram_results = {"peak": vram_before}
    monitor = threading.Thread(target=_monitor_vram, args=(stop_event, vram_results), daemon=True)
    monitor.start()

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "target.pdb")
        out_dir = os.path.join(tmpdir, "output")
        os.makedirs(out_dir)
        with open(pdb_path, "w", encoding="utf-8") as f:
            f.write(target_pdb)

        model_dir = os.environ.get("RFDIFFUSION_MODEL_DIR", "/root/.cache/rfdiffusion")
        try:
            for ckpt_name in _required_checkpoint_names(mode):
                _ensure_checkpoint(model_dir, ckpt_name)
            inference_args = _build_inference_args(
                target_pdb,
                mode,
                receptor_chain,
                binder_length,
                hotspot_residues,
            )
        except Exception as exc:
            stop_event.set()
            monitor.join(timeout=2)
            return {
                "summary": f"Error: Failed to prepare RFdiffusion inputs: {exc}",
                "error": "invalid_args",
            }

        cmd = [
            sys.executable,
            os.path.join(RFDIFFUSION_DIR, "scripts", "run_inference.py"),
            f"inference.output_prefix={out_dir}/design",
            f"inference.input_pdb={pdb_path}",
            f"inference.num_designs={num_designs}",
            f"inference.model_directory_path={model_dir}",
            f"diffuser.T={diffusion_steps}",
        ]
        cmd.extend(inference_args)

        if mode == "binder":
            cmd.append(
                f"inference.ckpt_override_path={os.path.join(model_dir, 'Complex_base_ckpt.pt')}"
            )

        t_inference = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=RFDIFFUSION_DIR,
                env={**os.environ, "DGLBACKEND": "pytorch"},
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

        metrics = {
            "vram_before_mb": vram_before,
            "vram_after_mb": vram_after,
            "vram_peak_mb": vram_peak,
            "time_inference_s": round(t_inference, 2),
            "time_total_s": round(time.time() - t0, 2),
        }

        if result.returncode != 0:
            return {
                "summary": f"Error: RFDiffusion failed: {result.stderr[-500:]}",
                "error": result.stderr[-500:],
                "metrics": metrics,
            }

        designs = []
        scores = []
        for f_name in sorted(os.listdir(out_dir)):
            if f_name.endswith(".pdb"):
                with open(os.path.join(out_dir, f_name), encoding="utf-8") as pf:
                    content = pf.read()
                designs.append(content[:3000])
                scores.append(sum(1 for line in content.split("\n") if line.startswith("ATOM")))

        if not designs:
            return {
                "summary": f"RFDiffusion ran but no designs. stdout: {result.stdout[-300:]}",
                "error": "no_output",
                "metrics": metrics,
            }

        return {
            "summary": f"RFDiffusion: generated {len(designs)} backbone designs with {scores[0]} atoms each.",
            "designs": designs,
            "scores": scores,
            "num_designs": len(designs),
            "metrics": metrics,
        }
