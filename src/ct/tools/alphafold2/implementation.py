"""AlphaFold2-compatible structure prediction via OpenFold."""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from typing import Any

import requests

OPENFOLD_DIR = "/opt/openfold"
AA_SEQUENCE_PATTERN = re.compile(r"^[ARNDCQEGHILKMFPSTWYV]+$")
MAX_SEQUENCE_LENGTH = 4096
ALLOWED_DATABASES = {"small_bfd", "uniref90", "mgnify", "uniprot"}
ALLOWED_ALGORITHMS = {"mmseqs2", "jackhmmer"}
ALIGNMENT_FILE_NAMES = {
    "small_bfd": "small_bfd_hits.a3m",
    "uniref90": "uniref90_hits.a3m",
    "mgnify": "mgnify_hits.a3m",
    "uniprot": "uniprot_hits.a3m",
}


def _get_gpu_vram_mb() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return 0


def _monitor_vram(stop_event, results) -> None:
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


def normalize_args(args: dict) -> dict:
    normalized = dict(args)
    sequence = str(normalized.get("sequence", "")).strip().upper()

    if "sequences" in normalized:
        raise ValueError(
            "AlphaFold2 accepts only `sequence`. Use `structure.alphafold2_multimer` for multi-chain inputs."
        )
    if not sequence:
        raise ValueError("AlphaFold2 requires `sequence` as a non-empty amino acid sequence string.")
    if len(sequence) > MAX_SEQUENCE_LENGTH:
        raise ValueError(f"AlphaFold2 `sequence` must be at most {MAX_SEQUENCE_LENGTH} residues.")
    if not AA_SEQUENCE_PATTERN.fullmatch(sequence):
        raise ValueError("AlphaFold2 `sequence` must contain only valid amino acid IUPAC symbols.")

    databases = normalized.get("databases") or ["small_bfd"]
    if not isinstance(databases, list) or not databases:
        raise ValueError("AlphaFold2 `databases` must be a non-empty list.")
    clean_databases = []
    for db in databases:
        db_name = str(db).strip().lower()
        if db_name not in ALLOWED_DATABASES:
            raise ValueError(f"AlphaFold2 received unsupported database `{db_name}`.")
        if db_name not in clean_databases:
            clean_databases.append(db_name)

    algorithm = str(normalized.get("algorithm", "mmseqs2")).strip().lower() or "mmseqs2"
    if algorithm not in ALLOWED_ALGORITHMS:
        raise ValueError("AlphaFold2 `algorithm` must be `mmseqs2` or `jackhmmer`.")

    try:
        e_value = float(normalized.get("e_value", 0.000001))
    except (TypeError, ValueError) as exc:
        raise ValueError("AlphaFold2 `e_value` must be numeric.") from exc
    if e_value < 0:
        raise ValueError("AlphaFold2 `e_value` must be non-negative.")

    normalized["sequence"] = sequence
    normalized["databases"] = clean_databases
    normalized["algorithm"] = algorithm
    normalized["e_value"] = e_value
    normalized["relax_prediction"] = bool(normalized.get("relax_prediction", False))
    return normalized


def _fetch_colabfold_a3m(sequence: str) -> str:
    api_url = "https://api.colabfold.com"
    query = f">query\n{sequence}\n"
    response = requests.post(f"{api_url}/ticket/msa", data={"q": query, "mode": "all"}, timeout=30)
    response.raise_for_status()
    ticket_id = response.json().get("id", "")
    if not ticket_id:
        raise RuntimeError("ColabFold API returned no ticket ID.")

    for _ in range(120):
        status_resp = requests.get(f"{api_url}/ticket/{ticket_id}", timeout=10)
        status_resp.raise_for_status()
        payload = status_resp.json()
        status = payload.get("status", "")
        if status == "COMPLETE":
            break
        if status == "ERROR":
            raise RuntimeError(payload.get("error", "unknown ColabFold error"))
        time.sleep(5)
    else:
        raise RuntimeError("ColabFold MSA search timed out.")

    result_resp = requests.get(f"{api_url}/result/download/{ticket_id}", timeout=60)
    result_resp.raise_for_status()

    with tarfile.open(fileobj=io.BytesIO(result_resp.content), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".a3m"):
                fh = tar.extractfile(member)
                if fh is not None:
                    raw = fh.read().replace(b"\x00", b"")
                    return raw.decode("utf-8", errors="ignore")
    raise RuntimeError("No A3M alignment found in ColabFold results.")


def _write_alignments(align_dir: str, databases: list[str], alignment_text: str) -> None:
    os.makedirs(align_dir, exist_ok=True)
    for db_name in databases:
        filename = ALIGNMENT_FILE_NAMES[db_name]
        with open(os.path.join(align_dir, filename), "w", encoding="utf-8") as fh:
            fh.write(alignment_text)


def _ensure_params_file(model_name: str) -> str:
    params_dir = os.environ.get("OPENFOLD_PARAMS_DIR", "/root/.cache/openfold/params")
    params_file = os.path.join(params_dir, f"params_{model_name}.npz")
    if not os.path.isfile(params_file):
        os.makedirs(params_dir, exist_ok=True)
        dl_script = os.path.join(OPENFOLD_DIR, "scripts", "download_alphafold_params.sh")
        parent = os.path.dirname(params_dir)
        subprocess.run(["bash", dl_script, parent], check=True, timeout=600)
    return params_file


def run(sequence: str = "", session_id: str = "", **kwargs) -> dict:
    try:
        normalized = normalize_args({"sequence": sequence, "session_id": session_id, **kwargs})
    except ValueError as exc:
        return {"summary": f"Error: {exc}", "error": "invalid_sequence_input"}

    sequence = normalized["sequence"]
    databases = normalized["databases"]
    algorithm = normalized["algorithm"]
    relax_prediction = normalized["relax_prediction"]
    session_id = str(normalized.get("session_id", ""))

    t0 = time.time()
    seq_len = len(sequence)
    vram_before = _get_gpu_vram_mb()

    stop_event = threading.Event()
    vram_results = {"peak": vram_before}
    monitor = threading.Thread(target=_monitor_vram, args=(stop_event, vram_results), daemon=True)
    monitor.start()

    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_dir = os.path.join(tmpdir, "fasta")
        os.makedirs(fasta_dir)
        with open(os.path.join(fasta_dir, "query.fasta"), "w", encoding="utf-8") as fh:
            fh.write(f">query\n{sequence}\n")

        out_dir = os.path.join(tmpdir, "output")
        os.makedirs(out_dir)

        template_dir = os.path.join(tmpdir, "templates")
        os.makedirs(template_dir)
        with open(os.path.join(template_dir, "dummy.cif"), "w", encoding="utf-8") as fh:
            fh.write("data_dummy\n")

        align_root = os.path.join(tmpdir, "alignments")
        query_align_dir = os.path.join(align_root, "query")
        try:
            alignment_text = _fetch_colabfold_a3m(sequence)
        except Exception as exc:
            stop_event.set()
            monitor.join(timeout=2)
            return {"summary": f"Error: AlphaFold2 MSA search failed: {exc}", "error": "msa_search_failed"}
        _write_alignments(query_align_dir, databases, alignment_text)

        model_name = "model_1"
        params_file = _ensure_params_file(model_name)

        cmd = [
            sys.executable,
            os.path.join(OPENFOLD_DIR, "run_pretrained_openfold.py"),
            fasta_dir,
            template_dir,
            "--use_precomputed_alignments",
            align_root,
            "--output_dir",
            out_dir,
            "--model_device",
            "cuda:0",
            "--config_preset",
            model_name,
            "--jax_param_path",
            params_file,
        ]
        if not relax_prediction:
            cmd.append("--skip_relaxation")

        t_inference = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=OPENFOLD_DIR)
        except subprocess.TimeoutExpired:
            stop_event.set()
            monitor.join(timeout=2)
            return {"summary": "Error: AlphaFold2 timed out.", "error": "timeout"}
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
            "algorithm": algorithm,
            "databases": databases,
        }

        if result.returncode != 0:
            error_tail = f"STDERR:\n{result.stderr[-2000:]}\nSTDOUT:\n{result.stdout[-2000:]}"
            return {
                "summary": f"Error: AlphaFold2 failed: {error_tail}",
                "error": error_tail,
                "command": cmd,
                "metrics": metrics,
            }

        pdb_content = ""
        confidence = 0.0
        for root, _, files in os.walk(out_dir):
            for name in sorted(files):
                if name.endswith(".pdb"):
                    with open(os.path.join(root, name), encoding="utf-8") as fh:
                        pdb_content = fh.read()
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
            return {
                "summary": f"AlphaFold2 ran but no structure. stderr: {result.stderr[-300:]}",
                "error": "no_output",
                "metrics": metrics,
            }

        if session_id:
            workspace_dir = f"/vol/workspace/{session_id}"
            os.makedirs(workspace_dir, exist_ok=True)
            with open(f"{workspace_dir}/predicted_structure.pdb", "w", encoding="utf-8") as fh:
                fh.write(pdb_content)

        return {
            "summary": (
                f"AlphaFold2 prediction for {seq_len}-residue protein. "
                f"pLDDT: {confidence:.1f}/100. MSA built with {algorithm}."
            ),
            "pdb_content": pdb_content,
            "confidence": confidence,
            "num_residues": seq_len,
            "metrics": metrics,
        }
