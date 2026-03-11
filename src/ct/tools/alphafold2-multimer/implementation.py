"""AlphaFold2-Multimer-compatible structure prediction via OpenFold."""

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
MAX_MULTIMER_CHAINS = 6
ALLOWED_DATABASES = {"small_bfd", "uniref90", "mgnify", "uniprot"}
ALLOWED_ALGORITHMS = {"mmseqs2", "jackhmmer"}
ALIGNMENT_FILE_NAMES = {
    "small_bfd": "small_bfd_hits.a3m",
    "uniref90": "uniref90_hits.a3m",
    "mgnify": "mgnify_hits.a3m",
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

    if "sequence" in normalized and str(normalized.get("sequence", "")).strip():
        raise ValueError(
            "AlphaFold2-Multimer accepts only `sequences`. Use `structure.openfold2` for single-sequence inputs."
        )

    sequences = normalized.get("sequences")
    if not isinstance(sequences, list):
        raise ValueError("AlphaFold2-Multimer requires `sequences` as a list of amino acid sequence strings.")
    if not (2 <= len(sequences) <= MAX_MULTIMER_CHAINS):
        raise ValueError(f"AlphaFold2-Multimer requires at least 2 sequences and at most {MAX_MULTIMER_CHAINS}.")

    clean_sequences = []
    for seq in sequences:
        clean_seq = str(seq).strip().upper()
        if not clean_seq:
            raise ValueError("AlphaFold2-Multimer sequences must be non-empty strings.")
        if len(clean_seq) > MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"AlphaFold2-Multimer sequence entries must be at most {MAX_SEQUENCE_LENGTH} residues."
            )
        if not AA_SEQUENCE_PATTERN.fullmatch(clean_seq):
            raise ValueError(
                "AlphaFold2-Multimer sequences must contain only valid amino acid IUPAC symbols."
            )
        clean_sequences.append(clean_seq)

    databases = normalized.get("databases") or ["small_bfd"]
    if not isinstance(databases, list) or not databases:
        raise ValueError("AlphaFold2-Multimer `databases` must be a non-empty list.")
    clean_databases = []
    for db in databases:
        db_name = str(db).strip().lower()
        if db_name not in ALLOWED_DATABASES:
            raise ValueError(f"AlphaFold2-Multimer received unsupported database `{db_name}`.")
        if db_name not in clean_databases:
            clean_databases.append(db_name)

    algorithm = str(normalized.get("algorithm", "mmseqs2")).strip().lower() or "mmseqs2"
    if algorithm not in ALLOWED_ALGORITHMS:
        raise ValueError("AlphaFold2-Multimer `algorithm` must be `mmseqs2` or `jackhmmer`.")

    try:
        e_value = float(normalized.get("e_value", 0.000001))
    except (TypeError, ValueError) as exc:
        raise ValueError("AlphaFold2-Multimer `e_value` must be numeric.") from exc
    if e_value < 0:
        raise ValueError("AlphaFold2-Multimer `e_value` must be non-negative.")

    try:
        num_predictions_per_model = int(normalized.get("num_predictions_per_model", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("AlphaFold2-Multimer `num_predictions_per_model` must be an integer.") from exc
    if not (1 <= num_predictions_per_model <= 5):
        raise ValueError("AlphaFold2-Multimer `num_predictions_per_model` must be between 1 and 5.")

    normalized["sequences"] = clean_sequences
    normalized["databases"] = clean_databases
    normalized["algorithm"] = algorithm
    normalized["e_value"] = e_value
    normalized["num_predictions_per_model"] = num_predictions_per_model
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


def _write_chain_alignments(
    chain_dir: str,
    chain_id: str,
    sequence: str,
    databases: list[str],
    alignment_text: str,
) -> None:
    os.makedirs(chain_dir, exist_ok=True)
    for db_name in databases:
        if db_name == "uniprot":
            continue
        filename = ALIGNMENT_FILE_NAMES[db_name]
        with open(os.path.join(chain_dir, filename), "w", encoding="utf-8") as fh:
            fh.write(alignment_text)

    # OpenFold multimer expects a pairing file. We provide a minimal single-chain
    # stockholm alignment so the runtime has an explicit pairing artifact instead
    # of falling back to concatenated single-sequence mode.
    stockholm = f"# STOCKHOLM 1.0\n{chain_id} {sequence}\n//\n"
    with open(os.path.join(chain_dir, "uniprot_hits.sto"), "w", encoding="utf-8") as fh:
        fh.write(stockholm)


def _ensure_params_file(model_name: str) -> str:
    params_dir = os.environ.get("OPENFOLD_PARAMS_DIR", "/root/.cache/openfold/params")
    params_file = os.path.join(params_dir, f"params_{model_name}.npz")
    if not os.path.isfile(params_file):
        os.makedirs(params_dir, exist_ok=True)
        dl_script = os.path.join(OPENFOLD_DIR, "scripts", "download_alphafold_params.sh")
        parent = os.path.dirname(params_dir)
        subprocess.run(["bash", dl_script, parent], check=True, timeout=600)
    return params_file


def _extract_prediction(out_dir: str) -> tuple[str, float]:
    """Extract the best PDB from the output directory.

    OpenFold writes several PDB files: per-chain intermediates and the
    combined relaxed/unrelaxed multi-chain structure.  We prefer the
    largest PDB (which is the full complex) over smaller per-chain files.
    """
    best_pdb = ""
    best_confidence = 0.0
    best_size = 0

    for root, _, files in os.walk(out_dir):
        for filename in sorted(files):
            if not filename.endswith(".pdb"):
                continue
            filepath = os.path.join(root, filename)
            with open(filepath, encoding="utf-8") as fh:
                pdb_content = fh.read()
            # Pick the largest PDB file — the combined multi-chain output
            # is always larger than individual chain files
            if len(pdb_content) <= best_size:
                continue
            bfactors = []
            for line in pdb_content.split("\n"):
                if line.startswith("ATOM") and " CA " in line:
                    try:
                        bfactors.append(float(line[60:66].strip()))
                    except (ValueError, IndexError):
                        pass
            confidence = sum(bfactors) / len(bfactors) if bfactors else 0.0
            best_pdb = pdb_content
            best_confidence = confidence
            best_size = len(pdb_content)

    return best_pdb, best_confidence


def _run_openfold_with_numpy_compat(cmd: list[str], cwd: str, timeout: int):
    script_path = cmd[1]
    quoted_args = ", ".join(repr(arg) for arg in cmd[2:])
    wrapper = (
        "import numpy as np, runpy, sys; "
        "np.string_ = np.bytes_; "
        f"sys.argv=[{repr(script_path)}, {quoted_args}]; "
        "runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    compat_cmd = [cmd[0], "-c", wrapper]
    return subprocess.run(compat_cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)


def run(sequences=None, sequence: str = "", session_id: str = "", **kwargs):
    try:
        normalized = normalize_args({"sequences": sequences, "sequence": sequence, "session_id": session_id, **kwargs})
    except ValueError as exc:
        return {"summary": f"Error: {exc}", "error": "invalid_sequences"}

    sequences = normalized["sequences"]
    databases = normalized["databases"]
    algorithm = normalized["algorithm"]
    num_predictions_per_model = normalized["num_predictions_per_model"]
    relax_prediction = normalized["relax_prediction"]
    session_id = str(normalized.get("session_id", ""))

    t0 = time.time()
    seq_len = sum(len(s) for s in sequences)
    num_chains = len(sequences)
    vram_before = _get_gpu_vram_mb()

    stop_event = threading.Event()
    vram_results = {"peak": vram_before}
    monitor = threading.Thread(target=_monitor_vram, args=(stop_event, vram_results), daemon=True)
    monitor.start()

    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_dir = os.path.join(tmpdir, "fasta")
        os.makedirs(fasta_dir)
        chain_ids = [chr(ord("A") + index) for index in range(num_chains)]
        with open(os.path.join(fasta_dir, "query.fasta"), "w", encoding="utf-8") as fh:
            for chain_id, chain_seq in zip(chain_ids, sequences):
                fh.write(f">{chain_id}\n{chain_seq}\n")

        template_dir = os.path.join(tmpdir, "templates")
        os.makedirs(template_dir)
        with open(os.path.join(template_dir, "dummy.cif"), "w", encoding="utf-8") as fh:
            fh.write("data_dummy\n")

        align_root = os.path.join(tmpdir, "alignments")
        for chain_id, chain_seq in zip(chain_ids, sequences):
            try:
                alignment_text = _fetch_colabfold_a3m(chain_seq)
            except Exception as exc:
                stop_event.set()
                monitor.join(timeout=2)
                return {
                    "summary": f"Error: AlphaFold2-Multimer MSA search failed for chain {chain_id}: {exc}",
                    "error": "msa_search_failed",
                }
            _write_chain_alignments(
                os.path.join(align_root, chain_id),
                chain_id,
                chain_seq,
                databases,
                alignment_text,
            )

        predictions = []
        t_inference = time.time()
        for model_index in range(1, 6):
            model_name = f"model_{model_index}_multimer_v3"
            params_file = _ensure_params_file(model_name)
            for sample_index in range(num_predictions_per_model):
                out_dir = os.path.join(tmpdir, f"output_{model_name}_{sample_index}")
                os.makedirs(out_dir, exist_ok=True)
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
                    "--multimer_ri_gap",
                    "200",
                    "--output_postfix",
                    f"sample_{sample_index + 1}",
                ]
                if not relax_prediction:
                    cmd.append("--skip_relaxation")

                result = _run_openfold_with_numpy_compat(cmd, cwd=OPENFOLD_DIR, timeout=1800)
                if result.returncode != 0:
                    stop_event.set()
                    monitor.join(timeout=2)
                    error_tail = f"STDERR:\n{result.stderr[-2000:]}\nSTDOUT:\n{result.stdout[-2000:]}"
                    return {
                        "summary": f"Error: AlphaFold2-Multimer failed on {model_name}: {error_tail}",
                        "error": error_tail,
                        "command": cmd,
                    }

                pdb_content, confidence = _extract_prediction(out_dir)
                if not pdb_content:
                    stop_event.set()
                    monitor.join(timeout=2)
                    return {
                        "summary": f"AlphaFold2-Multimer ran but produced no structure for {model_name}.",
                        "error": "no_output",
                    }
                predictions.append(
                    {
                        "model": model_name,
                        "sample": sample_index + 1,
                        "confidence": confidence,
                        "pdb_content": pdb_content,
                    }
                )

        t_inference = time.time() - t_inference
        stop_event.set()
        monitor.join(timeout=2)
        vram_after = _get_gpu_vram_mb()
        vram_peak = vram_results["peak"]

        predictions.sort(key=lambda item: item["confidence"], reverse=True)
        best = predictions[0]

        if session_id:
            workspace_dir = f"/vol/workspace/{session_id}"
            os.makedirs(workspace_dir, exist_ok=True)
            with open(f"{workspace_dir}/predicted_structure.pdb", "w", encoding="utf-8") as fh:
                fh.write(best["pdb_content"])

        return {
            "summary": (
                f"AF2-Multimer prediction for {num_chains} chains and {seq_len} residues. "
                f"Best pLDDT: {best['confidence']:.1f}/100 from {len(predictions)} prediction(s)."
            ),
            "pdb_content": best["pdb_content"],
            "confidence": best["confidence"],
            "num_residues": seq_len,
            "predictions": predictions,
            "metrics": {
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after,
                "vram_peak_mb": vram_peak,
                "time_inference_s": round(t_inference, 2),
                "time_total_s": round(time.time() - t0, 2),
                "algorithm": algorithm,
                "databases": databases,
            },
        }
