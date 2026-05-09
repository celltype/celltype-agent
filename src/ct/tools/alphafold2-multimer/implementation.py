"""AlphaFold2-Multimer-compatible structure prediction via OpenFold."""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

OPENFOLD_DIR = "/opt/openfold"
AA_SEQUENCE_PATTERN = re.compile(r"^[ARNDCQEGHILKMFPSTWYV]+$")
MAX_SEQUENCE_LENGTH = 4096
MAX_MULTIMER_CHAINS = 6
ALLOWED_DATABASES = {"small_bfd", "uniref90", "mgnify", "uniprot"}
ALLOWED_ALGORITHMS = {"mmseqs2"}
ALLOWED_MODEL_IDS = {1, 2, 3, 4, 5}
ALIGNMENT_FILE_NAMES = {
    "small_bfd": "small_bfd_hits.a3m",
    "uniref90": "uniref90_hits.a3m",
    "mgnify": "mgnify_hits.a3m",
}
_MSA_SEARCH_MODULE: Any | None = None


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

    msas = normalized.get("msas")
    if msas in (None, ""):
        clean_msas = []
    elif not isinstance(msas, list):
        raise ValueError("AlphaFold2-Multimer `msas` must be a list aligned to `sequences`.")
    elif len(msas) != len(clean_sequences):
        raise ValueError("AlphaFold2-Multimer `msas` must have the same length as `sequences`.")
    else:
        clean_msas = []
        for index, msa_text in enumerate(msas):
            if msa_text in (None, ""):
                clean_msas.append("")
                continue
            msa_string = str(msa_text)
            if not msa_string.strip():
                clean_msas.append("")
                continue
            if _extract_query_sequence_from_a3m(msa_string) != clean_sequences[index]:
                raise ValueError(
                    "AlphaFold2-Multimer `msas` entries must start with the exact query sequence for each chain."
                )
            clean_msas.append(msa_string)

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
        raise ValueError("AlphaFold2-Multimer currently supports only `mmseqs2` for local MSA generation.")

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

    try:
        iterations = int(normalized.get("iterations", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("AlphaFold2-Multimer `iterations` must be an integer.") from exc
    if not (1 <= iterations <= 6):
        raise ValueError("AlphaFold2-Multimer `iterations` must be between 1 and 6.")

    selected_models = normalized.get("selected_models")
    if selected_models in (None, "", []):
        clean_selected_models = [1, 2, 3, 4, 5]
    elif not isinstance(selected_models, list):
        raise ValueError("AlphaFold2-Multimer `selected_models` must be a list of integers from 1 to 5.")
    else:
        clean_selected_models = []
        for model_id in selected_models:
            try:
                model_int = int(model_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "AlphaFold2-Multimer `selected_models` must contain integers from 1 to 5."
                ) from exc
            if model_int not in ALLOWED_MODEL_IDS:
                raise ValueError("AlphaFold2-Multimer `selected_models` must contain values from 1 to 5.")
            if model_int not in clean_selected_models:
                clean_selected_models.append(model_int)

    normalized["sequences"] = clean_sequences
    normalized["databases"] = clean_databases
    normalized["algorithm"] = algorithm
    normalized["e_value"] = e_value
    normalized["num_predictions_per_model"] = num_predictions_per_model
    normalized["iterations"] = iterations
    normalized["selected_models"] = clean_selected_models
    normalized["relax_prediction"] = bool(normalized.get("relax_prediction", False))
    normalized["msa_cache_dir"] = str(normalized.get("msa_cache_dir", "")).strip()
    normalized["msa_backend"] = str(normalized.get("msa_backend", "")).strip().lower()
    normalized["msa_server_url"] = str(normalized.get("msa_server_url", "")).strip()
    normalized["msa_server_search_type"] = str(
        normalized.get("msa_server_search_type", "")
    ).strip().lower()
    normalized["msas"] = clean_msas
    return normalized


def _load_msa_search_module():
    global _MSA_SEARCH_MODULE
    if _MSA_SEARCH_MODULE is not None:
        return _MSA_SEARCH_MODULE

    current_file = Path(__file__).resolve()
    candidates = [
        current_file.parent.parent / "msa-search" / "implementation.py",
        current_file.parent.parent / "msa_search" / "implementation.py",
        current_file.parent / "msa_search.py",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        spec = importlib.util.spec_from_file_location("ct_msa_search_impl", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MSA_SEARCH_MODULE = module
        return module

    raise FileNotFoundError("Unable to locate the msa-search implementation required by AlphaFold2-Multimer.")


def _msa_search_database_arg(databases: list[str]) -> list[str]:
    search_databases = [db_name for db_name in databases if db_name != "uniprot"]
    if any(db_name in {"small_bfd", "mgnify"} for db_name in search_databases):
        return ["all"]
    return ["uniref30_2302"]


def _extract_query_sequence_from_a3m(msa_text: str) -> str:
    query_lines: list[str] = []
    in_query = False
    for raw_line in msa_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if in_query and query_lines:
                break
            in_query = True
            continue
        if in_query:
            query_lines.append(line)
    query = "".join(query_lines)
    return "".join(ch for ch in query if "A" <= ch <= "Z")


def _cache_precomputed_alignment(
    sequence: str,
    databases: list[str],
    msa_text: str,
    cache_dir: str,
    in_memory_cache: dict[str, str],
) -> None:
    msa_search = _load_msa_search_module()
    msa_search.cache_alignment_for_sequence(
        sequence,
        msa_text,
        database=_msa_search_database_arg(databases),
        cache_dir=cache_dir,
        in_memory_cache=in_memory_cache,
    )


def _get_alignment_for_sequence(
    sequence: str,
    databases: list[str],
    e_value: float,
    iterations: int,
    cache_dir: str,
    in_memory_cache: dict[str, str],
    cache_stats: dict[str, int],
    backend: str,
    server_url: str,
    server_search_type: str,
) -> str:
    msa_search = _load_msa_search_module()
    result = msa_search.get_alignment_for_sequence(
        sequence,
        database=_msa_search_database_arg(databases),
        e_value=e_value,
        iterations=iterations,
        cache_dir=cache_dir,
        in_memory_cache=in_memory_cache,
        cache_stats=cache_stats,
        backend=backend,
        server_url=server_url,
        server_search_type=server_search_type,
    )
    return result["msa"]


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
    iterations = normalized["iterations"]
    selected_models = normalized["selected_models"]
    relax_prediction = normalized["relax_prediction"]
    msa_cache_dir = normalized["msa_cache_dir"] or os.environ.get("AF2_MULTIMER_MSA_CACHE_DIR", "").strip()
    msa_backend = normalized["msa_backend"] or os.environ.get("AF2_MULTIMER_MSA_BACKEND", "").strip().lower()
    msa_server_url = normalized["msa_server_url"] or os.environ.get("AF2_MULTIMER_MSA_SERVER_URL", "").strip()
    msa_server_search_type = (
        normalized["msa_server_search_type"]
        or os.environ.get("AF2_MULTIMER_MSA_SERVER_SEARCH_TYPE", "").strip().lower()
    )
    provided_msas = normalized["msas"]
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
        alignment_cache: dict[str, str] = {}
        cache_stats = {"memory_hits": 0, "disk_hits": 0, "misses": 0, "precomputed_hits": 0}
        for index, (chain_id, chain_seq) in enumerate(zip(chain_ids, sequences)):
            try:
                if index < len(provided_msas) and provided_msas[index]:
                    alignment_text = provided_msas[index]
                    _cache_precomputed_alignment(chain_seq, databases, alignment_text, msa_cache_dir, alignment_cache)
                    cache_stats["precomputed_hits"] += 1
                else:
                    alignment_text = _get_alignment_for_sequence(
                        chain_seq,
                        databases,
                        normalized["e_value"],
                        iterations,
                        msa_cache_dir,
                        alignment_cache,
                        cache_stats,
                        msa_backend,
                        msa_server_url,
                        msa_server_search_type,
                    )
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
        for model_index in selected_models:
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
                "selected_models": selected_models,
                "msa_cache_dir": msa_cache_dir,
                "msa_cache_memory_hits": cache_stats["memory_hits"],
                "msa_cache_disk_hits": cache_stats["disk_hits"],
                "msa_cache_misses": cache_stats["misses"],
                "msa_precomputed_hits": cache_stats["precomputed_hits"],
            },
        }
