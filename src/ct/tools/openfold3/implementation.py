"""OpenFold3 structure prediction implementation."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
import time
from csv import DictReader
from io import StringIO
from typing import Any

PROTEIN_PATTERN = re.compile(r"^[ARNDCQEGHILKMFPSTWYV]+$")
DNA_PATTERN = re.compile(r"^[ACGT]+$")
RNA_PATTERN = re.compile(r"^[ACGU]+$")
MAX_SEQUENCE_LENGTH = 1000
ALLOWED_MOLECULE_TYPES = {"protein", "dna", "rna", "ligand"}
ALLOWED_MSA_FORMATS = {"csv", "a3m"}
ALLOWED_OUTPUT_FORMATS = {"pdb", "mmcif"}


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


def _clean_sequence(value: Any, molecule_type: str) -> str:
    sequence = str(value or "").strip().upper().replace(" ", "").replace("\n", "")
    if not sequence:
        raise ValueError(f"{molecule_type} molecules require `sequence`.")
    if len(sequence) > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"{molecule_type} `sequence` must be at most {MAX_SEQUENCE_LENGTH} characters."
        )

    pattern = {
        "protein": PROTEIN_PATTERN,
        "dna": DNA_PATTERN,
        "rna": RNA_PATTERN,
    }.get(molecule_type)
    if pattern is not None and not pattern.fullmatch(sequence):
        raise ValueError(f"{molecule_type} `sequence` contains invalid characters.")
    return sequence


def _normalize_msa_payload(msa: Any) -> dict:
    if not isinstance(msa, dict) or not msa:
        raise ValueError("`msa` must be an object keyed by database name.")

    normalized = {}
    for db_name, db_payload in msa.items():
        if not isinstance(db_payload, dict) or not db_payload:
            raise ValueError("Each MSA database entry must be an object.")
        normalized_db = {}
        for fmt_name, fmt_payload in db_payload.items():
            if fmt_name not in ALLOWED_MSA_FORMATS:
                raise ValueError(f"Unsupported MSA format `{fmt_name}`.")
            if not isinstance(fmt_payload, dict):
                raise ValueError("MSA format payloads must be objects.")
            alignment = str(fmt_payload.get("alignment", "")).strip()
            declared_format = str(fmt_payload.get("format", "")).strip().lower()
            if not alignment:
                raise ValueError("MSA payloads require non-empty `alignment` text.")
            if declared_format != fmt_name:
                raise ValueError(f"MSA payload format mismatch for `{fmt_name}`.")
            normalized_db[fmt_name] = {
                "alignment": alignment,
                "format": declared_format,
            }
        normalized[db_name] = normalized_db
    return normalized


def _csv_alignment_to_a3m(alignment: str, expected_sequence: str) -> str:
    reader = DictReader(StringIO(alignment))
    if "sequence" not in (reader.fieldnames or []):
        raise ValueError("CSV MSA payloads must include a `sequence` column.")

    records = []
    for index, row in enumerate(reader):
        sequence = str((row or {}).get("sequence", "")).strip()
        if not sequence:
            raise ValueError("CSV MSA payloads must include non-empty `sequence` values.")
        key = str((row or {}).get("key", "")).strip() or f"hit_{index}"
        records.append((key, sequence))

    if not records:
        raise ValueError("CSV MSA payloads must include at least one alignment row.")

    first_sequence = records[0][1].replace("-", "").upper()
    if first_sequence != expected_sequence:
        raise ValueError("The first MSA sequence must exactly match the input sequence.")

    lines = []
    for index, (key, sequence) in enumerate(records):
        header = "query" if index == 0 else key
        lines.append(f">{header}")
        lines.append(sequence)
    return "\n".join(lines) + "\n"


def _normalize_molecule(molecule: Any) -> dict:
    if not isinstance(molecule, dict):
        raise ValueError("Each molecule entry must be an object.")

    normalized = dict(molecule)
    molecule_type = str(normalized.get("type", "")).strip().lower()
    molecule_id = str(normalized.get("id", "")).strip()
    if molecule_type not in ALLOWED_MOLECULE_TYPES:
        raise ValueError(f"Unsupported molecule type `{molecule_type}`.")
    if not molecule_id:
        raise ValueError("Each molecule requires a non-empty `id`.")

    normalized["type"] = molecule_type
    normalized["id"] = molecule_id

    if molecule_type in {"protein", "dna", "rna"}:
        normalized["sequence"] = _clean_sequence(normalized.get("sequence", ""), molecule_type)
    elif molecule_type == "ligand":
        ccd_code = str(normalized.get("ccd_code", "")).strip()
        smiles = str(normalized.get("smiles", "")).strip()
        if not ccd_code and not smiles:
            raise ValueError("Ligand molecules require either `ccd_code` or `smiles`.")
        if ccd_code:
            normalized["ccd_code"] = ccd_code
        if smiles:
            normalized["smiles"] = smiles

    if "msa" in normalized and normalized["msa"] is not None:
        if molecule_type not in {"protein", "rna"}:
            raise ValueError("Only protein and RNA molecules may provide `msa`.")
        normalized["msa"] = _normalize_msa_payload(normalized["msa"])

    return normalized


def normalize_args(args: dict) -> dict:
    normalized = dict(args)

    # Backwards-compatible legacy shape.
    legacy_sequence = normalized.get("sequence")
    if legacy_sequence and not normalized.get("inputs"):
        request_id = str(normalized.get("request_id", "query_1")).strip() or "query_1"
        input_id = str(normalized.get("input_id", request_id)).strip() or request_id
        input_payload = {
            "input_id": input_id,
            "molecules": [
                {
                    "type": "protein",
                    "id": "A",
                    "sequence": legacy_sequence,
                }
            ],
            "output_format": "pdb",
        }
        normalized = {
            "request_id": request_id,
            "inputs": [input_payload],
            "session_id": normalized.get("session_id", ""),
            "run_msa": bool(normalized.get("run_msa", False)),
        }

    request_id = str(normalized.get("request_id", "")).strip()
    inputs = normalized.get("inputs")
    if not request_id:
        raise ValueError("OpenFold3 requires `request_id`.")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("OpenFold3 requires `inputs` as a non-empty list.")

    normalized_inputs = []
    for entry in inputs:
        if not isinstance(entry, dict):
            raise ValueError("Each OpenFold3 input must be an object.")
        input_id = str(entry.get("input_id", "")).strip()
        molecules = entry.get("molecules")
        output_format = str(entry.get("output_format", "mmcif")).strip().lower() or "mmcif"
        if not input_id:
            raise ValueError("Each OpenFold3 input requires `input_id`.")
        if not isinstance(molecules, list) or not molecules:
            raise ValueError("Each OpenFold3 input requires a non-empty `molecules` list.")
        if output_format not in ALLOWED_OUTPUT_FORMATS:
            raise ValueError("OpenFold3 `output_format` must be `pdb` or `mmcif`.")

        normalized_inputs.append(
            {
                "input_id": input_id,
                "molecules": [_normalize_molecule(molecule) for molecule in molecules],
                "output_format": output_format,
            }
        )

    return {
        "request_id": request_id,
        "inputs": normalized_inputs,
        "session_id": str(normalized.get("session_id", "")),
        "run_msa": bool(normalized.get("run_msa", False)),
    }


def _count_residues(inputs: list[dict]) -> int:
    total = 0
    for entry in inputs:
        for molecule in entry["molecules"]:
            if molecule["type"] in {"protein", "dna", "rna"}:
                total += len(molecule["sequence"])
    return total


def _needs_remote_msa(inputs: list[dict], run_msa: bool) -> bool:
    if run_msa:
        return True
    for entry in inputs:
        for molecule in entry["molecules"]:
            if molecule["type"] in {"protein", "rna"} and not molecule.get("msa"):
                return True
    return False


def _write_inline_msa_files(
    msa_root: str, query_id: str, chain_id: str, sequence: str, msa_payload: dict
) -> list[str]:
    chain_dir = os.path.join(msa_root, query_id, chain_id)
    os.makedirs(chain_dir, exist_ok=True)

    paths = []
    for db_name, db_payload in msa_payload.items():
        for fmt_name, fmt_payload in db_payload.items():
            file_format = fmt_name
            file_content = fmt_payload["alignment"]
            if fmt_name == "csv":
                file_format = "a3m"
                file_content = _csv_alignment_to_a3m(fmt_payload["alignment"], sequence)

            msa_path = os.path.join(chain_dir, f"{db_name}.{file_format}")
            with open(msa_path, "w", encoding="utf-8") as fh:
                fh.write(file_content)
            paths.append(msa_path)
    return paths


def _as_path_or_list(paths: list[str]) -> str | list[str]:
    if len(paths) == 1:
        return paths[0]
    return paths


def _collect_msa_database_names(inputs: list[dict]) -> list[str]:
    names: list[str] = []
    for entry in inputs:
        for molecule in entry["molecules"]:
            for db_name in (molecule.get("msa") or {}).keys():
                if db_name not in names:
                    names.append(db_name)
    return names


def _get_structure_format(inputs: list[dict]) -> str:
    formats = {entry.get("output_format", "mmcif") for entry in inputs}
    if len(formats) != 1:
        raise ValueError("OpenFold3 requires a single output_format across all inputs.")
    output_format = formats.pop()
    return "pdb" if output_format == "pdb" else "cif"


def _build_runner_yaml(inputs: list[dict], use_low_mem: bool) -> str:
    lines = [
        "model_update:",
        "  presets:",
        "    - predict",
    ]
    if use_low_mem:
        lines.append("    - low_mem")
    lines.append("    - pae_enabled")

    msa_names = _collect_msa_database_names(inputs)
    if msa_names:
        lines.extend(
            [
                "dataset_config_kwargs:",
                "  msa:",
                "    max_seq_counts:",
            ]
        )
        for name in msa_names:
            lines.append(f"      {json.dumps(name)}: 16384")
        lines.append("    aln_order:")
        for name in msa_names:
            lines.append(f"      - {json.dumps(name)}")

    lines.extend(
        [
            "output_writer_settings:",
            f"  structure_format: {_get_structure_format(inputs)}",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_query_json(inputs: list[dict], msa_root: str, run_msa: bool) -> dict:
    queries = {}

    for entry in inputs:
        query_id = entry["input_id"]
        if query_id in queries:
            raise ValueError(f"Duplicate OpenFold3 input_id `{query_id}`.")

        chains = []
        for molecule in entry["molecules"]:
            molecule_type = molecule["type"]
            chain = {
                "molecule_type": molecule_type,
                "chain_ids": molecule["id"],
            }

            if molecule_type in {"protein", "dna", "rna"}:
                chain["sequence"] = molecule["sequence"]

            if molecule_type in {"protein", "rna"}:
                msa_paths = []
                if molecule.get("msa"):
                    msa_paths = _write_inline_msa_files(
                        msa_root,
                        query_id,
                        molecule["id"],
                        molecule["sequence"],
                        molecule["msa"],
                    )

                if msa_paths:
                    chain["main_msa_file_paths"] = _as_path_or_list(msa_paths)

            elif molecule_type == "ligand":
                if molecule.get("ccd_code"):
                    chain["ccd_codes"] = [molecule["ccd_code"]]
                if molecule.get("smiles"):
                    chain["smiles"] = molecule["smiles"]

            chains.append(chain)

        queries[query_id] = {"chains": chains}

    return {"queries": queries}


def run(request_id: str = "", inputs: list | None = None, sequence: str = "", run_msa: bool = False, session_id: str = "", **kwargs) -> dict:
    try:
        normalized = normalize_args(
            {
                "request_id": request_id,
                "inputs": inputs,
                "sequence": sequence,
                "run_msa": run_msa,
                "session_id": session_id,
                **kwargs,
            }
        )
    except ValueError as exc:
        return {"summary": f"Error: {exc}", "error": "invalid_input"}

    request_id = normalized["request_id"]
    inputs = normalized["inputs"]
    session_id = normalized.get("session_id", "")
    run_msa = normalized.get("run_msa", False)

    t0 = time.time()
    residue_count = _count_residues(inputs)
    molecule_count = sum(len(entry["molecules"]) for entry in inputs)

    vram_before = _get_gpu_vram_mb()
    stop_event = threading.Event()
    vram_results = {"peak": vram_before}
    monitor = threading.Thread(target=_monitor_vram, args=(stop_event, vram_results), daemon=True)
    monitor.start()

    with tempfile.TemporaryDirectory() as tmpdir:
        query_json = os.path.join(tmpdir, "query.json")
        out_dir = os.path.join(tmpdir, "output")
        msa_root = os.path.join(tmpdir, "msas")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(msa_root, exist_ok=True)
        os.makedirs("/root/.triton/autotune", exist_ok=True)

        try:
            query = _build_query_json(inputs, msa_root, run_msa)
        except ValueError as exc:
            stop_event.set()
            monitor.join(timeout=2)
            return {"summary": f"Error: {exc}", "error": "invalid_input"}
        with open(query_json, "w", encoding="utf-8") as fh:
            json.dump(query, fh)

        ckpt_dir = os.environ.get("OPENFOLD3_CACHE", "/root/.openfold3")
        ckpt_file = os.path.join(ckpt_dir, "of3_ft3_v1.pt")
        if not os.path.isfile(ckpt_file):
            os.makedirs(ckpt_dir, exist_ok=True)
            try:
                subprocess.run(
                    [
                        "aws",
                        "s3",
                        "cp",
                        "s3://openfold/openfold3_params/of3_ft3_v1.pt",
                        ckpt_dir + "/",
                        "--no-sign-request",
                    ],
                    check=True,
                    timeout=600,
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                dl_script = "/opt/openfold3/openfold3/scripts/download_openfold3_params.sh"
                if os.path.isfile(dl_script):
                    subprocess.run(["bash", dl_script, f"--download_dir={ckpt_dir}"], check=True, timeout=600)
                else:
                    stop_event.set()
                    monitor.join(timeout=2)
                    return {
                        "summary": "Error: Cannot download OpenFold3 weights. Install awscli.",
                        "error": "weight_download_failed",
                    }

        use_low_mem = bool(os.environ.get("OPENFOLD3_LOW_MEM", ""))
        runner_yaml_path = os.path.join(tmpdir, "runner.yaml")
        with open(runner_yaml_path, "w", encoding="utf-8") as yf:
            yf.write(_build_runner_yaml(inputs, use_low_mem))

        cmd = [
            "run_openfold",
            "predict",
            f"--query_json={query_json}",
            f"--output_dir={out_dir}",
            f"--inference_ckpt_path={ckpt_file}",
            f"--runner_yaml={runner_yaml_path}",
            "--num_diffusion_samples=1",
            "--num_model_seeds=1",
        ]
        cmd.extend(["--use_msa_server", "True" if _needs_remote_msa(inputs, run_msa) else "False"])

        t_inference = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            stop_event.set()
            monitor.join(timeout=2)
            return {"summary": "Error: OpenFold3 timed out after 900s.", "error": "timeout"}
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
            error_tail = (result.stderr or result.stdout or "")[-4000:]
            return {
                "summary": f"Error: OpenFold3 failed: {error_tail}",
                "error": error_tail,
                "metrics": metrics,
            }

        structure_content = ""
        confidence = 0.0
        output_files = []
        summary_text = ""
        for root, _, files in os.walk(out_dir):
            for filename in files:
                rel_path = os.path.relpath(os.path.join(root, filename), out_dir)
                output_files.append(rel_path)
                full_path = os.path.join(root, filename)
                if filename.endswith((".cif", ".pdb")) and not structure_content:
                    with open(full_path, encoding="utf-8") as fh:
                        structure_content = fh.read()
                elif filename.endswith(".json") and ("score" in filename.lower() or "metric" in filename.lower()):
                    try:
                        with open(full_path, encoding="utf-8") as fh:
                            scores = json.load(fh)
                        if isinstance(scores, dict):
                            confidence = float(scores.get("ptm", scores.get("plddt", scores.get("confidence", 0))))
                    except Exception:
                        pass
                elif filename == "summary.txt":
                    try:
                        with open(full_path, encoding="utf-8") as fh:
                            summary_text = fh.read().strip()
                    except Exception:
                        pass

        if not structure_content:
            extra_summary = f" Summary: {summary_text}" if summary_text else ""
            log_tail = (result.stderr or result.stdout or "")[-4000:].strip()
            extra_logs = f" Logs: {log_tail}" if log_tail else ""
            return {
                "summary": f"OpenFold3 ran but no structure found. Files: {output_files}.{extra_summary}{extra_logs}",
                "error": "no_output",
                "metrics": metrics,
            }

        if session_id:
            workspace_dir = f"/vol/workspace/{session_id}"
            os.makedirs(workspace_dir, exist_ok=True)
            with open(f"{workspace_dir}/predicted_structure.pdb", "w", encoding="utf-8") as fh:
                fh.write(structure_content)

        return {
            "summary": (
                f"OpenFold3 prediction for request {request_id}: {len(inputs)} input(s), "
                f"{molecule_count} molecules, {residue_count} polymer residues. Confidence: {confidence:.2f}."
            ),
            "pdb_content": structure_content,
            "confidence": confidence,
            "num_residues": residue_count,
            "num_molecules": molecule_count,
            "request_id": request_id,
            "metrics": metrics,
        }
