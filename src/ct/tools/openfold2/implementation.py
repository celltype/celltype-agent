"""OpenFold2 structure prediction."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

try:
    from ct.tools._schema_contract import (
        ToolInputModel,
        ToolOutputModel,
        export_tool_contract,
        structured_error,
    )
except ModuleNotFoundError:
    class ToolInputModel(BaseModel):
        pass

    class ToolOutputModel(BaseModel):
        pass

    def export_tool_contract(input_model: type[BaseModel], output_model: type[BaseModel]) -> dict[str, Any]:
        return {
            "input_schema": input_model.model_json_schema(),
            "output_schema": output_model.model_json_schema(),
        }

    def structured_error(summary: str, error: str, **extra: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {"summary": summary, "error": error, "isError": True}
        payload.update(extra)
        return payload

OPENFOLD_DIR = "/opt/openfold"
AA_SEQUENCE_PATTERN = re.compile(r"^[ARNDCQEGHILKMFPSTWYV]+$")
MAX_SEQUENCE_LENGTH = 1000
ALLOWED_MODEL_IDS = {1, 2, 3, 4, 5}
ALIGNMENT_FILE_NAMES = {
    # Match the filenames expected by upstream OpenFold's monomer
    # precomputed-alignment data pipeline.
    "uniref90": "uniref90_hits.sto",
    "small_bfd": "bfd_uniclust_hits.a3m",
    "bfd_uniclust": "bfd_uniclust_hits.a3m",
    "mgnify": "mgnify_hits.sto",
    "uniprot": "uniprot_hits.sto",
    "pdb70": "hhsearch_output.hhr",
}


class A3MAlignment(ToolInputModel):
    alignment: str = Field(..., min_length=1, description="Raw A3M alignment text.")
    format: Literal["a3m"] = Field(
        "a3m",
        description="Alignment payload format. Must be `a3m` for A3M alignments.",
    )


class HHRAlignment(ToolInputModel):
    alignment: str = Field(..., min_length=1, description="Raw HHR template-search text.")
    format: Literal["hhr"] = Field(
        "hhr",
        description="Alignment payload format. Must be `hhr` for HHR alignments.",
    )


class AlignmentEntry(ToolInputModel):
    a3m: A3MAlignment | None = None
    hhr: HHRAlignment | None = None

    @model_validator(mode="after")
    def validate_non_empty(self) -> "AlignmentEntry":
        if self.a3m is None and self.hhr is None:
            raise ValueError("Alignment entries must include at least one supported alignment payload.")
        return self


class TemplateEntry(ToolInputModel):
    name: str = Field(
        ...,
        min_length=1,
        description="Template filename for a provided mmCIF template file.",
    )
    mmcif: str = Field(
        ...,
        min_length=1,
        description="Raw mmCIF template content.",
    )


class ToolInput(ToolInputModel):
    sequence: str = Field(
        ...,
        description="Single protein amino acid sequence in uppercase IUPAC notation.",
        min_length=1,
        max_length=MAX_SEQUENCE_LENGTH,
    )
    alignments: dict[str, AlignmentEntry] = Field(
        default_factory=dict,
        description="Optional precomputed alignments keyed by database name such as `uniref90` or `pdb70`.",
    )
    templates: list[TemplateEntry] = Field(
        default_factory=list,
        description="Optional raw mmCIF templates paired with precomputed template hits.",
    )
    selected_models: list[int] = Field(
        default_factory=lambda: [1],
        description="OpenFold model parameter sets to run. Predictions are returned ordered by confidence.",
    )
    relax_prediction: bool = Field(
        False,
        description="Whether to run the relaxation step after structure prediction.",
    )
    session_id: str = Field(
        "",
        description="Optional workspace session identifier used to persist output artifacts.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_payload(cls, payload: Any) -> Any:
        if not isinstance(payload, dict):
            raise ValueError("OpenFold2 inputs must be provided as an object.")
        if "sequences" in payload:
            raise ValueError(
                "OpenFold2 accepts only `sequence`. Use `structure.alphafold2_multimer` for multi-chain inputs."
            )

        normalized = dict(payload)
        normalized["sequence"] = str(normalized.get("sequence", "")).strip().upper()

        alignments = normalized.get("alignments") or {}
        if alignments and not isinstance(alignments, dict):
            raise ValueError("OpenFold2 `alignments` must be an object keyed by database name.")
        normalized["alignments"] = {
            str(db_name).strip().lower(): entry for db_name, entry in alignments.items()
        }

        templates = normalized.get("templates") or []
        if templates and not isinstance(templates, list):
            raise ValueError("OpenFold2 `templates` must be a list of mmCIF template payloads.")
        normalized["templates"] = templates
        return normalized

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, value: str) -> str:
        if not value:
            raise ValueError("OpenFold2 requires `sequence` as a non-empty amino acid sequence string.")
        if len(value) > MAX_SEQUENCE_LENGTH:
            raise ValueError(f"OpenFold2 `sequence` must be at most {MAX_SEQUENCE_LENGTH} residues.")
        if not AA_SEQUENCE_PATTERN.fullmatch(value):
            raise ValueError("OpenFold2 `sequence` must contain only valid amino acid IUPAC symbols.")
        return value

    @field_validator("selected_models")
    @classmethod
    def validate_selected_models(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("OpenFold2 `selected_models` must be a non-empty list of model ids.")
        clean_models: list[int] = []
        for model_id in value:
            model_int = int(model_id)
            if model_int not in ALLOWED_MODEL_IDS:
                raise ValueError("OpenFold2 `selected_models` must contain values from 1 to 5.")
            if model_int not in clean_models:
                clean_models.append(model_int)
        return clean_models


class ToolPrediction(ToolOutputModel):
    model: str = Field(..., description="Model preset used for the prediction.")
    confidence: float = Field(..., description="Average per-residue confidence for this prediction.")
    pdb_content: str = Field(..., description="Predicted structure in PDB format.")


class ToolMetrics(ToolOutputModel):
    vram_before_mb: int = Field(..., description="GPU VRAM used before inference.")
    vram_after_mb: int = Field(..., description="GPU VRAM used after inference.")
    vram_peak_mb: int = Field(..., description="Peak GPU VRAM observed during inference.")
    time_inference_s: float = Field(..., description="Wall-clock time spent inside model inference.")
    time_total_s: float = Field(..., description="Total wall-clock time for the full tool execution.")
    weights_cache_hit: bool = Field(
        ...,
        description="Whether all requested OpenFold weight files were already present in the persistent cache.",
    )
    weights_cache_path: str = Field(..., description="Persistent cache path used for OpenFold parameter files.")
    weights_cache_details: dict[str, bool] = Field(
        default_factory=dict,
        description="Per-model cache hit information keyed by OpenFold model preset.",
    )


class ToolOutput(ToolOutputModel):
    summary: str = Field(..., description="Human-readable summary of the OpenFold2 run.")
    pdb_content: str = Field(..., description="Best predicted structure in PDB format.")
    confidence: float = Field(..., description="Confidence score for the best prediction.")
    num_residues: int = Field(..., description="Number of residues in the input sequence.")
    predictions: list[ToolPrediction] = Field(
        default_factory=list,
        description="Predictions for each requested OpenFold model, ordered by confidence.",
    )
    metrics: ToolMetrics = Field(..., description="Execution metrics and cache information.")


AlignmentEntry.model_rebuild()
TemplateEntry.model_rebuild()
ToolInput.model_rebuild()
ToolPrediction.model_rebuild()
ToolMetrics.model_rebuild()
ToolOutput.model_rebuild()


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


def _normalize_alignment_entry(db_name: str, payload: Any) -> dict:
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Alignment entries must be objects keyed by format.")
    normalized = {}
    for fmt_name, fmt_payload in payload.items():
        if fmt_name not in {"a3m", "hhr"}:
            raise ValueError(f"Unsupported alignment format `{fmt_name}` for `{db_name}`.")
        if not isinstance(fmt_payload, dict):
            raise ValueError("Alignment format payloads must be objects.")
        alignment = str(fmt_payload.get("alignment", "")).strip()
        declared_format = str(fmt_payload.get("format", "")).strip().lower()
        if not alignment:
            raise ValueError("Alignment payloads require non-empty `alignment` text.")
        if declared_format != fmt_name:
            raise ValueError(f"Alignment payload format mismatch for `{db_name}` / `{fmt_name}`.")
        normalized[fmt_name] = {"alignment": alignment, "format": declared_format}
    return normalized


def normalize_args(args: dict) -> dict:
    try:
        model = ToolInput.model_validate(dict(args))
    except ValidationError as exc:
        raise ValueError(str(exc.errors()[0]["msg"])) from exc
    normalized = model.model_dump(exclude_none=True)
    normalized["alignments"] = {
        db_name: _normalize_alignment_entry(db_name, payload)
        for db_name, payload in normalized["alignments"].items()
    }
    return normalized


def _ensure_params_file(model_name: str) -> tuple[str, bool]:
    params_dir = os.environ.get("OPENFOLD_PARAMS_DIR", "/root/.cache/openfold/params")
    params_file = os.path.join(params_dir, f"params_{model_name}.npz")
    cache_hit = os.path.isfile(params_file)
    if not cache_hit:
        os.makedirs(params_dir, exist_ok=True)
        dl_script = os.path.join(OPENFOLD_DIR, "scripts", "download_alphafold_params.sh")
        parent = os.path.dirname(params_dir)
        subprocess.run(["bash", dl_script, parent], check=True, timeout=600)
    return params_file, cache_hit


def _write_alignments(align_dir: str, alignments: dict) -> None:
    os.makedirs(align_dir, exist_ok=True)
    for db_name, payload in alignments.items():
        for fmt_name, fmt_payload in payload.items():
            default_name = ALIGNMENT_FILE_NAMES.get(db_name)
            if default_name is None:
                default_name = f"{db_name}_hits.{fmt_name}"
            elif not default_name.endswith(f".{fmt_name}"):
                default_name = f"{db_name}_hits.{fmt_name}"
            with open(os.path.join(align_dir, default_name), "w", encoding="utf-8") as fh:
                fh.write(fmt_payload["alignment"])


def _write_templates(template_dir: str, templates: list[dict[str, str]]) -> None:
    os.makedirs(template_dir, exist_ok=True)
    for index, template in enumerate(templates, start=1):
        name = str(template.get("name", "")).strip() or f"template_{index}.cif"
        safe_name = os.path.basename(name)
        if not safe_name.endswith(".cif"):
            safe_name = f"{safe_name}.cif"
        with open(os.path.join(template_dir, safe_name), "w", encoding="utf-8") as fh:
            fh.write(str(template.get("mmcif", "")))


def _should_use_custom_template(alignments: dict[str, dict], templates: list[dict[str, str]]) -> bool:
    if not templates:
        return False
    pdb70_alignment = alignments.get("pdb70") or {}
    return "hhr" not in pdb70_alignment


def _extract_prediction(out_dir: str) -> tuple[str, float]:
    pdb_content = ""
    confidence = 0.0
    for root, _, files in os.walk(out_dir):
        for filename in sorted(files):
            if filename.endswith(".pdb"):
                with open(os.path.join(root, filename), encoding="utf-8") as fh:
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
                return pdb_content, confidence
    return "", 0.0


def get_contract() -> dict[str, Any]:
    """Expose structured input/output schemas for UI scaffolding and tests."""

    return export_tool_contract(ToolInput, ToolOutput)


def run(sequence: str = "", session_id: str = "", **kwargs) -> dict:
    try:
        normalized = normalize_args({"sequence": sequence, "session_id": session_id, **kwargs})
    except ValueError as exc:
        return structured_error(
            summary=f"Error: {exc}",
            error="invalid_sequence_input",
            detail=str(exc),
        )

    sequence = normalized["sequence"]
    alignments = normalized["alignments"]
    templates = normalized["templates"]
    use_custom_template = _should_use_custom_template(alignments, templates)
    selected_models = normalized["selected_models"]
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

        template_dir = os.path.join(tmpdir, "templates")
        os.makedirs(template_dir)
        if templates:
            _write_templates(template_dir, templates)
        else:
            with open(os.path.join(template_dir, "dummy.cif"), "w", encoding="utf-8") as fh:
                fh.write("data_dummy\n")

        align_root = os.path.join(tmpdir, "alignments")
        query_align_dir = os.path.join(align_root, "query")
        os.makedirs(query_align_dir)
        if alignments:
            _write_alignments(query_align_dir, alignments)

        predictions = []
        cache_details: dict[str, bool] = {}
        t_inference = time.time()
        for model_id in selected_models:
            out_dir = os.path.join(tmpdir, f"output_model_{model_id}")
            os.makedirs(out_dir, exist_ok=True)
            model_name = f"model_{model_id}"
            params_file, cache_hit = _ensure_params_file(model_name)
            cache_details[model_name] = cache_hit
            cmd = [
                sys.executable,
                os.path.join(OPENFOLD_DIR, "run_pretrained_openfold.py"),
                fasta_dir,
                template_dir,
                "--output_dir",
                out_dir,
                "--model_device",
                "cuda:0",
                "--config_preset",
                model_name,
                "--jax_param_path",
                params_file,
            ]
            cmd.extend(["--use_precomputed_alignments", align_root])
            if use_custom_template:
                cmd.append("--use_custom_template")
            if not alignments:
                cmd.append("--use_single_seq_mode")
            if not relax_prediction:
                cmd.append("--skip_relaxation")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=OPENFOLD_DIR)
            if result.returncode != 0:
                stop_event.set()
                monitor.join(timeout=2)
                error_tail = f"STDERR:\n{result.stderr[-2000:]}\nSTDOUT:\n{result.stdout[-2000:]}"
                return structured_error(
                    summary=f"Error: OpenFold2 failed on {model_name}: {error_tail}",
                    error="openfold2_runtime_error",
                    detail=error_tail,
                    command=cmd,
                    metrics={
                        "vram_before_mb": vram_before,
                        "vram_peak_mb": vram_results["peak"],
                        "time_total_s": round(time.time() - t0, 2),
                        "weights_cache_hit": cache_hit,
                        "weights_cache_path": os.environ.get(
                            "OPENFOLD_PARAMS_DIR", "/root/.cache/openfold/params"
                        ),
                        "weights_cache_details": cache_details,
                    },
                )

            pdb_content, confidence = _extract_prediction(out_dir)
            if not pdb_content:
                stop_event.set()
                monitor.join(timeout=2)
                return structured_error(
                    summary=f"OpenFold2 ran but produced no structure for {model_name}.",
                    error="no_output",
                    metrics={
                        "weights_cache_hit": cache_hit,
                        "weights_cache_path": os.environ.get(
                            "OPENFOLD_PARAMS_DIR", "/root/.cache/openfold/params"
                        ),
                        "weights_cache_details": cache_details,
                    },
                )
            predictions.append(
                ToolPrediction(
                    model=model_name,
                    confidence=confidence,
                    pdb_content=pdb_content,
                )
            )

        t_inference = time.time() - t_inference
        stop_event.set()
        monitor.join(timeout=2)
        vram_after = _get_gpu_vram_mb()
        vram_peak = vram_results["peak"]

        predictions.sort(key=lambda item: item.confidence, reverse=True)
        best = predictions[0]

        if session_id:
            workspace_dir = f"/vol/workspace/{session_id}"
            os.makedirs(workspace_dir, exist_ok=True)
            with open(f"{workspace_dir}/predicted_structure.pdb", "w", encoding="utf-8") as fh:
                fh.write(best.pdb_content)

        output = ToolOutput(
            summary=(
                f"OpenFold2 prediction for {seq_len}-residue protein using {len(predictions)} model(s). "
                f"Best pLDDT: {best.confidence:.1f}/100."
            ),
            pdb_content=best.pdb_content,
            confidence=best.confidence,
            num_residues=seq_len,
            predictions=predictions,
            metrics=ToolMetrics(
                vram_before_mb=vram_before,
                vram_after_mb=vram_after,
                vram_peak_mb=vram_peak,
                time_inference_s=round(t_inference, 2),
                time_total_s=round(time.time() - t0, 2),
                weights_cache_hit=all(cache_details.values()) if cache_details else False,
                weights_cache_path=os.environ.get(
                    "OPENFOLD_PARAMS_DIR", "/root/.cache/openfold/params"
                ),
                weights_cache_details=cache_details,
            ),
        )
        return output.model_dump()
