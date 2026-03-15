"""DeepImmuno container wrapper.

Wraps the upstream DeepImmuno-CNN CLI for:
- single peptide + HLA scoring
- batch CSV scoring

This tool is CPU-only and intended for immunogenicity triage workflows.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


DEEPIMMUNO_SCRIPT = Path("/opt/DeepImmuno/deepimmuno-cnn.py")
WORKSPACE_ROOT = Path("/vol/workspace")


def _clean_peptide(peptide: str) -> str:
    return (peptide or "").strip().upper().replace(" ", "").replace("\n", "")


def _clean_hla(hla: str) -> str:
    h = (hla or "").strip().upper()
    if not h:
        return h
    if not h.startswith("HLA-"):
        h = f"HLA-{h}"
    return h


def _extract_score(stdout: str) -> Optional[float]:
    """Best-effort extraction of a numeric immunogenicity score from CLI output."""
    if not stdout:
        return None

    patterns = [
        r"immunogenicity[^0-9\-+]*([-+]?\d+(?:\.\d+)?)",
        r"score[^0-9\-+]*([-+]?\d+(?:\.\d+)?)",
        r"([-+]?\d+\.\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, stdout, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                continue
    return None


def _copy_to_workspace(session_id: str, src_paths: list[Path]) -> list[str]:
    """Copy selected outputs into the shared workspace if a session id is provided."""
    if not session_id:
        return []

    out_dir = WORKSPACE_ROOT / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for src in src_paths:
        if src.exists():
            dest = out_dir / src.name
            if src.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(src, dest)
            else:
                shutil.copy2(src, dest)
            copied.append(str(dest))
    return copied


def _validate_inputs(peptide: str, hla: str, peptides_csv: str) -> Optional[dict]:
    if peptides_csv and (peptide or hla):
        return {
            "summary": "Provide either peptide/hla for single mode or peptides_csv for batch mode, not both.",
            "error": "ambiguous_input",
        }

    if not peptides_csv and not peptide:
        return {
            "summary": "Missing input. Provide peptide+hla for single mode or peptides_csv for batch mode.",
            "error": "missing_input",
        }

    if peptide and not hla:
        return {
            "summary": "Missing HLA allele for single-query mode.",
            "error": "missing_hla",
        }

    return None


def run(
    peptide: str = "",
    hla: str = "",
    peptides_csv: str = "",
    session_id: str = "",
    timeout_s: int = 900,
    **kwargs,
) -> dict:
    if not DEEPIMMUNO_SCRIPT.exists():
        return {
            "summary": "DeepImmuno script not found inside container.",
            "error": "missing_deepimmuno_script",
        }

    validation_error = _validate_inputs(peptide, hla, peptides_csv)
    if validation_error:
        return validation_error

    clean_peptide = _clean_peptide(peptide)
    clean_hla = _clean_hla(hla)

    warnings = []
    if clean_peptide and len(clean_peptide) not in (9, 10):
        warnings.append(
            "DeepImmuno is primarily intended for 9mer and 10mer peptides; score may be unreliable."
        )

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            generated_paths: list[Path] = []

            if peptides_csv:
                input_path = Path(peptides_csv)
                if not input_path.exists():
                    return {
                        "summary": f"Batch input file not found: {peptides_csv}",
                        "error": "missing_batch_file",
                    }

                out_dir = tmpdir_path / "deepimmuno_batch"
                out_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    "python",
                    str(DEEPIMMUNO_SCRIPT),
                    "--mode",
                    "multiple",
                    "--intdir",
                    str(input_path),
                    "--outdir",
                    str(out_dir),
                ]

                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=max(1, int(timeout_s)),
                )

                if proc.returncode != 0:
                    return {
                        "summary": f"DeepImmuno batch run failed: {proc.stderr[:500]}",
                        "error": proc.stderr,
                        "stdout": proc.stdout,
                    }

                generated_files = sorted(
                    str(p) for p in out_dir.rglob("*") if p.is_file()
                )
                generated_paths.append(out_dir)
                copied = _copy_to_workspace(session_id, generated_paths)

                return {
                    "summary": f"DeepImmuno batch scoring completed for {len(generated_files)} output file(s).",
                    "mode": "batch",
                    "input_file": str(input_path),
                    "generated_files": generated_files,
                    "workspace_outputs": copied,
                    "stdout": proc.stdout[-4000:],
                    "warnings": warnings,
                }

            cmd = [
                "python",
                str(DEEPIMMUNO_SCRIPT),
                "--mode",
                "single",
                "--epitope",
                clean_peptide,
                "--hla",
                clean_hla,
            ]

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=max(1, int(timeout_s)),
            )

            if proc.returncode != 0:
                return {
                    "summary": f"DeepImmuno single-query run failed: {proc.stderr[:500]}",
                    "error": proc.stderr,
                    "stdout": proc.stdout,
                    "mode": "single",
                    "peptide": clean_peptide,
                    "hla": clean_hla,
                }

            score = _extract_score(proc.stdout)

            raw_stdout_path = tmpdir_path / "deepimmuno_stdout.txt"
            raw_stdout_path.write_text(proc.stdout or "", encoding="utf-8")
            generated_paths.append(raw_stdout_path)
            copied = _copy_to_workspace(session_id, generated_paths)

            score_text = f"{score:.4f}" if score is not None else "unparsed"
            summary = (
                f"DeepImmuno scored peptide {clean_peptide} against {clean_hla}. "
                f"Predicted immunogenicity score: {score_text}."
            )

            result = {
                "summary": summary,
                "mode": "single",
                "peptide": clean_peptide,
                "hla": clean_hla,
                "score": score,
                "stdout": proc.stdout[-4000:],
                "workspace_outputs": copied,
            }
            if warnings:
                result["warnings"] = warnings
            return result

    except subprocess.TimeoutExpired:
        return {
            "summary": f"DeepImmuno timed out after {timeout_s}s.",
            "error": "timeout",
        }
    except Exception as exc:
        return {
            "summary": f"DeepImmuno failed: {exc}",
            "error": repr(exc),
        }
