"""Utilities for saving large structure outputs as local artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

PDB_PREVIEW_CHARS = 1000


def _artifact_root() -> Path:
    """Resolve the local directory where CLI-visible artifacts should live."""
    try:
        from ct.agent.config import Config

        configured = Config.load().get("sandbox.output_dir")
        if configured:
            root = Path(str(configured)).expanduser()
            root.mkdir(parents=True, exist_ok=True)
            return root
    except Exception:
        pass

    root = Path.cwd() / "outputs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _slugify(value: str, *, fallback: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", (value or "").strip()).strip("._-")
    return slug or fallback


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def _preview(text: str, max_chars: int = PDB_PREVIEW_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated, {len(text)} chars total]"


def save_structure_artifact(
    pdb_content: str,
    *,
    tool_name: str,
    result_id: str = "",
    max_preview_chars: int = PDB_PREVIEW_CHARS,
) -> dict[str, Any]:
    """Persist structure text locally and return path/preview metadata."""
    structures_dir = _artifact_root() / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)

    tool_slug = _slugify(tool_name.split(".")[-1], fallback="structure")
    suffix = _slugify(result_id, fallback="artifact") if result_id else "artifact"
    artifact_path = structures_dir / f"{tool_slug}_{suffix}.pdb"
    artifact_path.write_text(pdb_content, encoding="utf-8")

    return {
        "pdb_path": _display_path(artifact_path),
        "pdb_preview": _preview(pdb_content, max_chars=max_preview_chars),
        "pdb_chars": len(pdb_content),
    }


def normalize_structure_result(
    result: dict[str, Any],
    *,
    tool_name: str,
    result_id: str = "",
    max_preview_chars: int = PDB_PREVIEW_CHARS,
) -> dict[str, Any]:
    """Persist large structure text locally and return an agent-friendly payload."""
    pdb_content = result.get("pdb_content")
    if not isinstance(pdb_content, str) or not pdb_content.strip():
        return result

    normalized = dict(result)
    normalized.pop("pdb_content", None)
    normalized.update(
        save_structure_artifact(
            pdb_content,
            tool_name=tool_name,
            result_id=result_id,
            max_preview_chars=max_preview_chars,
        )
    )

    summary = str(normalized.get("summary", "") or "").strip()
    location_note = f"Full PDB saved to {normalized['pdb_path']}."
    if location_note not in summary:
        normalized["summary"] = f"{summary} {location_note}".strip()

    return normalized
