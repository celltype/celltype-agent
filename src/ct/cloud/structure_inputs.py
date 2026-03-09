"""Helpers for structure-file inputs shared by cloud and local runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any


INLINE_FILE_ARG_NAMES = {
    "target_pdb",
    "backbone_pdb",
    "protein_pdb",
}

STRUCTURE_FILE_SUFFIXES = {".pdb", ".cif", ".mmcif", ".ent"}


def _looks_like_structure_text(value: str) -> bool:
    stripped = value.lstrip()
    if not stripped:
        return False
    if "\n" in value:
        return True
    return stripped.startswith(("ATOM", "HETATM", "MODEL", "HEADER", "data_"))


def inline_structure_file_args(tool_name: str, tool_args: dict[str, Any], logger=None) -> dict[str, Any]:
    """Replace supported local structure-file paths with inline file contents."""
    prepared = dict(tool_args)

    for arg_name in INLINE_FILE_ARG_NAMES:
        value = prepared.get(arg_name)
        if not isinstance(value, str) or _looks_like_structure_text(value):
            continue

        path = Path(value).expanduser()
        if not path.is_file() or path.suffix.lower() not in STRUCTURE_FILE_SUFFIXES:
            continue

        prepared[arg_name] = path.read_text(encoding="utf-8", errors="replace")
        if logger is not None:
            logger.info(
                "Inlined local structure file for %s (%s=%s)",
                tool_name,
                arg_name,
                path,
            )

    return prepared
