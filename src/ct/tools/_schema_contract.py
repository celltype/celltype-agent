"""Shared schema and error helpers for structured GPU tools."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ToolInputModel(BaseModel):
    """Base input model for structured GPU tools."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ToolOutputModel(BaseModel):
    """Base output model for structured GPU tools."""

    model_config = ConfigDict(extra="forbid")


class ToolErrorModel(BaseModel):
    """Structured tool error payload for agent-friendly failures."""

    summary: str
    error: str
    isError: bool = True
    detail: str | None = None

    model_config = ConfigDict(extra="allow")


def structured_error(summary: str, error: str, **extra: Any) -> dict[str, Any]:
    payload = ToolErrorModel(summary=summary, error=error, **extra)
    return payload.model_dump(exclude_none=True)


def export_tool_contract(input_model: type[BaseModel], output_model: type[BaseModel]) -> dict[str, Any]:
    """Expose JSON Schema for the current tool contract."""

    return {
        "input_schema": input_model.model_json_schema(),
        "output_schema": output_model.model_json_schema(),
    }
