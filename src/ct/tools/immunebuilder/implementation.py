"""ImmuneBuilder structure prediction for antibodies, nanobodies, and TCRs."""

from __future__ import annotations

from pathlib import Path
import re
import tempfile
import time
import traceback
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

try:
    from ct.tools._schema_contract import (
        ToolInputModel,
        ToolOutputModel,
        export_tool_contract,
        structured_error,
    )
except ModuleNotFoundError:
    class ToolInputModel(BaseModel):
        model_config = ConfigDict(extra="forbid", populate_by_name=True)

    class ToolOutputModel(BaseModel):
        model_config = ConfigDict(extra="forbid")

    def export_tool_contract(
        input_model: type[BaseModel],
        output_model: type[BaseModel],
    ) -> dict[str, Any]:
        return {
            "input_schema": input_model.model_json_schema(),
            "output_schema": output_model.model_json_schema(),
        }

    def structured_error(summary: str, error: str, **extra: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {"summary": summary, "error": error, "isError": True}
        payload.update(extra)
        return payload


AA_SEQUENCE_PATTERN = re.compile(r"^[ARNDCQEGHILKMFPSTWYVBZXOU]+$")


def _clean_sequence(value: Any, field_name: str) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None
    if text.startswith(">"):
        raise ValueError(
            f"`{field_name}` does not accept FASTA input. "
            "Use the named chain fields with raw sequence text."
        )

    clean = "".join(text.split()).upper()
    if not clean:
        return None
    if not AA_SEQUENCE_PATTERN.fullmatch(clean):
        raise ValueError(f"`{field_name}` must contain only valid amino acid symbols.")
    return clean


class ToolInput(ToolInputModel):
    mode: Literal["antibody", "nanobody", "tcr"] = Field(
        ...,
        description="ImmuneBuilder prediction mode.",
    )
    heavy_chain: str | None = Field(
        None,
        description="Heavy-chain amino acid sequence. Required for `antibody` and `nanobody`.",
    )
    light_chain: str | None = Field(
        None,
        description="Light-chain amino acid sequence. Required for `antibody` only.",
    )
    alpha_chain: str | None = Field(
        None,
        description="TCR alpha-chain amino acid sequence. Required for `tcr` only.",
    )
    beta_chain: str | None = Field(
        None,
        description="TCR beta-chain amino acid sequence. Required for `tcr` only.",
    )
    session_id: str = Field(
        "",
        description="Optional workspace session identifier used to persist output artifacts.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_payload(cls, payload: Any) -> Any:
        if not isinstance(payload, dict):
            raise ValueError("ImmuneBuilder inputs must be provided as an object.")

        if "chains" in payload:
            raise ValueError(
                "ImmuneBuilder v1 does not accept `chains`. Use named chain fields instead."
            )
        if "sequence" in payload or "sequences" in payload or "fasta" in payload:
            raise ValueError(
                "ImmuneBuilder v1 does not accept generic sequence or FASTA inputs. "
                "Use named chain fields instead."
            )

        normalized = dict(payload)
        normalized["mode"] = str(normalized.get("mode", "")).strip().lower()
        normalized["session_id"] = str(normalized.get("session_id", "")).strip()
        return normalized

    @field_validator("heavy_chain", "light_chain", "alpha_chain", "beta_chain", mode="before")
    @classmethod
    def validate_chain(cls, value: Any, info: ValidationInfo) -> str | None:
        return _clean_sequence(value, info.field_name)

    @model_validator(mode="after")
    def validate_mode_fields(self) -> "ToolInput":
        if self.mode == "antibody":
            if not self.heavy_chain or not self.light_chain:
                raise ValueError("`antibody` mode requires both `heavy_chain` and `light_chain`.")
            if self.alpha_chain or self.beta_chain:
                raise ValueError("`antibody` mode does not accept `alpha_chain` or `beta_chain`.")
        elif self.mode == "nanobody":
            if not self.heavy_chain:
                raise ValueError("`nanobody` mode requires `heavy_chain`.")
            if self.light_chain or self.alpha_chain or self.beta_chain:
                raise ValueError("`nanobody` mode accepts only `heavy_chain`.")
        elif self.mode == "tcr":
            if not self.alpha_chain or not self.beta_chain:
                raise ValueError("`tcr` mode requires both `alpha_chain` and `beta_chain`.")
            if self.heavy_chain or self.light_chain:
                raise ValueError("`tcr` mode does not accept `heavy_chain` or `light_chain`.")
        return self


class ToolOutput(ToolOutputModel):
    summary: str = Field(..., description="Human-readable summary of the ImmuneBuilder run.")
    pdb_content: str = Field(..., description="Predicted structure in PDB format.")
    mode: Literal["antibody", "nanobody", "tcr"] = Field(..., description="Prediction mode used.")
    chain_labels: list[str] = Field(
        ...,
        description="ImmuneBuilder chain labels included in the prediction.",
    )
    num_chains: int = Field(..., description="Number of chains passed to the predictor.")
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional runtime metadata for the prediction.",
    )


ToolInput.model_rebuild()
ToolOutput.model_rebuild()


def normalize_args(args: dict[str, Any]) -> dict[str, Any]:
    try:
        model = ToolInput.model_validate(dict(args))
    except ValidationError as exc:
        raise ValueError(str(exc.errors()[0]["msg"])) from exc
    return model.model_dump(exclude_none=True)


def get_contract() -> dict[str, Any]:
    """Expose structured input/output schemas for tests and UI scaffolding."""

    return export_tool_contract(ToolInput, ToolOutput)


def _build_predictor_request(
    normalized: dict[str, Any],
) -> tuple[type[Any], dict[str, str], list[str]]:
    from ImmuneBuilder import ABodyBuilder2, NanoBodyBuilder2, TCRBuilder2

    mode = normalized["mode"]
    if mode == "antibody":
        return (
            ABodyBuilder2,
            {"H": normalized["heavy_chain"], "L": normalized["light_chain"]},
            ["H", "L"],
        )
    if mode == "nanobody":
        return (
            NanoBodyBuilder2,
            {"H": normalized["heavy_chain"]},
            ["H"],
        )
    return (
        TCRBuilder2,
        {"A": normalized["alpha_chain"], "B": normalized["beta_chain"]},
        ["A", "B"],
    )


def run(
    mode: str = "",
    heavy_chain: str = "",
    light_chain: str = "",
    alpha_chain: str = "",
    beta_chain: str = "",
    session_id: str = "",
    **kwargs,
) -> dict[str, Any]:
    try:
        normalized = normalize_args(
            {
                "mode": mode,
                "heavy_chain": heavy_chain,
                "light_chain": light_chain,
                "alpha_chain": alpha_chain,
                "beta_chain": beta_chain,
                "session_id": session_id,
                **kwargs,
            }
        )
    except ValueError as exc:
        return structured_error(
            summary=f"Error: {exc}",
            error="invalid_immunebuilder_input",
            detail=str(exc),
        )

    try:
        predictor_cls, sequences, chain_labels = _build_predictor_request(normalized)
    except Exception:
        return structured_error(
            summary="Error: ImmuneBuilder dependencies are not installed in this runtime.",
            error="immunebuilder_not_installed",
            detail=traceback.format_exc(),
        )

    t0 = time.time()

    try:
        predictor = predictor_cls()
        prediction = predictor.predict(sequences)

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            output_path = Path(tmp.name)

        try:
            prediction.save(str(output_path))
            pdb_content = output_path.read_text(encoding="utf-8")
        finally:
            if output_path.exists():
                output_path.unlink()

        if not pdb_content.strip():
            return structured_error(
                summary="ImmuneBuilder completed but produced an empty PDB output.",
                error="no_output",
            )

        session_id = normalized.get("session_id", "")
        if session_id:
            workspace_dir = Path(f"/vol/workspace/{session_id}")
            workspace_dir.mkdir(parents=True, exist_ok=True)
            (workspace_dir / "predicted_structure.pdb").write_text(pdb_content, encoding="utf-8")

        output = ToolOutput(
            summary=(
                f"ImmuneBuilder predicted a {normalized['mode']} structure "
                f"with {len(chain_labels)} chain(s)."
            ),
            pdb_content=pdb_content,
            mode=normalized["mode"],
            chain_labels=chain_labels,
            num_chains=len(chain_labels),
            metrics={
                "predictor": predictor_cls.__name__,
                "num_residues": sum(len(sequence) for sequence in sequences.values()),
                "time_total_s": round(time.time() - t0, 2),
            },
        )
        return output.model_dump()
    except Exception as exc:
        return structured_error(
            summary=f"Error: ImmuneBuilder failed: {exc}",
            error="immunebuilder_runtime_error",
            detail=traceback.format_exc(),
        )
