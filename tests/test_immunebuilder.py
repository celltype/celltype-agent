"""Behavioral tests for the ImmuneBuilder container-tool implementation."""

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
import sys
import types

import pytest


TOOL_DIR = Path(__file__).resolve().parents[1] / "src/ct/tools/immunebuilder"
IMPLEMENTATION_PATH = TOOL_DIR / "implementation.py"

VALID_HEAVY_CHAIN = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAIYSGGSTYYADSVKGRFTISRDNSKNTLY"
    "LQMNSLRAEDTAVYYCAR"
)
VALID_LIGHT_CHAIN = (
    "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYDASSLESGVPSRFSGSGSGTDFTLTISSLQ"
    "PEDFATYYCQQYNSYPYTFGQGTKVEIK"
)
VALID_ALPHA_CHAIN = (
    "MNHVTQTPKFQVLKTGQSMTLQCAQDMNHNSMYWYRQDPGQGLRLIYYSAAADITDKGEVPNGYNVSRLKKQNFLLGLES"
    "AAPSQTSVYFCASSLGQGAEAFFGQGTRLTVV"
)
VALID_BETA_CHAIN = (
    "DTGITQSPKYLFRKEGQNVTLSCEQNLNHDAMYWYRQDPGQGLRLIHYSVGAGTTDQGEVPNGYNVSRSTTEDFPLRLLS"
    "AAPSQTSVYFCASSPGLAGNEKLFFGSGTQLSVL"
)
MOCK_PDB = (
    "HEADER MOCK\n"
    "ATOM      1  CA  ALA H   1       0.0   0.0   0.0  1.00 70.00           C\n"
    "END\n"
)


def _messy_sequence(sequence: str) -> str:
    midpoint = len(sequence) // 2
    return f"  {sequence[:midpoint].lower()} \n {sequence[midpoint:]}  "


def _error_blob(result: dict[str, object]) -> str:
    return f"{result.get('summary', '')} {result.get('detail', '')}".lower()


def _load_immunebuilder_module(*, block_shared_helper: bool = False):
    if not IMPLEMENTATION_PATH.exists():
        pytest.skip("ImmuneBuilder implementation is not present in this workspace yet.")

    spec = importlib.util.spec_from_file_location(
        "ct_test_immunebuilder_implementation",
        IMPLEMENTATION_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load module from {IMPLEMENTATION_PATH}")

    module = importlib.util.module_from_spec(spec)
    if not block_shared_helper:
        spec.loader.exec_module(module)
        return module

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ct.tools._schema_contract":
            raise ModuleNotFoundError(name)
        return original_import(name, globals, locals, fromlist, level)

    try:
        builtins.__import__ = guarded_import
        spec.loader.exec_module(module)
    finally:
        builtins.__import__ = original_import

    return module


def _make_fake_immunebuilder_module(
    *,
    pdb_text: str = MOCK_PDB,
    save_side_effect: Exception | None = None,
):
    class FakePrediction:
        saved_paths: list[str] = []

        def save(self, path: str):
            type(self).saved_paths.append(path)
            if save_side_effect is not None:
                raise save_side_effect
            Path(path).write_text(pdb_text, encoding="utf-8")

    def _builder_class(name: str):
        class FakeBuilder:
            last_sequences: dict[str, str] | None = None

            def predict(self, sequences):
                type(self).last_sequences = dict(sequences)
                return FakePrediction()

        FakeBuilder.__name__ = name
        return FakeBuilder

    builders = {
        "ABodyBuilder2": _builder_class("FakeABodyBuilder2"),
        "NanoBodyBuilder2": _builder_class("FakeNanoBodyBuilder2"),
        "TCRBuilder2": _builder_class("FakeTCRBuilder2"),
    }
    return types.SimpleNamespace(**builders), builders, FakePrediction


def _redirect_workspace_paths(tmp_path: Path):
    real_path = Path

    def fake_path(value):
        raw = str(value)
        if raw.startswith("/vol/workspace/"):
            relative = raw.removeprefix("/vol/workspace/").lstrip("/")
            return tmp_path / "vol" / "workspace" / relative
        return real_path(value)

    return fake_path


@pytest.fixture(scope="module")
def immunebuilder_module():
    return _load_immunebuilder_module()


class TestImmuneBuilderContract:
    def test_dockerfile_installs_pydantic(self):
        dockerfile = TOOL_DIR / "Dockerfile"
        if not dockerfile.exists():
            pytest.skip("ImmuneBuilder Dockerfile is not present in this workspace yet.")

        dockerfile_text = dockerfile.read_text(encoding="utf-8").lower()
        assert "pydantic" in dockerfile_text

    def test_contract_exposes_named_chain_fields(self, immunebuilder_module):
        assert hasattr(
            immunebuilder_module,
            "get_contract",
        ), "implementation.py must expose get_contract()"

        contract = immunebuilder_module.get_contract()
        assert isinstance(contract, dict)
        assert {"input_schema", "output_schema"} <= set(contract)

        input_schema = contract["input_schema"]
        output_schema = contract["output_schema"]

        assert input_schema["type"] == "object"
        assert input_schema.get("additionalProperties") is False

        input_properties = input_schema["properties"]
        assert {
            "mode",
            "heavy_chain",
            "light_chain",
            "alpha_chain",
            "beta_chain",
            "session_id",
        } <= set(input_properties)
        assert "chains" not in input_properties
        assert "sequence" not in input_properties
        assert set(input_properties["mode"].get("enum", [])) == {"antibody", "nanobody", "tcr"}
        assert "mode" in set(input_schema.get("required", []))

        assert output_schema["type"] == "object"
        output_properties = output_schema["properties"]
        required_output_fields = {
            "summary",
            "pdb_content",
            "mode",
            "chain_labels",
            "num_chains",
        }
        assert required_output_fields <= set(output_properties)
        assert required_output_fields <= set(output_schema.get("required", []))
        assert "metrics" in output_properties

    def test_fallback_contract_stays_strict_without_shared_helper(self):
        immunebuilder_module = _load_immunebuilder_module(block_shared_helper=True)

        contract = immunebuilder_module.get_contract()
        assert contract["input_schema"].get("additionalProperties") is False

        result = immunebuilder_module.run(
            mode="nanobody",
            heavy_chain=VALID_HEAVY_CHAIN,
            unexpected_field="typo",
        )

        assert result.get("error") == "invalid_immunebuilder_input"
        assert result.get("isError") is True
        detail_blob = _error_blob(result)
        assert "extra" in detail_blob or "permit" in detail_blob or "unexpected_field" in detail_blob


class TestImmuneBuilderInputValidation:
    def test_normalize_args_canonicalizes_mode_sequences_and_session_id(
        self,
        immunebuilder_module,
    ):
        normalized = immunebuilder_module.normalize_args(
            {
                "mode": "  ANTIBODY ",
                "heavy_chain": _messy_sequence(VALID_HEAVY_CHAIN),
                "light_chain": _messy_sequence(VALID_LIGHT_CHAIN),
                "session_id": "  run-123  ",
            }
        )

        assert normalized == {
            "mode": "antibody",
            "heavy_chain": VALID_HEAVY_CHAIN,
            "light_chain": VALID_LIGHT_CHAIN,
            "session_id": "run-123",
        }

    @pytest.mark.parametrize(
        ("extra_field", "extra_value", "expected_text"),
        [
            ("chains", {"H": VALID_HEAVY_CHAIN}, "chains"),
            ("sequence", VALID_HEAVY_CHAIN, "generic sequence"),
            ("sequences", [VALID_HEAVY_CHAIN], "generic sequence"),
            ("fasta", f">heavy\n{VALID_HEAVY_CHAIN}", "fasta"),
        ],
    )
    def test_run_rejects_legacy_sequence_inputs(
        self,
        immunebuilder_module,
        extra_field,
        extra_value,
        expected_text,
    ):
        result = immunebuilder_module.run(
            mode="nanobody",
            heavy_chain=VALID_HEAVY_CHAIN,
            **{extra_field: extra_value},
        )

        assert result.get("error") == "invalid_immunebuilder_input"
        assert result.get("isError") is True
        assert expected_text in _error_blob(result)

    @pytest.mark.parametrize(
        ("payload", "expected_text"),
        [
            ({"mode": "antibody", "heavy_chain": VALID_HEAVY_CHAIN}, "light_chain"),
            ({"mode": "nanobody"}, "heavy_chain"),
            ({"mode": "tcr", "alpha_chain": VALID_ALPHA_CHAIN}, "beta_chain"),
        ],
    )
    def test_run_rejects_missing_mode_specific_chains(
        self,
        immunebuilder_module,
        payload,
        expected_text,
    ):
        result = immunebuilder_module.run(**payload)

        assert result.get("error") == "invalid_immunebuilder_input"
        assert result.get("isError") is True
        assert expected_text in _error_blob(result)

    def test_run_rejects_mixed_chain_families(self, immunebuilder_module):
        result = immunebuilder_module.run(
            mode="antibody",
            heavy_chain=VALID_HEAVY_CHAIN,
            light_chain=VALID_LIGHT_CHAIN,
            alpha_chain=VALID_ALPHA_CHAIN,
            beta_chain=VALID_BETA_CHAIN,
        )

        assert result.get("error") == "invalid_immunebuilder_input"
        assert result.get("isError") is True
        detail_blob = _error_blob(result)
        assert "alpha_chain" in detail_blob or "beta_chain" in detail_blob or "tcr" in detail_blob

    @pytest.mark.parametrize(
        ("payload", "expected_text"),
        [
            (
                {
                    "mode": "nanobody",
                    "heavy_chain": f">heavy\n{VALID_HEAVY_CHAIN}",
                },
                "fasta",
            ),
            (
                {
                    "mode": "antibody",
                    "heavy_chain": VALID_HEAVY_CHAIN,
                    "light_chain": f"{VALID_LIGHT_CHAIN}*",
                },
                "valid amino acid",
            ),
        ],
    )
    def test_run_rejects_invalid_chain_text(
        self,
        immunebuilder_module,
        payload,
        expected_text,
    ):
        result = immunebuilder_module.run(**payload)

        assert result.get("error") == "invalid_immunebuilder_input"
        assert result.get("isError") is True
        assert expected_text in _error_blob(result)


class TestImmuneBuilderExecution:
    @pytest.mark.parametrize(
        ("mode", "raw_kwargs", "expected_builder", "expected_sequences", "expected_labels"),
        [
            (
                "antibody",
                {
                    "heavy_chain": _messy_sequence(VALID_HEAVY_CHAIN),
                    "light_chain": _messy_sequence(VALID_LIGHT_CHAIN),
                },
                "ABodyBuilder2",
                {"H": VALID_HEAVY_CHAIN, "L": VALID_LIGHT_CHAIN},
                ["H", "L"],
            ),
            (
                "nanobody",
                {"heavy_chain": _messy_sequence(VALID_HEAVY_CHAIN)},
                "NanoBodyBuilder2",
                {"H": VALID_HEAVY_CHAIN},
                ["H"],
            ),
            (
                "tcr",
                {
                    "alpha_chain": _messy_sequence(VALID_ALPHA_CHAIN),
                    "beta_chain": _messy_sequence(VALID_BETA_CHAIN),
                },
                "TCRBuilder2",
                {"A": VALID_ALPHA_CHAIN, "B": VALID_BETA_CHAIN},
                ["A", "B"],
            ),
        ],
    )
    def test_build_predictor_request_selects_expected_builder(
        self,
        immunebuilder_module,
        monkeypatch,
        mode,
        raw_kwargs,
        expected_builder,
        expected_sequences,
        expected_labels,
    ):
        fake_module, builders, _ = _make_fake_immunebuilder_module()
        monkeypatch.setitem(sys.modules, "ImmuneBuilder", fake_module)

        normalized = immunebuilder_module.normalize_args({"mode": mode, **raw_kwargs})
        predictor_cls, sequences, chain_labels = immunebuilder_module._build_predictor_request(
            normalized
        )

        assert predictor_cls is builders[expected_builder]
        assert sequences == expected_sequences
        assert chain_labels == expected_labels

    @pytest.mark.parametrize(
        ("mode", "raw_kwargs", "expected_builder", "expected_sequences", "expected_labels"),
        [
            (
                "antibody",
                {
                    "heavy_chain": _messy_sequence(VALID_HEAVY_CHAIN),
                    "light_chain": _messy_sequence(VALID_LIGHT_CHAIN),
                },
                "ABodyBuilder2",
                {"H": VALID_HEAVY_CHAIN, "L": VALID_LIGHT_CHAIN},
                ["H", "L"],
            ),
            (
                "nanobody",
                {"heavy_chain": _messy_sequence(VALID_HEAVY_CHAIN)},
                "NanoBodyBuilder2",
                {"H": VALID_HEAVY_CHAIN},
                ["H"],
            ),
            (
                "tcr",
                {
                    "alpha_chain": _messy_sequence(VALID_ALPHA_CHAIN),
                    "beta_chain": _messy_sequence(VALID_BETA_CHAIN),
                },
                "TCRBuilder2",
                {"A": VALID_ALPHA_CHAIN, "B": VALID_BETA_CHAIN},
                ["A", "B"],
            ),
        ],
    )
    def test_run_executes_each_mode_with_normalized_sequences(
        self,
        immunebuilder_module,
        monkeypatch,
        mode,
        raw_kwargs,
        expected_builder,
        expected_sequences,
        expected_labels,
    ):
        fake_module, builders, fake_prediction = _make_fake_immunebuilder_module()
        monkeypatch.setitem(sys.modules, "ImmuneBuilder", fake_module)

        result = immunebuilder_module.run(mode=mode, **raw_kwargs)

        assert result["summary"] == (
            f"ImmuneBuilder predicted a {mode} structure with {len(expected_labels)} chain(s)."
        )
        assert result["mode"] == mode
        assert result["chain_labels"] == expected_labels
        assert result["num_chains"] == len(expected_labels)
        assert result["pdb_content"] == MOCK_PDB
        assert result["metrics"]["predictor"] == builders[expected_builder].__name__
        assert result["metrics"]["num_residues"] == sum(len(seq) for seq in expected_sequences.values())
        assert builders[expected_builder].last_sequences == expected_sequences
        assert fake_prediction.saved_paths
        assert not Path(fake_prediction.saved_paths[-1]).exists()

    def test_run_persists_session_artifact(self, immunebuilder_module, monkeypatch, tmp_path):
        fake_module, _, _ = _make_fake_immunebuilder_module()
        monkeypatch.setitem(sys.modules, "ImmuneBuilder", fake_module)
        monkeypatch.setattr(immunebuilder_module, "Path", _redirect_workspace_paths(tmp_path))

        result = immunebuilder_module.run(
            mode="nanobody",
            heavy_chain=VALID_HEAVY_CHAIN,
            session_id="  session-42  ",
        )

        saved_path = tmp_path / "vol" / "workspace" / "session-42" / "predicted_structure.pdb"
        assert saved_path.exists()
        assert saved_path.read_text(encoding="utf-8") == result["pdb_content"]

    def test_run_returns_no_output_when_saved_pdb_is_blank(
        self,
        immunebuilder_module,
        monkeypatch,
    ):
        fake_module, _, fake_prediction = _make_fake_immunebuilder_module(pdb_text=" \n")
        monkeypatch.setitem(sys.modules, "ImmuneBuilder", fake_module)

        result = immunebuilder_module.run(
            mode="nanobody",
            heavy_chain=VALID_HEAVY_CHAIN,
        )

        assert result.get("error") == "no_output"
        assert result.get("isError") is True
        assert fake_prediction.saved_paths
        assert not Path(fake_prediction.saved_paths[-1]).exists()

    def test_run_wraps_prediction_save_failure_and_cleans_temp_file(
        self,
        immunebuilder_module,
        monkeypatch,
    ):
        fake_module, _, fake_prediction = _make_fake_immunebuilder_module(
            save_side_effect=RuntimeError("simulated save failure")
        )
        monkeypatch.setitem(sys.modules, "ImmuneBuilder", fake_module)

        result = immunebuilder_module.run(
            mode="nanobody",
            heavy_chain=VALID_HEAVY_CHAIN,
        )

        assert result.get("error") == "immunebuilder_runtime_error"
        assert result.get("isError") is True
        assert "simulated save failure" in _error_blob(result)
        assert fake_prediction.saved_paths
        assert not Path(fake_prediction.saved_paths[-1]).exists()

    def test_run_returns_dependency_error_when_immunebuilder_import_fails(
        self,
        immunebuilder_module,
        monkeypatch,
    ):
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "ImmuneBuilder":
                raise ModuleNotFoundError(name)
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", guarded_import)
        monkeypatch.delitem(sys.modules, "ImmuneBuilder", raising=False)

        result = immunebuilder_module.run(
            mode="nanobody",
            heavy_chain=VALID_HEAVY_CHAIN,
        )

        assert result.get("error") == "immunebuilder_not_installed"
        assert result.get("isError") is True
        assert "not installed" in _error_blob(result)
