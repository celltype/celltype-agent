"""Contract tests for the ImmuneBuilder container-tool implementation."""

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
        } <= set(input_properties)
        assert "chains" not in input_properties
        assert "sequence" not in input_properties
        assert set(input_properties["mode"].get("enum", [])) == {"antibody", "nanobody", "tcr"}
        assert "mode" in set(input_schema.get("required", []))

        assert output_schema["type"] == "object"
        output_properties = output_schema["properties"]
        required_output_fields = {"summary", "pdb_content", "mode", "chain_labels", "num_chains"}
        assert required_output_fields <= set(output_properties)
        assert required_output_fields <= set(output_schema.get("required", []))

    def test_run_rejects_generic_chain_map_passthrough(self, immunebuilder_module):
        result = immunebuilder_module.run(
            mode="nanobody",
            heavy_chain=VALID_HEAVY_CHAIN,
            chains={"H": VALID_HEAVY_CHAIN},
        )

        assert isinstance(result, dict)
        assert result.get("error")
        assert result.get("isError") is True
        assert "chains" in f"{result.get('summary', '')} {result.get('detail', '')}".lower()

    def test_fallback_contract_stays_strict_without_shared_helper(self):
        immunebuilder_module = _load_immunebuilder_module(block_shared_helper=True)

        contract = immunebuilder_module.get_contract()
        assert contract["input_schema"].get("additionalProperties") is False

        result = immunebuilder_module.run(
            mode="nanobody",
            heavy_chain=VALID_HEAVY_CHAIN,
            unexpected_field="typo",
        )

        assert isinstance(result, dict)
        assert result.get("error") == "invalid_immunebuilder_input"
        assert result.get("isError") is True
        detail_blob = f"{result.get('summary', '')} {result.get('detail', '')}".lower()
        assert "extra" in detail_blob or "permit" in detail_blob or "unexpected_field" in detail_blob

    def test_run_rejects_antibody_without_light_chain(self, immunebuilder_module):
        result = immunebuilder_module.run(
            mode="antibody",
            heavy_chain=VALID_HEAVY_CHAIN,
        )

        assert isinstance(result, dict)
        assert result.get("error")
        assert result.get("isError") is True
        assert "light_chain" in f"{result.get('summary', '')} {result.get('detail', '')}"

    def test_run_rejects_tcr_without_beta_chain(self, immunebuilder_module):
        result = immunebuilder_module.run(
            mode="tcr",
            alpha_chain=VALID_ALPHA_CHAIN,
        )

        assert isinstance(result, dict)
        assert result.get("error")
        assert result.get("isError") is True
        assert "beta_chain" in f"{result.get('summary', '')} {result.get('detail', '')}"

    def test_run_rejects_mixed_chain_families(self, immunebuilder_module):
        result = immunebuilder_module.run(
            mode="antibody",
            heavy_chain=VALID_HEAVY_CHAIN,
            light_chain=VALID_LIGHT_CHAIN,
            alpha_chain=VALID_ALPHA_CHAIN,
            beta_chain=VALID_BETA_CHAIN,
        )

        assert isinstance(result, dict)
        assert result.get("error")
        assert result.get("isError") is True
        detail_blob = f"{result.get('summary', '')} {result.get('detail', '')}".lower()
        assert "alpha_chain" in detail_blob or "beta_chain" in detail_blob or "tcr" in detail_blob

    def test_run_predicts_antibody_with_mocked_immunebuilder(
        self,
        immunebuilder_module,
        monkeypatch,
    ):
        class FakePrediction:
            def save(self, path: str):
                Path(path).write_text(
                    "HEADER MOCK\n"
                    "ATOM      1  CA  ALA H   1       0.0   0.0   0.0  1.00 70.00           C\n"
                    "END\n"
                )

        class FakeABodyBuilder2:
            last_sequences = None

            def predict(self, sequences):
                type(self).last_sequences = sequences
                return FakePrediction()

        fake_module = types.SimpleNamespace(
            ABodyBuilder2=FakeABodyBuilder2,
            NanoBodyBuilder2=type("FakeNanoBodyBuilder2", (), {}),
            TCRBuilder2=type("FakeTCRBuilder2", (), {}),
        )
        monkeypatch.setitem(sys.modules, "ImmuneBuilder", fake_module)

        result = immunebuilder_module.run(
            mode="antibody",
            heavy_chain=VALID_HEAVY_CHAIN,
            light_chain=VALID_LIGHT_CHAIN,
        )

        assert result["mode"] == "antibody"
        assert result["chain_labels"] == ["H", "L"]
        assert result["num_chains"] == 2
        assert "HEADER MOCK" in result["pdb_content"]
        assert result["metrics"]["predictor"] == "FakeABodyBuilder2"
        assert FakeABodyBuilder2.last_sequences == {"H": VALID_HEAVY_CHAIN, "L": VALID_LIGHT_CHAIN}
