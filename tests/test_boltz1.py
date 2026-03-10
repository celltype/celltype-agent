"""Tests for Boltz-1 structure prediction tool.

All subprocess calls are mocked — no GPU or network required.
"""
from __future__ import annotations

import json
import os
import subprocess
import pytest
from unittest.mock import patch, MagicMock, call


VALID_SEQ = "MKTLLILAVLCLAV"
SAMPLE_PDB = "HEADER BOLTZ1 TEST\nATOM      1  CA  ALA A   1       1.0   2.0   3.0\nEND\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_run_success(pdb_text=SAMPLE_PDB, confidence=0.87):
    """Patch helper: write real output files and return success mock."""
    m = MagicMock()
    m.returncode = 0
    m.stdout = ""
    m.stderr = ""
    return m


def _make_fake_output(base_dir, pdb_text=SAMPLE_PDB, confidence=0.87):
    """Create the fake boltz output directory structure under base_dir."""
    out_dir = base_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predicted.pdb").write_text(pdb_text)
    (out_dir / "confidence_model.json").write_text(
        json.dumps({"confidence": confidence})
    )
    return out_dir


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestBoltz1Validation:
    def test_empty_sequence_returns_error(self):
        from ct.tools.boltz1.implementation import run
        result = run(sequence="")
        assert result["error"] == "no_sequence"

    def test_missing_sequence_kwarg(self):
        from ct.tools.boltz1.implementation import run
        result = run()
        assert result["error"] == "no_sequence"

    def test_whitespace_only_sequence(self):
        from ct.tools.boltz1.implementation import run
        result = run(sequence="   ")
        assert result["error"] == "no_sequence"


# ---------------------------------------------------------------------------
# Successful predictions
# ---------------------------------------------------------------------------

class TestBoltz1Success:
    def _run_mocked(self, tmp_path, sequence, ligand_smiles=None,
                    pdb_text=SAMPLE_PDB, confidence=0.87):
        from ct.tools.boltz1.implementation import run

        _make_fake_output(tmp_path, pdb_text, confidence)
        success_mock = _fake_run_success(pdb_text, confidence)

        with patch("ct.tools.boltz1.implementation.subprocess.run", return_value=success_mock), \
             patch("ct.tools.boltz1.implementation.shutil.which", return_value="boltz"), \
             patch("ct.tools.boltz1.implementation._get_vram_mb", return_value=0), \
             patch("ct.tools.boltz1.implementation.tempfile.TemporaryDirectory") as mock_tmp:

            mock_tmp.return_value.__enter__ = lambda s: str(tmp_path)
            mock_tmp.return_value.__exit__ = MagicMock(return_value=False)

            kwargs = {"sequence": sequence}
            if ligand_smiles:
                kwargs["ligand_smiles"] = ligand_smiles
            return run(**kwargs)

    def test_protein_only_success(self, tmp_path):
        result = self._run_mocked(tmp_path, VALID_SEQ)
        assert "error" not in result
        assert result["num_residues"] == len(VALID_SEQ)
        assert result["ligand_included"] is False
        assert SAMPLE_PDB[:50] in result["pdb_content"]

    def test_protein_ligand_complex(self, tmp_path):
        result = self._run_mocked(tmp_path, VALID_SEQ, ligand_smiles="CCO")
        assert "error" not in result
        assert result["ligand_included"] is True
        assert "CCO" in result["summary"]

    def test_confidence_in_result(self, tmp_path):
        result = self._run_mocked(tmp_path, VALID_SEQ, confidence=0.91)
        assert abs(result["confidence"] - 0.91) < 0.01

    def test_metrics_keys_present(self, tmp_path):
        result = self._run_mocked(tmp_path, VALID_SEQ)
        m = result["metrics"]
        for key in ("vram_before_mb", "vram_peak_mb", "vram_delta_mb",
                    "time_execution_s", "time_total_s"):
            assert key in m, f"Missing metric: {key}"

    def test_pdb_content_truncated_to_5000(self, tmp_path):
        long_pdb = "ATOM  \n" * 1000  # ~7000 chars
        result = self._run_mocked(tmp_path, VALID_SEQ, pdb_text=long_pdb)
        assert len(result["pdb_content"]) <= 5000

    def test_fasta_header_stripped(self, tmp_path):
        fasta = f">sp|P00000|TEST Test protein\n{VALID_SEQ}"
        result = self._run_mocked(tmp_path, fasta)
        assert result["num_residues"] == len(VALID_SEQ)

    def test_summary_contains_residue_count(self, tmp_path):
        result = self._run_mocked(tmp_path, VALID_SEQ)
        assert str(len(VALID_SEQ)) in result["summary"]


# ---------------------------------------------------------------------------
# Failure / edge cases
# ---------------------------------------------------------------------------

class TestBoltz1Failures:
    def test_timeout_returns_error(self, tmp_path):
        from ct.tools.boltz1.implementation import run

        with patch("ct.tools.boltz1.implementation.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="boltz", timeout=600)), \
             patch("ct.tools.boltz1.implementation.shutil.which", return_value="boltz"), \
             patch("ct.tools.boltz1.implementation._get_vram_mb", return_value=0), \
             patch("ct.tools.boltz1.implementation.tempfile.TemporaryDirectory") as mock_tmp:

            mock_tmp.return_value.__enter__ = lambda s: str(tmp_path)
            mock_tmp.return_value.__exit__ = MagicMock(return_value=False)
            (tmp_path / "output").mkdir(parents=True, exist_ok=True)

            result = run(sequence=VALID_SEQ)

        assert result["error"] == "timeout"

    def test_no_output_file_returns_error(self, tmp_path):
        from ct.tools.boltz1.implementation import run

        success_mock = MagicMock()
        success_mock.returncode = 0
        success_mock.stderr = ""
        success_mock.stdout = ""

        with patch("ct.tools.boltz1.implementation.subprocess.run", return_value=success_mock), \
             patch("ct.tools.boltz1.implementation.shutil.which", return_value="boltz"), \
             patch("ct.tools.boltz1.implementation._get_vram_mb", return_value=0), \
             patch("ct.tools.boltz1.implementation.tempfile.TemporaryDirectory") as mock_tmp:

            mock_tmp.return_value.__enter__ = lambda s: str(tmp_path)
            mock_tmp.return_value.__exit__ = MagicMock(return_value=False)
            # Output dir exists but is empty — no PDB produced
            (tmp_path / "output").mkdir(parents=True, exist_ok=True)

            result = run(sequence=VALID_SEQ)

        assert result["error"] == "no_output"

    def test_nonzero_returncode_retries_then_errors(self, tmp_path):
        """On failure, boltz should be called twice (retry with fewer steps)."""
        from ct.tools.boltz1.implementation import run

        fail_mock = MagicMock()
        fail_mock.returncode = 1
        fail_mock.stderr = "CUDA OOM on device"
        fail_mock.stdout = ""

        with patch("ct.tools.boltz1.implementation.subprocess.run", return_value=fail_mock) as mock_run, \
             patch("ct.tools.boltz1.implementation.shutil.which", return_value="boltz"), \
             patch("ct.tools.boltz1.implementation._get_vram_mb", return_value=0), \
             patch("ct.tools.boltz1.implementation.tempfile.TemporaryDirectory") as mock_tmp:

            mock_tmp.return_value.__enter__ = lambda s: str(tmp_path)
            mock_tmp.return_value.__exit__ = MagicMock(return_value=False)
            (tmp_path / "output").mkdir(parents=True, exist_ok=True)

            result = run(sequence=VALID_SEQ)

        # Called at least twice (original + retry)
        assert mock_run.call_count >= 2
        assert "error" in result

    def test_retry_uses_fewer_sampling_steps(self, tmp_path):
        """Retry command should use 50 sampling steps instead of 200."""
        from ct.tools.boltz1.implementation import run

        fail_mock = MagicMock(returncode=1, stderr="OOM", stdout="")
        calls = []

        def capture_run(cmd, **kwargs):
            calls.append(cmd)
            return fail_mock

        with patch("ct.tools.boltz1.implementation.subprocess.run", side_effect=capture_run), \
             patch("ct.tools.boltz1.implementation.shutil.which", return_value="boltz"), \
             patch("ct.tools.boltz1.implementation._get_vram_mb", return_value=0), \
             patch("ct.tools.boltz1.implementation.tempfile.TemporaryDirectory") as mock_tmp:

            mock_tmp.return_value.__enter__ = lambda s: str(tmp_path)
            mock_tmp.return_value.__exit__ = MagicMock(return_value=False)
            (tmp_path / "output").mkdir(parents=True, exist_ok=True)

            run(sequence=VALID_SEQ)

        assert len(calls) >= 2
        # Retry should have "50" not "200"
        assert "50" in calls[1]
        assert "200" not in calls[1]


# ---------------------------------------------------------------------------
# tool_entrypoint
# ---------------------------------------------------------------------------

class TestBoltz1Entrypoint:
    def test_entrypoint_reads_input_writes_output(self, tmp_path):
        input_file = tmp_path / "input.json"
        output_file = tmp_path / "output.json"
        input_file.write_text(json.dumps({"sequence": VALID_SEQ}))

        work = tmp_path / "work"
        _make_fake_output(work)

        success_mock = MagicMock(returncode=0, stdout="", stderr="")

        with patch.dict(os.environ, {
            "INPUT_FILE": str(input_file),
            "OUTPUT_FILE": str(output_file),
        }):
            with patch("ct.tools.boltz1.implementation.subprocess.run", return_value=success_mock), \
                 patch("ct.tools.boltz1.implementation.shutil.which", return_value="boltz"), \
                 patch("ct.tools.boltz1.implementation._get_vram_mb", return_value=0), \
                 patch("ct.tools.boltz1.implementation.tempfile.TemporaryDirectory") as mock_tmp:

                mock_tmp.return_value.__enter__ = lambda s: str(work)
                mock_tmp.return_value.__exit__ = MagicMock(return_value=False)

                import ct.tools.boltz1.tool_entrypoint as ep
                ep.main()

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert "summary" in data

    def test_entrypoint_writes_error_on_bad_json(self, tmp_path):
        input_file = tmp_path / "input.json"
        output_file = tmp_path / "output.json"
        input_file.write_text("{bad json{{")

        with patch.dict(os.environ, {
            "INPUT_FILE": str(input_file),
            "OUTPUT_FILE": str(output_file),
        }):
            import ct.tools.boltz1.tool_entrypoint as ep
            with pytest.raises(SystemExit):
                ep.main()

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert "error" in data
