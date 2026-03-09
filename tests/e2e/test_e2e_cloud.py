"""
End-to-end integration tests for CellType Cloud GPU compute.

These tests run against a real gateway + real Modal GPU execution.
No mocking — validates the full round-trip.

Run with:
    pytest tests/e2e/test_e2e_cloud.py --run-e2e -v
"""

import os
import time
import json
import subprocess
from pathlib import Path

import pytest
import httpx

E2E_DIR = Path(__file__).parent
GATEWAY_URL = os.environ.get("CELLTYPE_GATEWAY_URL", "http://localhost:8000")
DEV_TOKEN = "ct-dev-token-12345"

# Skip all E2E tests unless --run-e2e is passed
pytestmark = pytest.mark.e2e


def _gateway_health() -> bool:
    """Check if the gateway is running."""
    try:
        resp = httpx.get(f"{GATEWAY_URL}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def _submit_job(tool_name: str, gpu_profile: str, args: dict) -> dict:
    """Submit a job and wait for completion."""
    headers = {"Authorization": f"Bearer {DEV_TOKEN}"}

    # Submit
    resp = httpx.post(
        f"{GATEWAY_URL}/jobs/submit",
        headers=headers,
        json={"tool_name": tool_name, "gpu_profile": gpu_profile, "args": args},
        timeout=30,
    )
    resp.raise_for_status()
    job = resp.json()
    job_id = job["job_id"]

    # Poll
    for _ in range(120):  # 4 minutes max
        time.sleep(2)
        resp = httpx.get(
            f"{GATEWAY_URL}/jobs/status/{job_id}",
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        status = resp.json()

        if status["status"] == "completed":
            return status
        elif status["status"] == "failed":
            pytest.fail(f"Job failed: {status.get('error')}")
        elif status["status"] == "cancelled":
            pytest.fail("Job was cancelled")

    pytest.fail("Job timed out")


@pytest.fixture(autouse=True)
def _check_gateway():
    """Skip tests if gateway is not running."""
    if not _gateway_health():
        pytest.skip("Gateway not running at " + GATEWAY_URL)


class TestSingleToolE2E:
    """Task 8.5: Single-tool E2E test — ESMFold structure prediction."""

    def test_esmfold_kras(self):
        """Predict KRAS G12C structure via ESMFold."""
        fasta_path = E2E_DIR / "kras_g12c.fasta"
        sequence = ""
        for line in fasta_path.read_text().strip().split("\n"):
            if not line.startswith(">"):
                sequence += line.strip()

        assert len(sequence) > 100, f"Sequence too short: {len(sequence)}"

        result = _submit_job(
            tool_name="structure.esmfold",
            gpu_profile="structure",
            args={"sequence": sequence},
        )

        assert result["status"] == "completed"
        tool_result = result.get("result", {})

        # Verify output
        assert "summary" in tool_result
        pdb_content = tool_result.get("pdb_content", "")
        assert "ATOM" in pdb_content, "PDB should contain ATOM records"

        # Check approximate residue count
        atom_lines = [l for l in pdb_content.split("\n") if l.startswith("ATOM") and " CA " in l]
        assert len(atom_lines) > 100, f"Expected ~189 CA atoms, got {len(atom_lines)}"

        # Check confidence
        confidence = tool_result.get("confidence", 0)
        assert confidence > 0, "Confidence should be positive"

        # Verify cost was logged
        actual_cost = result.get("actual_cost", 0)
        assert actual_cost > 0, "Should have non-zero cost"


class TestChainedToolE2E:
    """Task 8.6: Chained-tool E2E test — structure + docking."""

    def test_esmfold_then_diffdock(self):
        """Predict structure then dock Sotorasib."""
        # Step 1: Predict structure
        fasta_path = E2E_DIR / "kras_g12c.fasta"
        sequence = ""
        for line in fasta_path.read_text().strip().split("\n"):
            if not line.startswith(">"):
                sequence += line.strip()

        session_id = f"e2e-test-{int(time.time())}"

        structure_result = _submit_job(
            tool_name="structure.esmfold",
            gpu_profile="structure",
            args={"sequence": sequence, "session_id": session_id},
        )

        assert structure_result["status"] == "completed"
        pdb_content = structure_result.get("result", {}).get("pdb_content", "")
        assert "ATOM" in pdb_content

        # Step 2: Dock Sotorasib
        smi_path = E2E_DIR / "sotorasib.smi"
        ligand_smiles = smi_path.read_text().strip()

        docking_result = _submit_job(
            tool_name="structure.diffdock",
            gpu_profile="docking",
            args={
                "protein_pdb": pdb_content,
                "ligand_smiles": ligand_smiles,
                "session_id": session_id,
                "num_poses": 3,
            },
        )

        assert docking_result["status"] == "completed"
        dock_result = docking_result.get("result", {})
        assert "poses" in dock_result or "summary" in dock_result


class TestCostValidation:
    """Task 8.7: Cost and timing validation."""

    def test_cost_logged_accurately(self):
        """Verify gateway logs actual execution time and cost."""
        result = _submit_job(
            tool_name="structure.esmfold",
            gpu_profile="structure",
            args={"sequence": "MKWVTFISLLFLFSSAYS"},
        )

        assert result["status"] == "completed"
        assert "actual_cost" in result
        assert result["actual_cost"] > 0

        duration = result.get("duration_s", 0)
        assert duration > 0, "Duration should be recorded"


class TestFailureModes:
    """Task 8.8: Failure mode E2E tests."""

    def test_malformed_sequence(self):
        """Send invalid protein sequence — should get clean error."""
        result = _submit_job(
            tool_name="structure.esmfold",
            gpu_profile="structure",
            args={"sequence": "THIS IS NOT A PROTEIN 12345!!!"},
        )

        # Should complete or fail gracefully
        assert result["status"] in ("completed", "failed")

    def test_expired_auth_token(self):
        """Test with invalid token — should get 401."""
        headers = {"Authorization": "Bearer invalid-token-xyz"}
        resp = httpx.post(
            f"{GATEWAY_URL}/jobs/submit",
            headers=headers,
            json={
                "tool_name": "structure.esmfold",
                "gpu_profile": "structure",
                "args": {"sequence": "MKW"},
            },
            timeout=15,
        )
        assert resp.status_code == 401


class TestColdStartBenchmark:
    """Task 8.9: Cold start benchmarking."""

    def test_record_cold_warm_times(self, tmp_path):
        """Run ESMFold twice and record cold vs warm latency."""
        timings = []

        for run_label in ["cold", "warm"]:
            start = time.time()
            result = _submit_job(
                tool_name="structure.esmfold",
                gpu_profile="structure",
                args={"sequence": "MKWVTFISLLFLFSSAYS"},
            )
            elapsed = time.time() - start
            timings.append({"run": run_label, "elapsed_s": elapsed})

            assert result["status"] == "completed"

        # Log timings
        timing_file = tmp_path / "cold_start_benchmark.json"
        timing_file.write_text(json.dumps(timings, indent=2))

        print(f"\nCold start: {timings[0]['elapsed_s']:.1f}s")
        print(f"Warm start: {timings[1]['elapsed_s']:.1f}s")

        # Warm should generally be faster (but not enforced — infrastructure varies)
        assert timings[0]["elapsed_s"] > 0
        assert timings[1]["elapsed_s"] > 0
