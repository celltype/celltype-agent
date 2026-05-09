"""Tests for the CellType Cloud client — mocked gateway API."""

from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


class TestCloudClient:
    """Test cloud client job submission and polling."""

    def test_get_balance(self):
        from ct.cloud.client import CloudClient

        client = CloudClient(endpoint="http://localhost:8000")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"balance": 8.50}
        mock_resp.raise_for_status = MagicMock()

        with patch("ct.cloud.client.httpx.Client") as MockClient:
            mock_http = MagicMock()
            mock_http.get.return_value = mock_resp
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_http)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            balance = client.get_balance("test-token")
            assert balance == 8.50

    def test_submit_insufficient_credits(self):
        from ct.cloud.client import CloudClient

        client = CloudClient(endpoint="http://localhost:8000")

        # Mock balance check returning low balance
        with patch.object(client, "get_balance", return_value=0.03):
            result = client.submit_and_wait(
                tool_name="structure.esmfold",
                gpu_profile="structure",
                estimated_cost=0.10,
                token="test-token",
                sequence="MKWVTF",
            )
            assert result["skipped"] is True
            assert result["reason"] == "insufficient_credits"

    @pytest.mark.parametrize(
        ("tool_name", "arg_name"),
        [
            ("design.rfdiffusion", "target_pdb"),
            ("design.proteinmpnn", "backbone_pdb"),
            ("structure.diffdock", "protein_pdb"),
        ],
    )
    def test_prepare_tool_args_inlines_local_pdb_file(self, tmp_path, tool_name, arg_name):
        from ct.cloud.client import CloudClient

        client = CloudClient(endpoint="http://localhost:8000")
        pdb_path = tmp_path / "target.pdb"
        pdb_content = "ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00  0.00           C\nEND\n"
        pdb_path.write_text(pdb_content, encoding="utf-8")

        prepared = client._prepare_tool_args(
            tool_name,
            {arg_name: str(pdb_path), "other_arg": 3},
        )

        assert prepared[arg_name] == pdb_content
        assert prepared["other_arg"] == 3

    def test_prepare_tool_args_keeps_inline_pdb_content(self):
        from ct.cloud.client import CloudClient

        client = CloudClient(endpoint="http://localhost:8000")
        pdb_content = "ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00  0.00           C\nEND\n"

        prepared = client._prepare_tool_args(
            "design.rfdiffusion",
            {"target_pdb": pdb_content},
        )

        assert prepared["target_pdb"] == pdb_content


class TestCloudClientIntegration:
    """Integration test — full flow with mocked gateway."""

    def test_submit_poll_complete(self, tmp_path):
        from ct.cloud.client import CloudClient

        client = CloudClient(endpoint="http://localhost:8000")

        # Mock the full flow: balance check, submit, poll
        submit_resp = MagicMock()
        submit_resp.status_code = 200
        submit_resp.json.return_value = {"job_id": "job-123", "estimated_cost": 0.05}
        submit_resp.raise_for_status = MagicMock()

        status_resp = MagicMock()
        status_resp.status_code = 200
        status_resp.json.return_value = {
            "status": "completed",
            "result": {"summary": "Structure predicted", "pdb_content": "ATOM...\nEND\n"},
            "actual_cost": 0.04,
            "balance": 9.96,
        }
        status_resp.raise_for_status = MagicMock()
        artifact_resp = MagicMock()
        artifact_resp.status_code = 200
        artifact_resp.text = "ATOM...\nEND\n"
        artifact_resp.raise_for_status = MagicMock()

        with patch.object(client, "get_balance", return_value=10.0):
            with patch("ct.cloud.client.httpx.Client") as MockClient:
                mock_http = MagicMock()
                mock_http.post.return_value = submit_resp
                mock_http.get.side_effect = [status_resp, artifact_resp]
                MockClient.return_value.__enter__ = MagicMock(return_value=mock_http)
                MockClient.return_value.__exit__ = MagicMock(return_value=False)

                with patch("ct.cloud.client.time.sleep"):
                    with patch("ct.cloud.artifacts._artifact_root", return_value=tmp_path / "outputs"):
                        result = client.submit_and_wait(
                            tool_name="structure.esmfold",
                            gpu_profile="structure",
                            estimated_cost=0.05,
                            token="test-token",
                            sequence="MKWVTF",
                        )
                        assert result["job_id"] == "job-123"
                        assert result["job_dashboard_url"] == "https://cloud.celltype.com/dashboard/jobs/job-123"
                        assert "Structure predicted" in result["summary"]
                        assert "Full PDB saved to" in result["summary"]
                        assert "https://cloud.celltype.com/dashboard/jobs/job-123" in result["summary"]
                        assert "pdb_content" not in result
                        assert result["pdb_path"].endswith("outputs/structures/esmfold_job-123.pdb")
                        assert result["pdb_preview"] == "ATOM...\nEND\n"
                        assert Path(result["pdb_path"]).read_text(encoding="utf-8") == "ATOM...\nEND\n"

    def test_submit_auto_proceeds(self):
        """Verify no approval prompt — jobs auto-proceed."""
        from ct.cloud.client import CloudClient

        client = CloudClient(endpoint="http://localhost:8000")

        submit_resp = MagicMock()
        submit_resp.status_code = 200
        submit_resp.json.return_value = {"job_id": "job-456", "estimated_cost": 0.05}
        submit_resp.raise_for_status = MagicMock()

        status_resp = MagicMock()
        status_resp.status_code = 200
        status_resp.json.return_value = {
            "status": "completed",
            "result": {"summary": "Done"},
            "actual_cost": 0.03,
        }
        status_resp.raise_for_status = MagicMock()
        artifact_resp = MagicMock()
        artifact_resp.status_code = 404
        artifact_resp.text = ""

        with patch.object(client, "get_balance", return_value=10.0):
            with patch("ct.cloud.client.httpx.Client") as MockClient:
                mock_http = MagicMock()
                mock_http.post.return_value = submit_resp
                mock_http.get.side_effect = [status_resp, artifact_resp]
                MockClient.return_value.__enter__ = MagicMock(return_value=mock_http)
                MockClient.return_value.__exit__ = MagicMock(return_value=False)

                with patch("ct.cloud.client.time.sleep"):
                    # Should NOT call input() at all
                    with patch("builtins.input", side_effect=AssertionError("input() should not be called")):
                        result = client.submit_and_wait(
                            tool_name="structure.esmfold",
                            gpu_profile="structure",
                            estimated_cost=0.05,
                            token="test-token",
                            sequence="MKWVTF",
                        )
                        assert result["job_id"] == "job-456"
                        assert result["job_dashboard_url"] == "https://cloud.celltype.com/dashboard/jobs/job-456"
                        assert "Done" in result["summary"]
