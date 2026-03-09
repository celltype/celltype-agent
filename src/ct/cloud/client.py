"""
CellType Cloud client — submits GPU jobs to the API gateway,
polls for status, streams progress to terminal, handles cancellation.
"""

import logging
import time
from typing import Optional

import httpx

from ct.cloud.structure_inputs import inline_structure_file_args

logger = logging.getLogger("ct.cloud.client")

# Polling interval for job status
JOB_POLL_INTERVAL = 2.0
JOB_TIMEOUT = 600.0  # 10 minutes max


class CloudClient:
    """Client for interacting with the CellType Cloud API gateway."""

    def __init__(self, endpoint: str = "https://api.celltype.com"):
        self.endpoint = endpoint.rstrip("/")

    def _headers(self, token: str) -> dict:
        return {"Authorization": f"Bearer {token}"}

    def _dashboard_base_url(self) -> str:
        try:
            from ct.agent.config import Config
            cfg = Config.load()
            return str(cfg.get("cloud.dashboard_url", "https://cloud.celltype.com")).rstrip("/")
        except Exception:
            return "https://cloud.celltype.com"

    def _job_dashboard_url(self, job_id: str) -> str:
        return f"{self._dashboard_base_url()}/dashboard/jobs/{job_id}"

    def _prepare_tool_args(self, tool_name: str, tool_args: dict) -> dict:
        """Inline supported local file arguments before cloud submission.

        Some GPU tools accept structure text. If the caller passes a local
        structure file path instead, inline the file contents before submission.
        """
        return inline_structure_file_args(tool_name, tool_args, logger=logger)

    def get_balance(self, token: str) -> float:
        """Fetch the user's credit balance."""
        with httpx.Client(timeout=15) as client:
            resp = client.get(
                f"{self.endpoint}/account/credits",
                headers=self._headers(token),
            )
            resp.raise_for_status()
            return resp.json().get("balance", 0.0)

    def submit_and_wait(
        self,
        tool_name: str,
        gpu_profile: str,
        estimated_cost: float,
        token: str,
        **kwargs,
    ) -> dict:
        """Submit a GPU job and wait for completion.

        Auto-proceeds without approval prompt. Costs are tracked in the
        dashboard and visible via `ct credits`.

        Returns tool result dict.
        """
        from rich.console import Console

        console = Console()

        # Strip internal kwargs that shouldn't be sent to the API
        tool_args = {
            k: v for k, v in kwargs.items()
            if not k.startswith("_")
        }
        tool_args = self._prepare_tool_args(tool_name, tool_args)

        # Check balance
        try:
            balance = self.get_balance(token)
        except Exception:
            balance = None

        if balance is not None and balance < estimated_cost:
            return {
                "summary": (
                    f"Insufficient credits (Balance: ${balance:.2f}, "
                    f"Estimated: ~${estimated_cost:.2f}). "
                    "Add credits at celltype.com/billing."
                ),
                "skipped": True,
                "reason": "insufficient_credits",
            }

        # Submit job
        console.print(f"  [dim]Submitting {tool_name} to CellType Cloud...[/dim]")

        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{self.endpoint}/jobs/submit",
                headers=self._headers(token),
                json={
                    "tool_name": tool_name,
                    "gpu_profile": gpu_profile,
                    "args": tool_args,
                },
            )
            if resp.status_code == 402:
                data = resp.json()
                return {
                    "summary": data.get("error", "Insufficient credits."),
                    "skipped": True,
                    "reason": "insufficient_credits",
                }
            resp.raise_for_status()
            job_data = resp.json()

        job_id = job_data["job_id"]
        job_dashboard_url = self._job_dashboard_url(job_id)
        console.print(f"  [dim]Job {job_id} submitted. Waiting for result...[/dim]")
        console.print(f"  [dim]View in CellType Cloud:[/dim] {job_dashboard_url}")

        return self._poll_job(job_id, token, tool_name, console, job_dashboard_url)

    def _poll_job(
        self,
        job_id: str,
        token: str,
        tool_name: str,
        console,
        job_dashboard_url: str,
    ) -> dict:
        """Poll job status until complete."""
        start = time.time()
        last_status = ""

        with httpx.Client(timeout=15) as client:
            while time.time() - start < JOB_TIMEOUT:
                time.sleep(JOB_POLL_INTERVAL)

                try:
                    resp = client.get(
                        f"{self.endpoint}/jobs/status/{job_id}",
                        headers=self._headers(token),
                    )
                    resp.raise_for_status()
                    status_data = resp.json()
                except Exception as e:
                    logger.warning("Job status poll failed: %s", e)
                    continue

                status = status_data.get("status", "unknown")
                message = status_data.get("message", "")
                elapsed = int(time.time() - start)

                # Show progress updates
                if message and message != last_status:
                    console.print(f"  {message} [{elapsed}s]")
                    last_status = message

                if status == "completed":
                    result = status_data.get("result", {})
                    actual_cost = status_data.get("actual_cost", 0.0)
                    if actual_cost > 0:
                        console.print(
                            f"  [green]Done[/green] ({elapsed}s, ${actual_cost:.4f})"
                        )

                        # Low balance warning
                        new_balance = status_data.get("balance")
                        if new_balance is not None and new_balance < 1.0:
                            console.print(
                                f"\n  [yellow]Low balance (${new_balance:.2f}).[/yellow] "
                                "Add credits at celltype.com/billing"
                            )
                    result_dict = result if isinstance(result, dict) else {"summary": str(result)}
                    result_dict.setdefault("job_id", job_id)
                    result_dict.setdefault("job_dashboard_url", job_dashboard_url)
                    summary = result_dict.get("summary")
                    if summary and job_dashboard_url not in summary:
                        result_dict["summary"] = (
                            f"{summary}\nView in CellType Cloud: {job_dashboard_url}"
                        )
                    return result_dict

                elif status == "failed":
                    error = status_data.get("error", "Unknown error")
                    return {
                        "summary": (
                            f"[Failed] {tool_name} — {error}\n"
                            f"View in CellType Cloud: {job_dashboard_url}"
                        ),
                        "error": error,
                        "job_id": job_id,
                        "job_dashboard_url": job_dashboard_url,
                    }

                elif status == "cancelled":
                    return {
                        "summary": (
                            f"[Cancelled] {tool_name} — job cancelled.\n"
                            f"View in CellType Cloud: {job_dashboard_url}"
                        ),
                        "skipped": True,
                        "reason": "cancelled",
                        "job_id": job_id,
                        "job_dashboard_url": job_dashboard_url,
                    }

        return {
            "summary": (
                f"[Timeout] {tool_name} — job timed out after {JOB_TIMEOUT}s.\n"
                f"View in CellType Cloud: {job_dashboard_url}"
            ),
            "skipped": True,
            "reason": "timeout",
            "job_id": job_id,
            "job_dashboard_url": job_dashboard_url,
        }
