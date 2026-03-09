"""Tests for job observability metrics (in-memory DAL)."""

import pytest
import sys
import os

# Add celltype-cloud to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "celltype-cloud"))


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset in-memory metrics store between tests."""
    import dal_memory
    dal_memory._job_metrics.clear()
    yield
    dal_memory._job_metrics.clear()


class TestStoreJobMetrics:
    @pytest.mark.asyncio
    async def test_store_success_metrics(self):
        import dal_memory as dal

        await dal.store_job_metrics(
            job_id="job-1",
            tool_name="structure.esmfold",
            gpu_type="A10G",
            cold_start_ms=5000,
            execution_ms=40000,
            total_duration_ms=45000,
            cost_usd=0.05,
            success=True,
        )

        assert len(dal._job_metrics) == 1
        m = dal._job_metrics[0]
        assert m["tool_name"] == "structure.esmfold"
        assert m["cold_start_ms"] == 5000
        assert m["success"] is True

    @pytest.mark.asyncio
    async def test_store_failure_metrics(self):
        import dal_memory as dal

        await dal.store_job_metrics(
            job_id="job-2",
            tool_name="structure.diffdock",
            total_duration_ms=10000,
            success=False,
            error="CUDA OOM",
        )

        assert len(dal._job_metrics) == 1
        m = dal._job_metrics[0]
        assert m["success"] is False
        assert m["error"] == "CUDA OOM"


class TestGetToolMetrics:
    @pytest.mark.asyncio
    async def test_empty_metrics(self):
        import dal_memory as dal

        result = await dal.get_tool_metrics("structure.esmfold")
        assert result["total_runs"] == 0

    @pytest.mark.asyncio
    async def test_aggregated_metrics(self):
        import dal_memory as dal

        for i in range(5):
            await dal.store_job_metrics(
                job_id=f"job-{i}",
                tool_name="structure.esmfold",
                gpu_type="A10G",
                cold_start_ms=3000 + i * 1000,
                execution_ms=20000 + i * 5000,
                total_duration_ms=23000 + i * 6000,
                cost_usd=0.03 + i * 0.01,
                success=True,
            )

        # Add one failure
        await dal.store_job_metrics(
            job_id="job-fail",
            tool_name="structure.esmfold",
            total_duration_ms=5000,
            success=False,
            error="timeout",
        )

        result = await dal.get_tool_metrics("structure.esmfold")
        assert result["total_runs"] == 6
        assert result["success_rate"] == pytest.approx(5 / 6)
        assert result["avg_execution_ms"] > 0
        assert result["total_cost"] > 0


class TestGetCostDashboard:
    @pytest.mark.asyncio
    async def test_dashboard(self):
        import dal_memory as dal

        await dal.store_job_metrics(
            job_id="j1", tool_name="structure.esmfold",
            total_duration_ms=45000, cost_usd=0.05, success=True,
        )
        await dal.store_job_metrics(
            job_id="j2", tool_name="structure.diffdock",
            total_duration_ms=30000, cost_usd=0.03, success=True,
        )

        dashboard = await dal.get_cost_dashboard()
        assert len(dashboard) == 2
        # Sorted by cost desc
        assert dashboard[0]["tool_name"] == "structure.esmfold"
        assert dashboard[0]["total_cost"] == 0.05
