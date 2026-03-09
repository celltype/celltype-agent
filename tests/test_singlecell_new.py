"""Tests for singlecell.cellrank_fate."""

import sys
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import numpy as np
import pandas as pd

from ct.tools.singlecell import cellrank_fate


def _make_mock_adata(n_cells=100, n_genes=50, has_velocity=True, cluster_key="cluster"):
    """Build a mock AnnData object for testing."""
    adata = MagicMock()
    adata.shape = (n_cells, n_genes)

    # Layers
    layers = {}
    if has_velocity:
        layers["velocity"] = np.random.randn(n_cells, n_genes)
    adata.layers = layers

    # obsm
    obsm = {}
    if has_velocity:
        obsm["velocities"] = np.random.randn(n_cells, 2)
    adata.obsm = obsm

    # obs with cluster column
    clusters = pd.Series(
        np.random.choice(["0", "1", "2"], size=n_cells),
        name=cluster_key,
    )
    obs = pd.DataFrame({cluster_key: clusters})
    adata.obs = obs

    return adata


class TestCellRankFate:
    """Tests for the CellRank fate mapping tool."""

    def test_cellrank_not_installed(self):
        """When cellrank is not installed, should return install instructions."""
        original = sys.modules.get("cellrank")
        sys.modules["cellrank"] = None
        try:
            result = cellrank_fate(data_path="/some/data.h5ad")
        finally:
            if original is not None:
                sys.modules["cellrank"] = original
            else:
                sys.modules.pop("cellrank", None)
        assert "error" in result
        assert "cellrank" in result["summary"].lower()

    def test_scanpy_not_installed(self):
        """When scanpy is not available, should return install instructions."""
        mock_cr = MagicMock()
        with patch.dict("sys.modules", {"cellrank": mock_cr}):
            with patch("ct.tools.singlecell._check_scanpy", return_value=None):
                result = cellrank_fate(data_path="/some/data.h5ad")
        assert "error" in result
        assert "scanpy" in result["summary"].lower()

    def test_failed_to_load_data(self):
        """When the h5ad file cannot be read, should return error."""
        mock_cr = MagicMock()
        mock_sc = MagicMock()
        mock_sc.read_h5ad.side_effect = FileNotFoundError("No such file")

        with patch.dict("sys.modules", {"cellrank": mock_cr}):
            with patch("ct.tools.singlecell._check_scanpy", return_value=mock_sc):
                result = cellrank_fate(data_path="/nonexistent/data.h5ad")

        assert "error" in result
        assert "Failed to load" in result["error"]

    def test_successful_fate_mapping_with_velocity(self):
        """Successful fate mapping with velocity data."""
        mock_cr = MagicMock()
        mock_sc = MagicMock()

        adata = _make_mock_adata(n_cells=100, has_velocity=True)
        mock_sc.read_h5ad.return_value = adata

        # Mock kernels
        mock_vk = MagicMock()
        mock_ck = MagicMock()
        mock_combined = MagicMock()
        mock_vk.__rmul__ = MagicMock(return_value=mock_vk)
        mock_vk.__mul__ = MagicMock(return_value=mock_vk)
        mock_ck.__rmul__ = MagicMock(return_value=mock_ck)
        mock_ck.__mul__ = MagicMock(return_value=mock_ck)
        mock_vk.__add__ = MagicMock(return_value=mock_combined)

        mock_vk_class = MagicMock(return_value=mock_vk)
        mock_ck_class = MagicMock(return_value=mock_ck)

        # Mock GPCCA estimator
        mock_estimator = MagicMock()
        terminal_series = pd.Series(["terminal_A", None, "terminal_B"] * 33 + ["terminal_A"])
        mock_estimator.terminal_states = terminal_series

        # Mock fate probabilities
        mock_fate = MagicMock()
        mock_fate.names = ["terminal_A", "terminal_B"]
        mock_fate.shape = (100, 2)
        mock_fate.__getitem__ = MagicMock(
            return_value=np.random.rand(50, 2)
        )
        mock_estimator.fate_probabilities = mock_fate

        mock_gpcca_class = MagicMock(return_value=mock_estimator)

        with patch.dict("sys.modules", {
            "cellrank": mock_cr,
            "cellrank.kernels": MagicMock(
                VelocityKernel=mock_vk_class,
                ConnectivityKernel=mock_ck_class,
            ),
            "cellrank.estimators": MagicMock(GPCCA=mock_gpcca_class),
        }):
            with patch("ct.tools.singlecell._check_scanpy", return_value=mock_sc):
                result = cellrank_fate(data_path="/some/data.h5ad")

        assert "summary" in result
        assert result["n_cells"] == 100
        assert result["has_velocity"] is True
        assert "terminal_states" in result
        assert "velocity+connectivity" in result["summary"]

    def test_successful_fate_mapping_without_velocity(self):
        """Successful fate mapping without velocity (connectivity-only kernel)."""
        mock_cr = MagicMock()
        mock_sc = MagicMock()

        adata = _make_mock_adata(n_cells=50, has_velocity=False)
        mock_sc.read_h5ad.return_value = adata

        mock_ck = MagicMock()
        mock_ck_class = MagicMock(return_value=mock_ck)

        # Mock GPCCA estimator
        mock_estimator = MagicMock()
        mock_estimator.terminal_states = pd.Series([None] * 50)
        mock_fate = MagicMock()
        mock_fate.names = []
        mock_fate.shape = (50, 0)
        mock_estimator.fate_probabilities = mock_fate

        mock_gpcca_class = MagicMock(return_value=mock_estimator)

        with patch.dict("sys.modules", {
            "cellrank": mock_cr,
            "cellrank.kernels": MagicMock(ConnectivityKernel=mock_ck_class),
            "cellrank.estimators": MagicMock(GPCCA=mock_gpcca_class),
        }):
            with patch("ct.tools.singlecell._check_scanpy", return_value=mock_sc):
                result = cellrank_fate(data_path="/some/data.h5ad")

        assert result["has_velocity"] is False
        assert "connectivity-only" in result["summary"]

    def test_kernel_computation_failure(self):
        """When kernel computation fails, should return error."""
        mock_cr = MagicMock()
        mock_sc = MagicMock()

        adata = _make_mock_adata(n_cells=50, has_velocity=False)
        mock_sc.read_h5ad.return_value = adata

        mock_ck_class = MagicMock(side_effect=RuntimeError("Kernel failed"))

        with patch.dict("sys.modules", {
            "cellrank": mock_cr,
            "cellrank.kernels": MagicMock(ConnectivityKernel=mock_ck_class),
        }):
            with patch("ct.tools.singlecell._check_scanpy", return_value=mock_sc):
                result = cellrank_fate(data_path="/some/data.h5ad")

        assert "error" in result
        assert "kernel" in result["summary"].lower()

    def test_gpcca_estimator_failure(self):
        """When GPCCA estimator fails, should return error."""
        mock_cr = MagicMock()
        mock_sc = MagicMock()

        adata = _make_mock_adata(n_cells=50, has_velocity=False)
        mock_sc.read_h5ad.return_value = adata

        mock_ck = MagicMock()
        mock_ck_class = MagicMock(return_value=mock_ck)

        mock_gpcca_class = MagicMock(side_effect=RuntimeError("GPCCA failed"))

        with patch.dict("sys.modules", {
            "cellrank": mock_cr,
            "cellrank.kernels": MagicMock(ConnectivityKernel=mock_ck_class),
            "cellrank.estimators": MagicMock(GPCCA=mock_gpcca_class),
        }):
            with patch("ct.tools.singlecell._check_scanpy", return_value=mock_sc):
                result = cellrank_fate(data_path="/some/data.h5ad")

        assert "error" in result
        assert "CellRank" in result["summary"]

    def test_n_macrostates_clamped(self):
        """n_macrostates should be clamped between 2 and 30."""
        mock_cr = MagicMock()
        mock_sc = MagicMock()

        adata = _make_mock_adata(n_cells=50, has_velocity=False)
        mock_sc.read_h5ad.return_value = adata

        mock_ck = MagicMock()
        mock_ck_class = MagicMock(return_value=mock_ck)

        mock_estimator = MagicMock()
        mock_estimator.terminal_states = pd.Series([None] * 50)
        mock_fate = MagicMock()
        mock_fate.names = []
        mock_fate.shape = (50, 0)
        mock_estimator.fate_probabilities = mock_fate
        mock_gpcca_class = MagicMock(return_value=mock_estimator)

        with patch.dict("sys.modules", {
            "cellrank": mock_cr,
            "cellrank.kernels": MagicMock(ConnectivityKernel=mock_ck_class),
            "cellrank.estimators": MagicMock(GPCCA=mock_gpcca_class),
        }):
            with patch("ct.tools.singlecell._check_scanpy", return_value=mock_sc):
                result = cellrank_fate(data_path="/some/data.h5ad", n_macrostates=1)

        # n_macrostates should be clamped to 2
        assert result["n_macrostates"] == 2
        mock_estimator.fit.assert_called_once_with(n_states=2)
