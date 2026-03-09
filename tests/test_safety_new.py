"""Tests for safety.admetai_predict and safety.chemprop_predict."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from ct.tools.safety import admetai_predict, chemprop_predict


# ---------------------------------------------------------------------------
# admetai_predict tests
# ---------------------------------------------------------------------------


class TestAdmetAIPredict:
    """Tests for the ADMET-AI prediction tool."""

    def test_empty_smiles(self):
        """Empty SMILES should return an error."""
        with patch("ct.tools.chemistry._extract_smiles", return_value=""):
            result = admetai_predict(smiles="")
        assert "error" in result
        assert "requires" in result["summary"].lower() or "required" in result["summary"].lower()

    def test_not_installed(self):
        """When admet_ai is not installed, should return install instructions."""
        mock_extract = patch("ct.tools.chemistry._extract_smiles", return_value="CCO")
        with mock_extract:
            # Make `from admet_ai import ADMETModel` fail
            original = sys.modules.get("admet_ai")
            sys.modules["admet_ai"] = None
            try:
                result = admetai_predict(smiles="CCO")
            finally:
                if original is not None:
                    sys.modules["admet_ai"] = original
                else:
                    sys.modules.pop("admet_ai", None)
        assert "error" in result
        assert "not installed" in result["summary"].lower() or "install" in result["summary"].lower()

    def test_successful_prediction_dict(self):
        """Successful prediction returning a dict should be parsed properly."""
        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = {
            "hERG": 0.1,
            "AMES": 0.2,
            "Caco2": 0.95,
        }
        mock_admet_module = MagicMock()
        mock_admet_module.ADMETModel.return_value = mock_model_instance

        with patch("ct.tools.chemistry._extract_smiles", return_value="CCO"):
            with patch.dict("sys.modules", {"admet_ai": mock_admet_module}):
                result = admetai_predict(smiles="CCO")

        assert result["n_endpoints"] == 3
        assert result["verdict"] == "FAVORABLE"
        assert result["smiles"] == "CCO"
        assert len(result["flagged_liabilities"]) == 0

    def test_successful_prediction_dataframe(self):
        """Prediction returning a DataFrame-like object should be handled."""
        mock_df = MagicMock()
        mock_df.to_dict.return_value = {
            "hERG": {0: 0.1},
            "AMES": {0: 0.3},
        }
        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = mock_df
        mock_admet_module = MagicMock()
        mock_admet_module.ADMETModel.return_value = mock_model_instance

        with patch("ct.tools.chemistry._extract_smiles", return_value="CCO"):
            with patch.dict("sys.modules", {"admet_ai": mock_admet_module}):
                result = admetai_predict(smiles="CCO")

        assert result["n_endpoints"] == 2
        assert result["verdict"] == "FAVORABLE"

    def test_flagged_liabilities(self):
        """Predictions exceeding thresholds should be flagged."""
        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = {
            "hERG": 0.8,
            "AMES": 0.9,
            "DILI": 0.7,
            "Caco2": 0.5,
        }
        mock_admet_module = MagicMock()
        mock_admet_module.ADMETModel.return_value = mock_model_instance

        with patch("ct.tools.chemistry._extract_smiles", return_value="CCO"):
            with patch.dict("sys.modules", {"admet_ai": mock_admet_module}):
                result = admetai_predict(smiles="CCO")

        assert result["verdict"] == "UNFAVORABLE"
        assert len(result["flagged_liabilities"]) == 3

    def test_endpoint_filtering(self):
        """When endpoints are specified, only those should be returned."""
        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = {
            "hERG": 0.1,
            "AMES": 0.2,
            "Caco2": 0.95,
            "DILI": 0.3,
        }
        mock_admet_module = MagicMock()
        mock_admet_module.ADMETModel.return_value = mock_model_instance

        with patch("ct.tools.chemistry._extract_smiles", return_value="CCO"):
            with patch.dict("sys.modules", {"admet_ai": mock_admet_module}):
                result = admetai_predict(smiles="CCO", endpoints=["hERG", "AMES"])

        assert result["n_endpoints"] == 2
        assert "hERG" in result["predictions"]
        assert "AMES" in result["predictions"]
        assert "Caco2" not in result["predictions"]

    def test_prediction_failure(self):
        """When model.predict raises, should return error."""
        mock_model_instance = MagicMock()
        mock_model_instance.predict.side_effect = RuntimeError("Model crash")
        mock_admet_module = MagicMock()
        mock_admet_module.ADMETModel.return_value = mock_model_instance

        with patch("ct.tools.chemistry._extract_smiles", return_value="CCO"):
            with patch.dict("sys.modules", {"admet_ai": mock_admet_module}):
                result = admetai_predict(smiles="CCO")

        assert "error" in result
        assert "Model crash" in result["error"]


# ---------------------------------------------------------------------------
# chemprop_predict tests
# ---------------------------------------------------------------------------


class TestChempropPredict:
    """Tests for the Chemprop prediction tool."""

    def test_empty_smiles(self):
        """Empty SMILES should return an error."""
        result = chemprop_predict(smiles="", model_path="/some/model")
        assert "error" in result
        assert "smiles" in result["summary"].lower()

    def test_missing_model_path(self):
        """Missing model_path should return an error."""
        result = chemprop_predict(smiles="CCO")
        assert "error" in result
        assert "model_path" in result["error"]

    def test_not_installed(self):
        """When chemprop is not installed, should return install instructions."""
        original = sys.modules.get("chemprop")
        sys.modules["chemprop"] = None
        try:
            with patch("os.path.exists", return_value=True):
                result = chemprop_predict(smiles="CCO", model_path="/some/model")
        finally:
            if original is not None:
                sys.modules["chemprop"] = original
            else:
                sys.modules.pop("chemprop", None)
        assert "error" in result
        assert "not installed" in result["summary"].lower() or "install" in result["summary"].lower()

    def test_model_path_not_found(self):
        """Non-existent model_path should return an error."""
        mock_chemprop = MagicMock()
        with patch.dict("sys.modules", {"chemprop": mock_chemprop}):
            with patch("os.path.exists", return_value=False):
                result = chemprop_predict(smiles="CCO", model_path="/nonexistent/model")
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_successful_prediction_single(self):
        """Successful single-molecule prediction."""
        mock_chemprop = MagicMock()
        mock_chemprop.predict.return_value = [[0.85]]

        with patch.dict("sys.modules", {"chemprop": mock_chemprop}):
            with patch("os.path.exists", return_value=True):
                result = chemprop_predict(smiles="CCO", model_path="/models/tox")

        assert "summary" in result
        assert result["n_molecules"] == 1
        assert len(result["results"]) == 1
        assert result["results"][0]["prediction"] == 0.85

    def test_successful_prediction_batch(self):
        """Successful batch prediction with comma-separated SMILES."""
        mock_chemprop = MagicMock()
        mock_chemprop.predict.return_value = [[0.85], [0.42]]

        with patch.dict("sys.modules", {"chemprop": mock_chemprop}):
            with patch("os.path.exists", return_value=True):
                result = chemprop_predict(smiles="CCO, c1ccccc1", model_path="/models/tox")

        assert result["n_molecules"] == 2
        assert len(result["results"]) == 2

    def test_prediction_failure(self):
        """When chemprop.predict raises, should return error."""
        mock_chemprop = MagicMock()
        mock_chemprop.predict.side_effect = RuntimeError("Featurization failed")

        with patch.dict("sys.modules", {"chemprop": mock_chemprop}):
            with patch("os.path.exists", return_value=True):
                result = chemprop_predict(smiles="CCO", model_path="/models/tox")

        assert "error" in result
        assert "Featurization failed" in result["error"]
