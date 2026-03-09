"""Tests for repurposing.txgnn_predict tool."""

import pytest
from unittest.mock import patch, MagicMock
import sys


class TestTxGNNPredict:
    def test_empty_disease(self):
        from ct.tools.repurposing import txgnn_predict
        result = txgnn_predict(disease="")
        assert "error" in result

    def test_not_installed(self):
        from ct.tools.repurposing import txgnn_predict
        # txgnn is not actually installed, so this should return install instructions
        result = txgnn_predict(disease="Alzheimer disease")
        assert "not installed" in result.get("summary", "").lower() or "error" in result

    def test_successful_prediction(self):
        mock_model = MagicMock()
        mock_model.predict_drug_disease.return_value = [
            {"drug_name": "Metformin", "score": 0.85},
            {"drug_name": "Rapamycin", "score": 0.72},
        ]
        mock_txdata = MagicMock()
        mock_txgnn_cls = MagicMock(return_value=mock_model)
        mock_txdata_cls = MagicMock(return_value=mock_txdata)
        mock_module = MagicMock()
        mock_module.TxGNN = mock_txgnn_cls
        mock_module.TxData = mock_txdata_cls

        with patch.dict("sys.modules", {"txgnn": mock_module}):
            from ct.tools.repurposing import txgnn_predict
            result = txgnn_predict(disease="Alzheimer disease")
        assert result["n_predictions"] == 2
        assert "Metformin" in result["summary"]

    def test_drug_filter(self):
        mock_model = MagicMock()
        mock_model.predict_drug_disease.return_value = [
            {"drug_name": "Metformin", "score": 0.85},
            {"drug_name": "Rapamycin", "score": 0.72},
            {"drug_name": "Aspirin", "score": 0.5},
        ]
        mock_txdata = MagicMock()
        mock_module = MagicMock()
        mock_module.TxGNN = MagicMock(return_value=mock_model)
        mock_module.TxData = MagicMock(return_value=mock_txdata)

        with patch.dict("sys.modules", {"txgnn": mock_module}):
            from ct.tools.repurposing import txgnn_predict
            result = txgnn_predict(disease="Alzheimer disease", drug_filter="metformin")
        assert result["n_predictions"] == 1

    def test_top_k(self):
        mock_model = MagicMock()
        mock_model.predict_drug_disease.return_value = [
            {"drug_name": f"Drug{i}", "score": 1.0 - i * 0.1} for i in range(10)
        ]
        mock_txdata = MagicMock()
        mock_module = MagicMock()
        mock_module.TxGNN = MagicMock(return_value=mock_model)
        mock_module.TxData = MagicMock(return_value=mock_txdata)

        with patch.dict("sys.modules", {"txgnn": mock_module}):
            from ct.tools.repurposing import txgnn_predict
            result = txgnn_predict(disease="cancer", top_k=5)
        assert result["n_predictions"] == 5

    def test_none_disease(self):
        from ct.tools.repurposing import txgnn_predict
        result = txgnn_predict(disease=None)
        assert "error" in result
