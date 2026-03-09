"""Tests for protein.esmc_embed tool."""

import pytest
from unittest.mock import patch, MagicMock
import sys
import numpy as np


class TestEsmcEmbed:
    def test_empty_sequence(self):
        from ct.tools.protein import esmc_embed
        result = esmc_embed(sequence="")
        assert "error" in result
        assert "no sequence" in result["summary"].lower()

    def test_invalid_characters(self):
        from ct.tools.protein import esmc_embed
        result = esmc_embed(sequence="MKTL123")
        assert "error" in result
        assert "Invalid" in result["summary"] or "invalid" in result["summary"].lower()

    def test_too_long_sequence(self):
        from ct.tools.protein import esmc_embed
        result = esmc_embed(sequence="M" * 2049)
        assert "error" in result
        assert "2048" in result["summary"]

    def test_not_installed(self):
        """When neither ESM-C nor ESM-2 is available, returns install instructions."""
        from ct.tools.protein import esmc_embed
        # Neither esm (ESM-C) nor fair-esm (ESM-2) are installed in test env
        result = esmc_embed(sequence="MKTLLILAVL")
        assert "error" in result or "embedding_dim" in result
        # If error, should mention installation
        if "error" in result:
            assert "install" in result.get("summary", "").lower() or "not" in result.get("summary", "").lower()

    def test_successful_esmc_embedding(self):
        """Mock ESM-C to test successful embedding path."""
        mock_embedding = np.random.randn(1, 10, 320).astype(np.float32)

        mock_protein_tensor = MagicMock()
        mock_protein_tensor.embeddings = mock_embedding

        mock_esmc_model = MagicMock()
        mock_esmc_model.encode.return_value = mock_protein_tensor

        mock_ESMC = MagicMock()
        mock_ESMC.from_pretrained.return_value = mock_esmc_model

        mock_ESMProtein = MagicMock()

        mock_esm_models_esmc = MagicMock()
        mock_esm_models_esmc.ESMC = mock_ESMC

        mock_esm_sdk_api = MagicMock()
        mock_esm_sdk_api.ESMProtein = mock_ESMProtein

        with patch.dict("sys.modules", {
            "esm": MagicMock(),
            "esm.models": MagicMock(),
            "esm.models.esmc": mock_esm_models_esmc,
            "esm.sdk": MagicMock(),
            "esm.sdk.api": mock_esm_sdk_api,
        }):
            from ct.tools.protein import esmc_embed
            result = esmc_embed(sequence="MKTLLILAVL")

        assert "error" not in result
        assert result["model"] == "esmc_300m"
        assert result["sequence_length"] == 10
        assert result["embedding_dim"] == 320
        assert "embedding" in result
        assert "embedding_norm" in result
