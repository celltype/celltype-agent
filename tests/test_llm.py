"""Tests for LLM client: usage tracking, cost estimation, streaming."""

import pytest
from unittest.mock import MagicMock, patch
from ct.models.llm import LLMClient, LLMResponse, UsageTracker, MODEL_PRICING


class TestUsageTracker:
    def test_empty_tracker(self):
        tracker = UsageTracker()
        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.total_tokens == 0
        assert tracker.total_cost == 0.0

    def test_record_usage(self):
        tracker = UsageTracker()
        tracker.record("claude-sonnet-4-5-20250929", {"input": 1000, "output": 500})
        assert tracker.total_input_tokens == 1000
        assert tracker.total_output_tokens == 500
        assert tracker.total_tokens == 1500

    def test_cost_estimation(self):
        tracker = UsageTracker()
        # Sonnet: $3/M input, $15/M output
        tracker.record("claude-sonnet-4-5-20250929", {"input": 1_000_000, "output": 1_000_000})
        assert tracker.total_cost == pytest.approx(18.0)  # $3 + $15

    def test_haiku_cheaper_than_sonnet(self):
        t1 = UsageTracker()
        t1.record("claude-sonnet-4-5-20250929", {"input": 1000, "output": 500})

        t2 = UsageTracker()
        t2.record("claude-haiku-4-5-20251001", {"input": 1000, "output": 500})

        assert t2.total_cost < t1.total_cost

    def test_multiple_calls(self):
        tracker = UsageTracker()
        tracker.record("claude-sonnet-4-5-20250929", {"input": 500, "output": 200})
        tracker.record("claude-sonnet-4-5-20250929", {"input": 300, "output": 100})
        assert len(tracker.calls) == 2
        assert tracker.total_input_tokens == 800
        assert tracker.total_output_tokens == 300

    def test_summary_format(self):
        tracker = UsageTracker()
        tracker.record("claude-sonnet-4-5-20250929", {"input": 1000, "output": 500})
        s = tracker.summary()
        assert "1 LLM calls" in s
        assert "1,000 in" in s
        assert "500 out" in s
        assert "$" in s
        assert "sonnet" in s

    def test_reset(self):
        tracker = UsageTracker()
        tracker.record("claude-sonnet-4-5-20250929", {"input": 100, "output": 50})
        tracker.reset()
        assert tracker.total_tokens == 0
        assert len(tracker.calls) == 0

    def test_claude_list_prices(self):
        tracker = UsageTracker()
        tracker.record("claude-opus-4-6", {"input": 1_000_000, "output": 1_000_000})
        tracker.record("claude-haiku-4-5-20251001", {"input": 1_000_000, "output": 1_000_000})
        assert tracker.total_cost == pytest.approx(30.0 + 6.0)

    def test_model_outside_table_uses_list_price(self):
        tracker = UsageTracker()
        tracker.record("claude-opus-4-7", {"input": 1_000_000, "output": 0})
        assert tracker.total_cost == pytest.approx(5.0)

    def test_unknown_model_zero_cost(self):
        tracker = UsageTracker()
        tracker.record("unknown-model-xyz", {"input": 1000, "output": 500})
        assert tracker.total_cost == 0.0
        assert tracker.total_tokens == 1500

    def test_no_usage_skipped(self):
        tracker = UsageTracker()
        tracker.record("claude-sonnet-4-5-20250929", None)
        assert len(tracker.calls) == 0


class TestLLMClientUsageTracking:
    def test_chat_records_usage(self):
        client = LLMClient(provider="anthropic", model="claude-sonnet-4-5-20250929")

        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_anthropic.messages.create.return_value = mock_response
        client._client = mock_anthropic

        resp = client.chat(system="test", messages=[{"role": "user", "content": "hi"}])

        assert resp.content == "Hello"
        assert len(client.usage.calls) == 1
        assert client.usage.total_input_tokens == 100
        assert client.usage.total_output_tokens == 50
        assert client.usage.total_cost > 0

    def test_multiple_chats_accumulate(self):
        client = LLMClient(provider="anthropic", model="claude-sonnet-4-5-20250929")

        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hi")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_anthropic.messages.create.return_value = mock_response
        client._client = mock_anthropic

        client.chat(system="test", messages=[{"role": "user", "content": "1"}])
        client.chat(system="test", messages=[{"role": "user", "content": "2"}])

        assert len(client.usage.calls) == 2
        assert client.usage.total_input_tokens == 200


class TestModelPricing:
    def test_known_models_have_pricing(self):
        expected = [
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-6",
            "gpt-4o",
            "gpt-4o-mini",
        ]
        for model in expected:
            assert model in MODEL_PRICING, f"Missing pricing for {model}"
            assert "input" in MODEL_PRICING[model]
            assert "output" in MODEL_PRICING[model]
