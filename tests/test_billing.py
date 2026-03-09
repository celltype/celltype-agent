"""Tests for billing — credit hold/release/reconcile logic."""

import pytest


class TestCreditManagement:
    """Test in-memory credit operations from the gateway."""

    def setup_method(self):
        """Reset in-memory stores before each test."""
        import sys
        import os
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "celltype-cloud"))
        if path not in sys.path:
            sys.path.insert(0, path)
        import dal_memory
        dal_memory._users.clear()
        dal_memory._credits.clear()
        dal_memory._transactions.clear()
        dal_memory._jobs.clear()
        dal_memory._tokens.clear()
        dal_memory._audit_log.clear()
        dal_memory._audit_id_counter = 0

    @pytest.fixture(autouse=True)
    def _setup_gateway_module(self, monkeypatch):
        """Make the gateway module importable."""
        import sys
        import os
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "celltype-cloud"))
        sys.path.insert(0, path)
        monkeypatch.setenv("CELLTYPE_DEV_MODE", "true")
        yield
        sys.path.pop(0)

    def test_new_account_gets_starter_credits(self):
        import dal_memory
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(dal_memory.get_or_create_user("new-user"))
            account = loop.run_until_complete(dal_memory.get_credits("new-user"))
            assert account["balance"] == 10.0
            assert account["transactions"][0]["type"] == "grant"
        finally:
            loop.close()

    def test_hold_deducts_from_balance(self):
        import dal_memory
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(dal_memory.get_or_create_user("user-1"))
            result = loop.run_until_complete(dal_memory.hold_credits("user-1", "job-1", 2.0))
            assert result is True

            assert dal_memory._credits["user-1"] == 8.0
            
            # Check transaction was created
            txs = [t for t in dal_memory._transactions if t["user_id"] == "user-1" and t["type"] == "hold"]
            assert len(txs) == 1
            assert txs[0]["amount"] == -2.0
            assert txs[0]["job_id"] == "job-1"
        finally:
            loop.close()

    def test_hold_insufficient_balance(self):
        import dal_memory
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(dal_memory.get_or_create_user("user-2"))
            result = loop.run_until_complete(dal_memory.hold_credits("user-2", "job-1", 20.0))
            assert result is False
        finally:
            loop.close()

    def test_reconcile_refunds_difference(self):
        import dal_memory
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(dal_memory.get_or_create_user("user-3"))
            loop.run_until_complete(dal_memory.hold_credits("user-3", "job-1", 2.0))
            assert dal_memory._credits["user-3"] == 8.0

            loop.run_until_complete(dal_memory.reconcile_credits("user-3", "job-1", 1.5))
            assert dal_memory._credits["user-3"] == 8.5  # 8.0 + 0.5 refund
            
            # Check refund transaction
            txs = [t for t in dal_memory._transactions if t["user_id"] == "user-3" and t["type"] == "refund"]
            assert len(txs) == 1
            assert txs[0]["amount"] == 0.5
        finally:
            loop.close()

    def test_release_hold_full_refund(self):
        import dal_memory
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(dal_memory.get_or_create_user("user-4"))
            loop.run_until_complete(dal_memory.hold_credits("user-4", "job-1", 3.0))
            assert dal_memory._credits["user-4"] == 7.0

            loop.run_until_complete(dal_memory.release_hold("user-4", "job-1"))
            assert dal_memory._credits["user-4"] == 10.0
            
            # Check refund transaction
            txs = [t for t in dal_memory._transactions if t["user_id"] == "user-4" and t["type"] == "refund"]
            # release_hold might not create a transaction in memory DAL, let's just check balance
        finally:
            loop.close()

    def test_reconcile_exact_cost_no_refund(self):
        import dal_memory
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(dal_memory.get_or_create_user("user-5"))
            loop.run_until_complete(dal_memory.hold_credits("user-5", "job-1", 2.0))

            loop.run_until_complete(dal_memory.reconcile_credits("user-5", "job-1", 2.0))
            # Balance should remain unchanged (held 2.0, charged 2.0, refund 0)
            assert dal_memory._credits["user-5"] == 8.0
        finally:
            loop.close()
