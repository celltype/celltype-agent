"""Tests for Stripe checkout and webhook integration."""

import json
import os
import sys
import time
from unittest.mock import patch, MagicMock

import pytest

# Ensure celltype-cloud is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "celltype-cloud"))


@pytest.fixture(autouse=True)
def _env_setup(monkeypatch):
    """Set up environment for testing."""
    monkeypatch.setenv("CELLTYPE_DEV_MODE", "true")
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_fake")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test_fake")
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:5173")


@pytest.fixture
def _reset_dal():
    """Reset in-memory DAL state between tests."""
    import dal_memory
    dal_memory._users.clear()
    dal_memory._credits.clear()
    dal_memory._transactions.clear()
    dal_memory._jobs.clear()
    dal_memory._tokens.clear()
    dal_memory._audit_log.clear()
    dal_memory._audit_id_counter = 0
    yield
    dal_memory._users.clear()
    dal_memory._credits.clear()
    dal_memory._transactions.clear()
    dal_memory._jobs.clear()
    dal_memory._tokens.clear()
    dal_memory._audit_log.clear()
    dal_memory._audit_id_counter = 0


@pytest.fixture
def client(_reset_dal):
    """Create a FastAPI test client with auth overridden."""
    from fastapi.testclient import TestClient
    import main

    # Override auth dependency to always return test user
    async def _mock_verify_token(request=None):
        return {"user_id": "dev-user", "email": "dev@test.com"}

    main.app.dependency_overrides[main._verify_token] = _mock_verify_token
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


@pytest.fixture
def auth_headers():
    """Headers (auth is mocked, but we still send a bearer for realism)."""
    return {"Authorization": "Bearer dev-test-token"}


# ─── 5.1 Test checkout creates Stripe session with correct parameters ────

class TestCheckoutCreatesStripeSession:

    def test_checkout_creates_session(self, client, auth_headers, monkeypatch):
        """Checkout in prod mode calls stripe.checkout.Session.create with correct params."""
        # Switch to production mode for this test
        import main
        monkeypatch.setattr(main, "DEV_MODE", False)
        monkeypatch.setattr(main, "STRIPE_DASHBOARD_URL", "https://app.celltype.com")

        # Set up user account first (need dev mode temporarily)
        import dal_memory
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            dal_memory.get_or_create_user("dev-user", email="dev@test.com")
        )

        mock_session = MagicMock()
        mock_session.url = "https://checkout.stripe.com/test_session"

        with patch("stripe.checkout.Session.create", return_value=mock_session) as mock_create:
            resp = client.post(
                "/billing/checkout",
                json={"amount": 25},
                headers=auth_headers,
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["url"] == "https://checkout.stripe.com/test_session"

        # Verify stripe was called with correct params
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["mode"] == "payment"
        assert call_kwargs["line_items"][0]["price_data"]["unit_amount"] == 2500
        assert call_kwargs["line_items"][0]["price_data"]["currency"] == "usd"
        assert call_kwargs["metadata"]["user_id"] == "dev-user"
        assert call_kwargs["client_reference_id"] == "dev-user"
        assert "success=true" in call_kwargs["success_url"]
        assert "cancelled=true" in call_kwargs["cancel_url"]


# ─── 5.2 Test checkout rejects invalid amounts ──────────────────────────

class TestCheckoutRejectsInvalidAmounts:

    def test_rejects_invalid_amount(self, client, auth_headers):
        resp = client.post(
            "/billing/checkout",
            json={"amount": 15},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "Amount must be one of" in resp.json()["detail"]

    def test_rejects_zero_amount(self, client, auth_headers):
        resp = client.post(
            "/billing/checkout",
            json={"amount": 0},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_accepts_valid_amounts(self, client, auth_headers, monkeypatch):
        import main
        monkeypatch.setattr(main, "STRIPE_SECRET_KEY", "")
        for amount in [10, 25, 50, 100]:
            resp = client.post(
                "/billing/checkout",
                json={"amount": amount},
                headers=auth_headers,
            )
            # Dev mode without Stripe key → direct credit → 200
            assert resp.status_code == 200, f"Failed for amount={amount}"


# ─── 5.3 Test webhook processes checkout.session.completed ───────────────

class TestWebhookProcessesCheckout:

    def test_webhook_credits_account(self, client, monkeypatch):
        """Valid webhook event credits the user account."""
        import main
        monkeypatch.setattr(main, "DEV_MODE", False)

        # Set up user account
        import dal_memory
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            dal_memory.get_or_create_user("user-stripe-1", email="test@test.com")
        )

        event_data = {
            "id": "evt_test_123",
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "id": "cs_test_session_abc",
                    "amount_total": 2500,
                    "metadata": {"user_id": "user-stripe-1"},
                }
            },
        }

        with patch("stripe.Webhook.construct_event", return_value=event_data):
            resp = client.post(
                "/webhooks/stripe",
                content=json.dumps(event_data),
                headers={"stripe-signature": "t=123,v1=fake_sig"},
            )

        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Verify credits were added (10 starter + 25 purchase = 35)
        assert dal_memory._credits["user-stripe-1"] == 35.0

        # Verify transaction was recorded
        purchase_txs = [
            t for t in dal_memory._transactions
            if t["type"] == "purchase" and t["user_id"] == "user-stripe-1"
        ]
        assert len(purchase_txs) == 1
        assert purchase_txs[0]["amount"] == 25.0
        assert purchase_txs[0]["stripe_session_id"] == "cs_test_session_abc"

        # Verify audit log entry
        audit_entries = [
            e for e in dal_memory._audit_log
            if e["event"] == "credits_purchased"
        ]
        assert len(audit_entries) == 1
        assert audit_entries[0]["detail"]["amount"] == 25.0


# ─── 5.4 Test webhook rejects invalid signature ─────────────────────────

class TestWebhookRejectsInvalidSignature:

    def test_invalid_signature_returns_400(self, client, monkeypatch):
        import main
        monkeypatch.setattr(main, "DEV_MODE", False)

        with patch(
            "stripe.Webhook.construct_event",
            side_effect=stripe_sig_error(),
        ):
            resp = client.post(
                "/webhooks/stripe",
                content=b'{"bad": "payload"}',
                headers={"stripe-signature": "t=123,v1=invalid"},
            )

        assert resp.status_code == 400
        assert "Invalid signature" in resp.json()["detail"]


def stripe_sig_error():
    """Create a stripe SignatureVerificationError."""
    import stripe
    return stripe.SignatureVerificationError("bad sig", "t=123,v1=invalid")


# ─── 5.5 Test webhook idempotency ───────────────────────────────────────

class TestWebhookIdempotency:

    def test_duplicate_session_returns_200_no_double_credit(self, client, monkeypatch):
        """Second webhook with same session ID returns 200 without double-crediting."""
        import main
        monkeypatch.setattr(main, "DEV_MODE", False)

        import dal_memory
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            dal_memory.get_or_create_user("user-idem-1", email="test@test.com")
        )

        event_data = {
            "id": "evt_test_456",
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "id": "cs_test_session_duplicate",
                    "amount_total": 5000,
                    "metadata": {"user_id": "user-idem-1"},
                }
            },
        }

        with patch("stripe.Webhook.construct_event", return_value=event_data):
            # First webhook
            resp1 = client.post(
                "/webhooks/stripe",
                content=json.dumps(event_data),
                headers={"stripe-signature": "t=123,v1=fake_sig"},
            )
            assert resp1.status_code == 200

            # Second webhook (duplicate)
            resp2 = client.post(
                "/webhooks/stripe",
                content=json.dumps(event_data),
                headers={"stripe-signature": "t=123,v1=fake_sig"},
            )
            assert resp2.status_code == 200
            assert "Already processed" in resp2.json().get("message", "")

        # Balance should be 10 (starter) + 50 (one purchase) = 60, NOT 110
        assert dal_memory._credits["user-idem-1"] == 60.0

        # Only one purchase transaction
        purchase_txs = [
            t for t in dal_memory._transactions
            if t["type"] == "purchase" and t["user_id"] == "user-idem-1"
        ]
        assert len(purchase_txs) == 1


# ─── 5.6 Test dev mode ──────────────────────────────────────────────────

class TestDevMode:

    def test_dev_checkout_without_stripe_key_directly_credits(self, client, auth_headers, monkeypatch):
        """Dev mode without Stripe keys directly credits account."""
        import main
        monkeypatch.setattr(main, "STRIPE_SECRET_KEY", "")

        resp = client.post(
            "/billing/checkout",
            json={"amount": 50},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "success=true" in data["url"]
        assert "stripe.com" not in data["url"]

        import dal_memory
        assert dal_memory._credits["dev-user"] == 60.0

    def test_dev_webhook_without_secret_returns_200_noop(self, client, monkeypatch):
        """Dev mode without webhook secret returns 200 no-op."""
        import main
        monkeypatch.setattr(main, "STRIPE_WEBHOOK_SECRET", "")

        resp = client.post(
            "/webhooks/stripe",
            content=b'{}',
            headers={"stripe-signature": ""},
        )
        assert resp.status_code == 200
        assert resp.json()["message"] == "Dev mode: webhook ignored"

        import dal_memory
        purchase_txs = [t for t in dal_memory._transactions if t["type"] == "purchase"]
        assert len(purchase_txs) == 0
