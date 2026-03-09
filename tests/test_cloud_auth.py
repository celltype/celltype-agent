"""Tests for CellType Cloud authentication client (API token model)."""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def tmp_auth_file(tmp_path):
    """Use a temporary auth file for tests."""
    auth_file = tmp_path / "auth.json"
    with patch("ct.cloud.auth.AUTH_FILE", auth_file):
        yield auth_file


class TestAuthStorage:
    def test_save_and_load(self, tmp_auth_file):
        from ct.cloud.auth import _save_auth, _load_auth

        data = {"api_token": "ct_test123", "email": "test@example.com", "user_id": "user_abc"}
        _save_auth(data)

        loaded = _load_auth()
        assert loaded["api_token"] == "ct_test123"
        assert loaded["email"] == "test@example.com"

    def test_load_missing_file(self, tmp_auth_file):
        from ct.cloud.auth import _load_auth
        assert _load_auth() == {}

    def test_load_corrupt_file(self, tmp_auth_file):
        from ct.cloud.auth import _load_auth
        tmp_auth_file.write_text("not json")
        assert _load_auth() == {}


class TestGetToken:
    def test_get_token_returns_api_token(self, tmp_auth_file):
        from ct.cloud.auth import _save_auth, get_token

        _save_auth({"api_token": "ct_mytoken"})
        assert get_token() == "ct_mytoken"

    def test_get_token_no_auth(self, tmp_auth_file):
        from ct.cloud.auth import get_token
        assert get_token() is None


class TestLogout:
    def test_logout_when_logged_in(self, tmp_auth_file):
        from ct.cloud.auth import _save_auth, logout

        _save_auth({"api_token": "ct_test"})
        assert logout() is True
        assert not tmp_auth_file.exists()

    def test_logout_when_not_logged_in(self, tmp_auth_file):
        from ct.cloud.auth import logout
        assert logout() is False


class TestIsLoggedIn:
    def test_logged_in(self, tmp_auth_file):
        from ct.cloud.auth import _save_auth, is_logged_in

        _save_auth({"api_token": "ct_test"})
        assert is_logged_in() is True

    def test_not_logged_in(self, tmp_auth_file):
        from ct.cloud.auth import is_logged_in
        assert is_logged_in() is False


class TestGetUserEmail:
    def test_get_email(self, tmp_auth_file):
        from ct.cloud.auth import _save_auth, get_user_email

        _save_auth({"api_token": "ct_t", "email": "user@test.com"})
        assert get_user_email() == "user@test.com"

    def test_no_email(self, tmp_auth_file):
        from ct.cloud.auth import get_user_email
        assert get_user_email() is None


class TestLogin:
    def test_login_already_logged_in(self, tmp_auth_file):
        from ct.cloud.auth import _save_auth, login

        _save_auth({"api_token": "ct_existing", "email": "me@test.com"})
        result = login()
        assert result["already_logged_in"] is True
        assert result["email"] == "me@test.com"

    def test_login_returns_session_code(self, tmp_auth_file):
        from ct.cloud.auth import login

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "session_code": "ABCD1234",
            "auth_url": "/authorize-device?code=ABCD1234",
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("ct.cloud.auth.httpx.Client") as MockClient:
            mock_http = MagicMock()
            mock_http.post.return_value = mock_resp
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_http)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            result = login()
            assert result["session_code"] == "ABCD1234"
            assert result["already_logged_in"] is False
            assert "authorize-device" in result["auth_url"]


class TestPollForApproval:
    def test_poll_receives_token(self, tmp_auth_file):
        from ct.cloud.auth import poll_for_approval

        # First call: pending (202), second call: approved (200)
        pending_resp = MagicMock()
        pending_resp.status_code = 202

        approved_resp = MagicMock()
        approved_resp.status_code = 200
        approved_resp.json.return_value = {
            "api_token": "ct_newtoken",
            "email": "user@test.com",
            "user_id": "user_123",
        }

        with patch("ct.cloud.auth.httpx.Client") as MockClient:
            mock_http = MagicMock()
            mock_http.post.side_effect = [pending_resp, approved_resp]
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_http)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            with patch("ct.cloud.auth.time.sleep"):
                result = poll_for_approval("ABCD1234")

        assert result["api_token"] == "ct_newtoken"
        assert result["email"] == "user@test.com"

        # Should have been saved to disk
        loaded = json.loads(tmp_auth_file.read_text())
        assert loaded["api_token"] == "ct_newtoken"
