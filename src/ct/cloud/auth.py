"""
CellType Cloud authentication client.

CLI authenticates via device authorization flow:
1. CLI requests a session code from the gateway
2. Opens browser to the web dashboard's /authorize-device page
3. User approves on the web (they're already signed in via WorkOS)
4. CLI polls until approved, receives a long-lived API token
5. Token stored at ~/.ct/auth.json

The API token is tied to the user's WorkOS account. Same user, same credits,
whether accessed via CLI or web dashboard.
"""

import json
import logging
import time
import webbrowser
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger("ct.cloud.auth")

AUTH_FILE = Path.home() / ".ct" / "auth.json"
POLL_INTERVAL = 2.0
POLL_TIMEOUT = 300.0


def _get_api_url() -> str:
    try:
        from ct.agent.config import Config
        cfg = Config.load()
        return str(cfg.get("cloud.endpoint", "https://api.celltype.com")).rstrip("/")
    except Exception:
        return "https://api.celltype.com"


def _load_auth() -> dict:
    if AUTH_FILE.exists():
        try:
            return json.loads(AUTH_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_auth(data: dict) -> None:
    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUTH_FILE.write_text(json.dumps(data, indent=2))
    AUTH_FILE.chmod(0o600)


def login() -> dict:
    """Start device authorization flow. Returns {email, user_id, is_new}."""
    auth_data = _load_auth()
    if auth_data.get("api_token"):
        return {"already_logged_in": True, "email": auth_data.get("email", "unknown")}

    api_url = _get_api_url()

    # Build a descriptive device name
    import platform
    import socket
    device_name = f"{socket.gethostname()} ({platform.system()})"

    # Step 1: Init device session
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{api_url}/auth/device-authorize/init",
            json={"device_name": device_name},
        )
        resp.raise_for_status()
        session = resp.json()

    session_code = session["session_code"]

    # Build the auth URL pointing to the web dashboard (not the API gateway)
    try:
        from ct.agent.config import Config
        cfg = Config.load()
        dashboard_url = str(cfg.get("cloud.dashboard_url", "http://localhost:5173")).rstrip("/")
    except Exception:
        dashboard_url = "http://localhost:5173"

    auth_url = f"{dashboard_url}/authorize-device?code={session_code}"

    return {
        "session_code": session_code,
        "auth_url": auth_url,
        "already_logged_in": False,
    }


def poll_for_approval(session_code: str) -> dict:
    """Poll until device is approved. Returns {api_token, email, user_id}."""
    api_url = _get_api_url()
    start = time.time()

    with httpx.Client(timeout=30) as client:
        while time.time() - start < POLL_TIMEOUT:
            time.sleep(POLL_INTERVAL)
            resp = client.post(
                f"{api_url}/auth/device-authorize",
                json={"session_code": session_code},
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "pending":
                    continue
                # Approved
                auth_result = {
                    "api_token": data["api_token"],
                    "email": data.get("email", ""),
                    "user_id": data.get("user_id", ""),
                }
                _save_auth(auth_result)
                return auth_result
            elif resp.status_code == 202:
                continue
            else:
                raise RuntimeError(f"Authorization failed: {resp.status_code}")

    raise RuntimeError("Authorization timed out.")


def logout() -> bool:
    """Revoke the API token on the server, then delete local auth file."""
    auth_data = _load_auth()
    token = auth_data.get("api_token")

    if token:
        # Tell the gateway to revoke this token
        try:
            api_url = _get_api_url()
            token_id = token[:16]
            with httpx.Client(timeout=10) as client:
                client.delete(
                    f"{api_url}/auth/device-tokens/{token_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
        except Exception:
            pass  # Best-effort — still delete locally

    if AUTH_FILE.exists():
        AUTH_FILE.unlink()
        return True
    return False


def get_token() -> Optional[str]:
    auth_data = _load_auth()
    return auth_data.get("api_token")


def get_user_email() -> Optional[str]:
    return _load_auth().get("email")


def get_user_id() -> Optional[str]:
    return _load_auth().get("user_id")


def is_logged_in() -> bool:
    return bool(_load_auth().get("api_token"))


def check_auth() -> bool:
    """Verify the stored token is still valid with the gateway.

    If the token has been revoked, deletes local auth and returns False.
    Returns True if valid, False if revoked or not logged in.
    Silently returns True if the gateway is unreachable (offline tolerance).
    """
    token = get_token()
    if not token:
        return False

    try:
        api_url = _get_api_url()
        with httpx.Client(timeout=5) as client:
            resp = client.get(
                f"{api_url}/account/credits",
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code == 401:
                # Token revoked — clean up local auth
                logger.info("Token revoked remotely, logging out locally.")
                if AUTH_FILE.exists():
                    AUTH_FILE.unlink()
                return False
            # Any other response (200, 500, etc.) means the token is still recognized
            return True
    except Exception:
        # Gateway unreachable — assume token is fine (offline tolerance)
        return True
