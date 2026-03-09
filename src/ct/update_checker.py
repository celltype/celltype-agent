"""
Non-blocking CLI update checker with 24-hour caching.

Checks PyPI for the latest version of celltype-cli and displays
a one-liner notification if the installed version is outdated.
"""

import json
import threading
import time
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("ct.update_checker")

PYPI_URL = "https://pypi.org/pypi/celltype-cli/json"
CACHE_FILE = Path.home() / ".ct" / "update_check.json"
CHECK_INTERVAL = 86400  # 24 hours in seconds
REQUEST_TIMEOUT = 3  # seconds — keep it fast


def _parse_version(v: str) -> Tuple[int, ...]:
    """Parse a version string like '0.1.3' into a comparable tuple."""
    try:
        return tuple(int(x) for x in v.strip().lstrip("v").split("."))
    except (ValueError, AttributeError):
        return (0,)


def _read_cache() -> Optional[dict]:
    try:
        if CACHE_FILE.exists():
            data = json.loads(CACHE_FILE.read_text())
            if time.time() - data.get("checked_at", 0) < CHECK_INTERVAL:
                return data
    except Exception:
        pass
    return None


def _write_cache(latest_version: str) -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps({
            "latest_version": latest_version,
            "checked_at": time.time(),
        }))
    except Exception:
        pass


def _fetch_latest_version() -> Optional[str]:
    """Fetch the latest version from PyPI. Returns None on any failure."""
    try:
        import urllib.request
        req = urllib.request.Request(PYPI_URL, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read())
            return data["info"]["version"]
    except Exception:
        return None


def _check_and_cache() -> Optional[str]:
    """Check PyPI (or cache) and return the latest version string, or None."""
    cached = _read_cache()
    if cached:
        return cached.get("latest_version")

    latest = _fetch_latest_version()
    if latest:
        _write_cache(latest)
    return latest


_update_result: Optional[str] = None


def _background_check(current_version: str) -> None:
    """Run the version check and store the result for later retrieval."""
    global _update_result
    try:
        latest = _check_and_cache()
        if latest and _parse_version(latest) > _parse_version(current_version):
            _update_result = latest
    except Exception:
        pass


def start_check(current_version: str) -> None:
    """Kick off a non-blocking update check in a background daemon thread."""
    t = threading.Thread(target=_background_check, args=(current_version,), daemon=True)
    t.start()


def get_update_message() -> Optional[str]:
    """
    Return a formatted update message if a newer version was found,
    or None if the user is up to date (or the check hasn't finished yet).

    Call this after start_check() and a brief delay (e.g. after CLI init).
    """
    from ct import __version__
    if _update_result is None:
        return None
    return (
        f"[yellow]Update available:[/yellow] [dim]{__version__}[/dim] → "
        f"[green bold]{_update_result}[/green bold]  "
        f"[dim]Run:[/dim] [bold]pip install --upgrade celltype-cli[/bold]"
    )
