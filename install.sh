#!/usr/bin/env bash
# celltype-agent — One-liner installer
# Usage: curl -fsSL https://raw.githubusercontent.com/celltype/cli/main/install.sh | bash
set -euo pipefail

PACKAGE="celltype-agent"
REPO_URL="git+https://github.com/celltype/cli.git"
MIN_PYTHON="3.10"

# ── Helpers ────────────────────────────────────────────────────

info()  { printf '\033[0;36m%s\033[0m\n' "$*"; }
ok()    { printf '\033[0;32m%s\033[0m\n' "$*"; }
warn()  { printf '\033[0;33m%s\033[0m\n' "$*" >&2; }
fail()  { printf '\033[0;31mError: %s\033[0m\n' "$*" >&2; exit 1; }

# ── Detect Python ──────────────────────────────────────────────

PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done
[ -n "$PYTHON" ] || fail "Python not found. Install Python ${MIN_PYTHON}+ and try again."

# Version check
PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    fail "Python ${MIN_PYTHON}+ required (found ${PY_VERSION})"
fi
ok "Found Python ${PY_VERSION}"

# ── Detect OS ──────────────────────────────────────────────────

OS="$(uname -s)"
case "$OS" in
    Darwin) info "Detected macOS" ;;
    Linux)  info "Detected Linux" ;;
    *)      warn "Unsupported OS: $OS — installation may still work" ;;
esac

# ── Install package ────────────────────────────────────────────

INSTALL_SPEC="${PACKAGE} @ ${REPO_URL}"

if command -v uv >/dev/null 2>&1; then
    info "Installing ${PACKAGE} via uv..."
    uv tool install "$INSTALL_SPEC" || uv tool install --force "$INSTALL_SPEC"
    ok "Installed with uv"
elif command -v pipx >/dev/null 2>&1; then
    info "Installing ${PACKAGE} via pipx..."
    pipx install "$REPO_URL" || pipx install --force "$REPO_URL"
    ok "Installed with pipx"
else
    warn "uv/pipx not found — falling back to pip install --user"
    "$PYTHON" -m pip install --user --upgrade "$INSTALL_SPEC"
    ok "Installed with pip"

    # Check if user bin is on PATH
    USER_BIN=$("$PYTHON" -m site --user-base)/bin
    if ! echo "$PATH" | tr ':' '\n' | grep -qx "$USER_BIN"; then
        warn ""
        warn "Add this to your shell profile (~/.bashrc, ~/.zshrc, etc.):"
        warn "  export PATH=\"${USER_BIN}:\$PATH\""
        warn ""
    fi
fi

# ── Verify ct is available ─────────────────────────────────────

if ! command -v ct >/dev/null 2>&1; then
    warn "'ct' not found on PATH. You may need to restart your shell."
    warn "Then run: ct setup"
    exit 0
fi

ok "ct $(ct --version 2>/dev/null || echo '(installed)')"

# ── Run setup wizard ───────────────────────────────────────────

info ""
info "Running setup wizard..."
info ""
ct setup
