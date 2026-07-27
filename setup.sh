#!/usr/bin/env bash
# One-time environment setup for macOS/Linux.
#
# Creates ./.venv and installs pressurekeeper into it (editable, with dev
# extras). After this script finishes, no further setup is needed — just
# invoke the venv's python executables directly, e.g.:
#
#   .venv/bin/pressurekeeper --config config/default.yaml --sim --target 1.0
#
# Usage:
#   ./setup.sh          # dev extras only (matches README's default flow)
#   ./setup.sh --gui     # also install GUI extras (PyQt6/pyqtgraph)

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

INSTALL_GUI=0
for arg in "$@"; do
    case "$arg" in
        --gui)
            INSTALL_GUI=1
            ;;
        -h|--help)
            echo "Usage: $0 [--gui]"
            echo "  --gui   also install GUI extras (PyQt6/pyqtgraph)"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Usage: $0 [--gui]" >&2
            exit 1
            ;;
    esac
done

# Prefer python3.11 explicitly (this project requires >=3.11), then fall
# back to whatever python3/python resolves to, as long as it's new enough.
PYTHON_BIN=""
for candidate in python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        version="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
        if [ -n "$version" ]; then
            major="${version%%.*}"
            minor="${version#*.}"
            if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then
                PYTHON_BIN="$candidate"
                break
            fi
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "Error: could not find a Python 3.11+ interpreter (checked python3.11, python3, python)." >&2
    echo "Install Python 3.11 or newer and re-run this script." >&2
    exit 1
fi

echo "Using $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"

if [ -d ".venv" ]; then
    echo ".venv already exists, reusing it."
else
    "$PYTHON_BIN" -m venv .venv
    echo "Created .venv"
fi

VENV_PY=".venv/bin/python"

# A pre-existing .venv created by a tool other than the stdlib venv module
# (e.g. `uv venv`) may not ship pip. Bootstrap it if missing.
if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
    echo "pip not found in .venv, bootstrapping via ensurepip..."
    "$VENV_PY" -m ensurepip --upgrade
fi

"$VENV_PY" -m pip install --upgrade pip

if [ "$INSTALL_GUI" -eq 1 ]; then
    echo "Installing pressurekeeper with [dev,gui] extras..."
    "$VENV_PY" -m pip install -e ".[dev,gui]"
else
    echo "Installing pressurekeeper with [dev] extras..."
    "$VENV_PY" -m pip install -e ".[dev]"
fi

cat <<'EOF'

Setup complete. From now on, a single command is enough:

  Simulator (no hardware/network required):
    .venv/bin/pressurekeeper --config config/default.yaml --sim --target 1.0

  Dry-run against real APIs (reads real ruby data, never writes to PACE5000):
    .venv/bin/pressurekeeper --config config/default.yaml --target 1.0

  Test suite:
    .venv/bin/pytest -q

EOF

if [ "$INSTALL_GUI" -eq 1 ]; then
    cat <<'EOF'
  GUI (simulator mode):
    .venv/bin/pressurekeeper-gui --config config/default.yaml --sim --target 1.0

EOF
else
    cat <<'EOF'
  GUI extras were not installed. Re-run './setup.sh --gui' to add them, then:
    .venv/bin/pressurekeeper-gui --config config/default.yaml --sim --target 1.0

EOF
fi
