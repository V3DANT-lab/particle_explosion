#!/usr/bin/env bash
# run.sh - convenience wrapper to launch the demo using the project venv.
#
# Usage:
#   ./run.sh                       # default: camera 0, 1280x720
#   ./run.sh --camera 1            # use camera index 1
#   ./run.sh --width 1920 --height 1080

set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR=".venv"
if [[ ! -d "$VENV_DIR" ]]; then
    echo "[run] venv not found. Run ./setup.sh first." >&2
    exit 1
fi

exec "$VENV_DIR/bin/python" main.py "$@"
