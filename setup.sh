#!/usr/bin/env bash
# setup.sh - create a Python virtual environment, install dependencies and
# fetch the MediaPipe hand-tracking model for the particle explosion demo.
#
# Usage:
#   ./setup.sh            # create .venv, install deps, fetch the model
#   ./setup.sh --force    # delete .venv first, then recreate
#
# After running this, run the demo with:
#   ./run.sh
#
# Or directly:
#   .venv/bin/python main.py

set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"
MODEL_DIR="models"
MODEL_FILE="$MODEL_DIR/hand_landmarker.task"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# --force: delete the existing venv first
if [[ "${1:-}" == "--force" ]]; then
    echo "[setup] removing existing $VENV_DIR ..."
    rm -rf "$VENV_DIR"
fi

if [[ -d "$VENV_DIR" ]]; then
    echo "[setup] venv already exists at $VENV_DIR (use --force to recreate)"
else
    echo "[setup] creating venv at $VENV_DIR with $PYTHON ..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

echo "[setup] upgrading pip ..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip --quiet

echo "[setup] installing requirements ..."
"$VENV_DIR/bin/pip" install -r requirements.txt

# ---------------------------------------------------------------------------
# MediaPipe hand-tracking model (required by the Tasks API used in
# hand_tracker.py). The app still runs without it, just without gestures.
# ---------------------------------------------------------------------------
if [[ -f "$MODEL_FILE" ]]; then
    echo "[setup] hand model already present at $MODEL_FILE"
else
    echo "[setup] downloading hand-tracking model -> $MODEL_FILE ..."
    mkdir -p "$MODEL_DIR"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "$MODEL_FILE" "$MODEL_URL"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$MODEL_FILE" "$MODEL_URL"
    else
        echo "[setup] ERROR: neither curl nor wget found." >&2
        echo "[setup] Download the model manually and save it to $MODEL_FILE:" >&2
        echo "         $MODEL_URL" >&2
        exit 1
    fi
    echo "[setup] model downloaded ($(du -h "$MODEL_FILE" | cut -f1))."
fi

echo
echo "[setup] done. Now run:"
echo "    ./run.sh"
echo
echo "Or activate the venv first:"
echo "    source .venv/bin/activate"
echo "    python main.py"
