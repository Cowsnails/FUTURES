#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# FUTURES Charting Application — One-Click Launcher
# Double-click this file (or run from terminal) to start.
# Handles venv creation, dependency install, and server launch.
# ═══════════════════════════════════════════════════════════════════

set -e

# cd to script directory regardless of where it's launched from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/venv"
PYTHON=""
PORT=8000
LOG_FILE="$SCRIPT_DIR/futures.log"

echo "════════════════════════════════════════════════════════"
echo "  FUTURES Charting Application"
echo "════════════════════════════════════════════════════════"

# ── Step 1: Find Python ──
find_python() {
    for cmd in python3 python; do
        if command -v "$cmd" &>/dev/null; then
            ver=$("$cmd" --version 2>&1 | grep -oP '\d+\.\d+')
            major=$(echo "$ver" | cut -d. -f1)
            minor=$(echo "$ver" | cut -d. -f2)
            if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
                PYTHON="$cmd"
                return 0
            fi
        fi
    done
    echo "ERROR: Python 3.9+ required but not found."
    echo "Install Python: https://www.python.org/downloads/"
    read -p "Press Enter to exit..."
    exit 1
}

# ── Step 2: Setup venv if needed ──
setup_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo "  [1/3] Creating virtual environment..."
        $PYTHON -m venv "$VENV_DIR"
        echo "  [2/3] Installing dependencies (first run only)..."
        "$VENV_DIR/bin/pip" install --upgrade pip -q
        "$VENV_DIR/bin/pip" install -r requirements.txt -q
        echo "  [3/3] Setup complete!"
        echo ""
    else
        # Check if requirements changed
        if [ requirements.txt -nt "$VENV_DIR/.deps_installed" ]; then
            echo "  Updating dependencies..."
            "$VENV_DIR/bin/pip" install -r requirements.txt -q
            touch "$VENV_DIR/.deps_installed"
        fi
    fi
    touch "$VENV_DIR/.deps_installed"
}

# ── Step 3: Kill any existing instance ──
kill_existing() {
    if lsof -i :$PORT -t &>/dev/null; then
        echo "  Stopping existing instance on port $PORT..."
        lsof -i :$PORT -t | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
}

# ── Step 4: Launch ──
launch() {
    echo "  Starting server on http://localhost:$PORT"
    echo "  Log file: $LOG_FILE"
    echo ""
    echo "  Close this window or press Ctrl+C to stop."
    echo "════════════════════════════════════════════════════════"
    echo ""

    # Launch with venv python
    "$VENV_DIR/bin/python" run.py 2>&1 | tee "$LOG_FILE"
}

# ── Run ──
find_python
setup_venv
kill_existing
launch
