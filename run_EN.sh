#!/bin/bash

# Ice Cream Sales Forecasting — startup script (macOS / Linux)

set -e

echo ""
echo " ============================================"
echo "  Ice Cream Sales Forecasting — Starting"
echo " ============================================"
echo ""

# Script directory (always run from correct location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for Python (look for python3 or python)
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo " [ERROR] Python is not installed."
    echo ""
    echo " Download Python from:"
    echo "   https://www.python.org/downloads/"
    echo ""
    echo " macOS — you can also install via Homebrew:"
    echo "   brew install python3"
    echo ""
    exit 1
fi

PY_VER=$($PYTHON --version 2>&1)
echo " [OK] Found $PY_VER"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo " [INFO] Creating virtual environment..."
    $PYTHON -m venv venv
    echo " [OK] Virtual environment created."
else
    echo " [OK] Virtual environment already exists."
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo " [INFO] Checking dependencies (may take a while on first run)..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo " [OK] All dependencies installed."

# Open browser after 3 seconds (background)
(
    sleep 3
    URL="http://localhost:8501"
    if command -v xdg-open &>/dev/null; then
        xdg-open "$URL"          # Linux
    elif command -v open &>/dev/null; then
        open "$URL"              # macOS
    fi
) &

echo ""
echo " [INFO] Starting application..."
echo " [INFO] App will open in browser: http://localhost:8501"
echo ""
echo " To stop the application, press Ctrl+C"
echo ""

# Run Streamlit
python -m streamlit run app_ice_cream_sales.py --server.headless true --server.port 8501
