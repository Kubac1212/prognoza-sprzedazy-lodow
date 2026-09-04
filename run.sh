#!/bin/bash

# Prognoza Sprzedaży Lodów — skrypt uruchamiający (macOS / Linux)

set -e

echo ""
echo " ============================================"
echo "  Prognoza Sprzedaży Lodów — Uruchamianie"
echo " ============================================"
echo ""

# Katalog skryptu (uruchamiaj zawsze z właściwego miejsca)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Sprawdź Python (szukaj python3 lub python)
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo " [BŁĄD] Python nie jest zainstalowany."
    echo ""
    echo " Pobierz Python ze strony:"
    echo "   https://www.python.org/downloads/"
    echo ""
    echo " macOS — możesz też zainstalować przez Homebrew:"
    echo "   brew install python3"
    echo ""
    exit 1
fi

PY_VER=$($PYTHON --version 2>&1)
echo " [OK] Znaleziono $PY_VER"

# Stwórz wirtualne środowisko jeśli nie istnieje
if [ ! -d "venv" ]; then
    echo " [INFO] Tworzenie wirtualnego środowiska..."
    $PYTHON -m venv venv
    echo " [OK] Wirtualne środowisko utworzone."
else
    echo " [OK] Wirtualne środowisko już istnieje."
fi

# Aktywuj venv
source venv/bin/activate

# Zainstaluj zależności
echo " [INFO] Sprawdzanie zależności (może chwilę trwać przy pierwszym uruchomieniu)..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo " [OK] Wszystkie zależności zainstalowane."

# Otwórz przeglądarkę po 3 sekundach (w tle)
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
echo " [INFO] Uruchamianie aplikacji..."
echo " [INFO] Aplikacja otworzy się w przeglądarce: http://localhost:8501"
echo ""
echo " Aby zatrzymać aplikację, naciśnij Ctrl+C"
echo ""

# Uruchom Streamlit
python -m streamlit run app_prognoza_lodow.py --server.headless true --server.port 8501
