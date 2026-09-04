@echo off
chcp 65001 >nul
title Prognoza Sprzedaży Lodów

echo.
echo  ============================================
echo   Prognoza Sprzedazy Lodow — Uruchamianie
echo  ============================================
echo.

:: Sprawdz czy Python jest zainstalowany
python --version >nul 2>&1
if errorlevel 1 (
    echo  [BLAD] Python nie jest zainstalowany lub nie jest dostepny w PATH.
    echo.
    echo  Pobierz Python ze strony:
    echo    https://www.python.org/downloads/
    echo.
    echo  Podczas instalacji zaznacz opcje "Add Python to PATH"!
    echo.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo  [OK] Znaleziono Python %PY_VER%

:: Stworz wirtualne srodowisko jesli nie istnieje
if not exist "venv\" (
    echo  [INFO] Tworzenie wirtualnego srodowiska...
    python -m venv venv
    if errorlevel 1 (
        echo  [BLAD] Nie udalo sie utworzyc venv.
        pause
        exit /b 1
    )
    echo  [OK] Wirtualne srodowisko utworzone.
) else (
    echo  [OK] Wirtualne srodowisko juz istnieje.
)

:: Aktywuj venv
call venv\Scripts\activate.bat

:: Zainstaluj/zaktualizuj zaleznosci
echo  [INFO] Sprawdzanie zaleznosci (moze zajac chwile przy pierwszym uruchomieniu)...
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo  [BLAD] Nie udalo sie zainstalowac zaleznosci.
    echo  Sprawdz polaczenie z internetem i sprobuj ponownie.
    pause
    exit /b 1
)
echo  [OK] Wszystkie zaleznosci zainstalowane.

:: Uruchom aplikacje
echo.
echo  [INFO] Uruchamianie aplikacji...
echo  [INFO] Aplikacja otworzy sie w przegladarce: http://localhost:8501
echo.
echo  Aby zatrzymac aplikacje, zamknij to okno lub nacisnij Ctrl+C
echo.

:: Otworz przegladarke po 3 sekundach (w tle)
start /b cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8501"

:: Uruchom Streamlit
python -m streamlit run app_prognoza_lodow.py --server.headless true --server.port 8501

pause
