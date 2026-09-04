@echo off
chcp 65001 >nul
title Ice Cream Sales Forecasting - Running

echo.
echo  ============================================
echo   Ice Cream Sales Forecasting — Starting
echo  ============================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python is not installed or not available in PATH.
    echo.
    echo  Download Python from:
    echo    https://www.python.org/downloads/
    echo.
    echo  During installation, check the "Add Python to PATH" option!
    echo.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo  [OK] Found Python %PY_VER%

:: Create virtual environment if it doesn't exist
if not exist "venv\" (
    echo  [INFO] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo  [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment created.
) else (
    echo  [OK] Virtual environment already exists.
)

:: Activate venv
call venv\Scripts\activate.bat

:: Install/update dependencies
echo  [INFO] Checking dependencies (may take a while on first run)...
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo  [ERROR] Failed to install dependencies.
    echo  Check your internet connection and try again.
    pause
    exit /b 1
)
echo  [OK] All dependencies installed.

:: Run application
echo.
echo  [INFO] Starting application...
echo  [INFO] App will open in browser: http://localhost:8501
echo.
echo  To stop the application, close this window or press Ctrl+C
echo.

:: Open browser after 3 seconds (background)
start /b cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8501"

:: Run Streamlit
python -m streamlit run app_ice_cream_sales.py --server.headless true --server.port 8501

pause
