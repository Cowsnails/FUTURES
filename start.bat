@echo off
title FUTURES Charting Application
echo ════════════════════════════════════════════════════════
echo   FUTURES Charting Application
echo ════════════════════════════════════════════════════════

cd /d "%~dp0"

:: Check for venv
if not exist "venv\Scripts\python.exe" (
    echo   [1/3] Creating virtual environment...
    python -m venv venv
    echo   [2/3] Installing dependencies (first run only^)...
    venv\Scripts\pip install --upgrade pip -q
    venv\Scripts\pip install -r requirements.txt -q
    echo   [3/3] Setup complete!
    echo.
)

echo   Starting server on http://localhost:8000
echo   Close this window to stop.
echo ════════════════════════════════════════════════════════
echo.

venv\Scripts\python run.py

pause
