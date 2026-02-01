@echo off
title FUTURES Charting Application
echo ════════════════════════════════════════════════════════
echo   FUTURES Charting Application
echo ════════════════════════════════════════════════════════

cd /d "%~dp0"

:: Create venv if it doesn't exist
if not exist "venv\Scripts\python.exe" (
    echo   [1/3] Creating virtual environment...
    python -m venv venv
)

:: Always ensure deps are installed (handles partial installs)
echo   Checking dependencies...
venv\Scripts\python.exe -m pip install --upgrade pip -q 2>nul
venv\Scripts\python.exe -m pip install -r requirements.txt -q
echo   Dependencies ready!
echo.

echo   Starting server on http://localhost:8000
echo   Close this window to stop.
echo ════════════════════════════════════════════════════════
echo.

venv\Scripts\python run.py

pause
