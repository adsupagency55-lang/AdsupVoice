@echo off
setlocal
set "PYTHONIOENCODING=utf-8"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
set "ADSUP_HOST=127.0.0.1"
set "ADSUP_PORT=8080"

echo ==========================================
echo    AdsupVoice Studio - Auto Launcher
echo ==========================================

:: Check if uv is installed
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] 'uv' is not found. Installing 'uv' package manager...
    powershell -ExecutionPolicy ByPass -c "Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force; irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

:: Verify installation
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Could not install 'uv' automatically.
    echo Please install it manually from: https://astral.sh/uv
    pause
    exit /b 1
)

:: Check if .venv exists, if not run sync
if not exist ".venv" (
    echo [INFO] First time setup: Installing dependencies (may take 5-10 minutes)...
    uv sync
)

echo [INFO] Starting AdsupVoice Studio...
echo [INFO] Once started, open your browser and visit: http://127.0.0.1:8080
echo [INFO] (The window will stay open while the server is running - DO NOT close it)
echo.

uv run python apps/web_ui_server.py

pause
endlocal


