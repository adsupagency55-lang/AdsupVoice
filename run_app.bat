@echo off
setlocal
set "GRADIO_SERVER_PORT=7861"
set "PYTHONIOENCODING=utf-8"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

echo ==========================================
echo    VieNeu-TTS Studio - Auto Launcher
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
    echo [INFO] First time setup: Installing dependencies...
    uv sync
)

echo [INFO] Starting VieNeu-TTS Studio...
echo [INFO] Once started, visit: http://127.0.0.1:7861
echo.

uv run vieneu-web

pause
endlocal

