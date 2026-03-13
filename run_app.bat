@echo off
set "GRADIO_SERVER_PORT=7861"
set "PYTHONIOENCODING=utf-8"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

echo Starting VieNeu-TTS Studio...
echo Please wait about 1-2 minutes for models to load.
echo Once started, visit: http://127.0.0.1:7861
echo.

uv run vieneu-web

pause
