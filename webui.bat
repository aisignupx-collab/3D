@echo off
rem Opens the asset pipeline web UI in your browser. Uses .venv if it exists.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" webui.py %*
) else (
  python webui.py %*
)
pause
