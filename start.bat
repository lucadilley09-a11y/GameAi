@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\pythonw.exe (
  echo Run setup.bat first.
  pause
  exit /b 1
)
rem Opens just the Game AI window (no black window).
rem Problems? Use start-debug.bat.
rem----------
goto run
rem--------
:run
start "" ".venv\Scripts\pythonw.exe" -m gameai.launch
exit /b 0
