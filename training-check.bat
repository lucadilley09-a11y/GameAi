@echo off
cd /d "%~dp0"
title Game AI - training check
echo Checking your setup and running a short training. This window shows everything.
echo Everything is also saved to logs\training-check.txt
echo.
if not exist logs mkdir logs
.venv\Scripts\python.exe -u -m gameai.diagnose 2>&1 | .venv\Scripts\python.exe -u -m gameai.tee logs\training-check.txt
echo.
echo Done. Send Claude the file logs\training-check.txt (or a screenshot of this window).
pause
