@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Run setup.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
title Game AI (keep this window open)
:start
python -m gameai.update
python -m gameai.app
if %errorlevel%==42 (
  echo Restarting with the new version...
  goto start
)
