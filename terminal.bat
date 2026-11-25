@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
title Game AI terminal
python -m gameai.update
python -m gameai
