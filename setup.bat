@echo off
setlocal
cd /d "%~dp0"
echo ============================================
echo   Game AI setup (only needed once)
echo ============================================

rem Find a Python that really runs (Windows has a fake "python" that only opens the Microsoft Store)
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "PY=python"
if not defined PY goto nopython
echo Using Python:
%PY% --version

rem Remove a half-made environment from an earlier failed run
if exist .venv if not exist .venv\Scripts\python.exe rmdir /s /q .venv

if not exist .venv (
  echo Creating a private Python environment in this folder...
  %PY% -m venv .venv || goto fail
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul

python -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul && goto torchok
for %%C in (cu130 cu128 cu126) do (
  echo Installing PyTorch for your NVIDIA GPU [%%C], this is a big download...
  python -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/%%C
  python -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul && goto torchok
)
goto nogpu

:torchok
echo Installing the other parts...
python -m pip install -r requirements.txt || goto fail
python -c "import torch;print('GPU found:', torch.cuda.get_device_name(0))"
echo.
echo Setup done. Now double-click start.bat
pause
exit /b 0

:nopython
echo Python 3.10 or newer was not found on this PC.
echo (The "Microsoft Store" message is Windows' placeholder, not real Python.)
choice /m "Install Python 3.12 now with winget"
if errorlevel 2 goto manual
winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
echo.
echo If it says installed: CLOSE this window, then double-click setup.bat again.
pause
exit /b 1

:manual
echo Install Python 3.12 from https://www.python.org/downloads/
echo During install, tick "Add Python to PATH". Then double-click setup.bat again.
pause
exit /b 1

:nogpu
echo PyTorch could not see your GPU. Update your NVIDIA driver from https://www.nvidia.com/drivers
echo then double-click setup.bat again. If it still fails, send Claude the text above.
pause
exit /b 1

:fail
echo Setup failed. Copy the text above and send it to Claude.
pause
exit /b 1
