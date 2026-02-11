@echo off

set "PYTHON=%~dp0venv\Scripts\python.exe"
set "GIT="
set "VENV_DIR=.\venv"

set "COMMANDLINE_ARGS=--auto-launch --use-quad-cross-attention --reserve-vram 0.9"

set "ZLUDA_COMGR_LOG_LEVEL=1"

:: Check Git version
where git >NUL 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed or not found in the system PATH.
    echo         Please install Git from https://git-scm.com and ensure it's added to your PATH during installation.
) else (
    for /f "tokens=3" %%v in ('git --version') do (
        echo [INFO] Detected Git version: %%v
    )
)



echo [INFO] Launching application via ZLUDA...
echo.
.\zluda\zluda.exe -- %PYTHON% main.py %COMMANDLINE_ARGS%
pause

