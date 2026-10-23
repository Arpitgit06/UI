@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"
where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Install Python 3.10+ and add it to PATH.
    exit /b 1
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 exit /b 1
if not exist "%PROJECT_ROOT%.venv\Scripts\python.exe" (
    python -m venv "%PROJECT_ROOT%.venv"
    if errorlevel 1 exit /b 1
)
"%PROJECT_ROOT%.venv\Scripts\python.exe" "%PROJECT_ROOT%tools\setup_project.py" %*
exit /b %errorlevel%
