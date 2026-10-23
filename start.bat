@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"
if not exist "%PROJECT_ROOT%.venv\Scripts\python.exe" (
    echo ERROR: Virtual environment missing. Run setup.bat first.
    exit /b 1
)
"%PROJECT_ROOT%.venv\Scripts\python.exe" "%PROJECT_ROOT%tools\start_server.py" %*
exit /b %errorlevel%
