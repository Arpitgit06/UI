@echo off
setlocal DisableDelayedExpansion
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "PROJECT_ROOT=%~dp0"
if /i "%~1"=="--help" goto :HELP
if /i "%~1"=="/?" goto :HELP
if not exist "%PROJECT_ROOT%tools\start_server.py" (
    echo ERROR: tools\start_server.py is missing. Restore the project files.
    exit /b 1
)
if not exist "%PROJECT_ROOT%.venv\Scripts\python.exe" (
    echo ERROR: Virtual environment missing. Run setup.bat first.
    exit /b 1
)
pushd "%PROJECT_ROOT%" >nul 2>nul
if errorlevel 1 (
    echo ERROR: Cannot access the project directory.
    exit /b 1
)
"%PROJECT_ROOT%.venv\Scripts\python.exe" -I -S -B "%PROJECT_ROOT%tools\check_venv.py" >nul
if errorlevel 1 (
    echo ERROR: The project virtual environment is broken. Recreate .venv and run setup.bat.
    popd
    exit /b 1
)
"%PROJECT_ROOT%.venv\Scripts\python.exe" -I -B "%PROJECT_ROOT%tools\start_server.py" %*
set "RESULT=%errorlevel%"
popd
exit /b %RESULT%
:HELP
echo Usage: start.bat [--port PORT] [--check] [--reload]
echo --check validates local dependencies and cached models without starting a server.
echo --reload enables development restarts; active jobs may be interrupted.
exit /b 0
