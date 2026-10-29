@echo off
setlocal DisableDelayedExpansion
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "PROJECT_ROOT=%~dp0"
set "PLAN_ONLY="
if not "%~2"=="" goto :USAGE
if /i "%~1"=="--help" goto :HELP
if /i "%~1"=="/?" goto :HELP
if /i "%~1"=="--plan" set "PLAN_ONLY=1"
if not "%~1"=="" if not defined PLAN_ONLY goto :USAGE
if not exist "%PROJECT_ROOT%tools\setup_project.py" (
    echo ERROR: tools\setup_project.py is missing. Restore the project files.
    exit /b 1
)
pushd "%PROJECT_ROOT%" >nul 2>nul
if errorlevel 1 (
    echo ERROR: Cannot access the project directory.
    exit /b 1
)
set "VENV_PYTHON=%PROJECT_ROOT%.venv\Scripts\python.exe"
if exist "%VENV_PYTHON%" goto :EXISTING_ENV
if exist "%PROJECT_ROOT%.venv" goto :BROKEN_ENV

set "SETUP_PYTHON="
set "SETUP_PYTHON_FLAG="
call :CHECK_PYTHON python
if not errorlevel 1 set "SETUP_PYTHON=python"
if defined SETUP_PYTHON goto :FOUND_PYTHON
for %%V in (3.11 3.12) do (
    if not defined SETUP_PYTHON (
        call :CHECK_PYTHON py -%%V
        if not errorlevel 1 (
            set "SETUP_PYTHON=py"
            set "SETUP_PYTHON_FLAG=-%%V"
        )
    )
)
if not defined SETUP_PYTHON (
    echo ERROR: Install 64-bit Python 3.11 or 3.12 and add Python or its py launcher to PATH.
    set "RESULT=1"
    goto :FINISH
)
:FOUND_PYTHON
if defined PLAN_ONLY (
    %SETUP_PYTHON% %SETUP_PYTHON_FLAG% -I -S -B "%PROJECT_ROOT%tools\setup_project.py" --plan
    goto :CAPTURE_RESULT
)
%SETUP_PYTHON% %SETUP_PYTHON_FLAG% -I -S -B "%PROJECT_ROOT%tools\check_venv.py" --create
if errorlevel 1 (
    echo ERROR: Could not create the project virtual environment.
    set "RESULT=1"
    goto :FINISH
)
:EXISTING_ENV
call :CHECK_PYTHON "%VENV_PYTHON%"
if errorlevel 1 goto :BROKEN_ENV
"%VENV_PYTHON%" -I -S -B "%PROJECT_ROOT%tools\check_venv.py" >nul
if errorlevel 1 goto :BROKEN_ENV
"%VENV_PYTHON%" -I -B "%PROJECT_ROOT%tools\setup_project.py" %*
:CAPTURE_RESULT
set "RESULT=%errorlevel%"
:FINISH
popd
exit /b %RESULT%
:BROKEN_ENV
echo ERROR: .venv is incomplete, broken, or uses an unsupported Python version.
echo Recreate only .venv with 64-bit Python 3.11-3.12. Existing models and storage can be reused.
set "RESULT=1"
goto :FINISH
:CHECK_PYTHON
%* -I -S -B -c "import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] < (3,13) and sys.maxsize > 2**32 else 1)" >nul 2>nul
exit /b %errorlevel%
:HELP
echo Usage: setup.bat [--plan ^| --help]
echo Installs dependencies, configured VL-7B/coder models, OCR and LTX, and the browser.
echo --plan prints commands without creating .venv, changing .env, or downloading.
exit /b 0
:USAGE
echo Usage: setup.bat [--plan ^| --help]
exit /b 2
