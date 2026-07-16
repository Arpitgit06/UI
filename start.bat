@echo off
setlocal enabledelayedexpansion

:: Automatically resolve dynamic project root location
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

echo ===============================================================================
echo                 Starting OmniUI Core Service (Local GPU)
echo ===============================================================================
echo Project Directory: %PROJECT_ROOT%
echo.

:: 1. Check virtual environment (.venv)
if not exist "%PROJECT_ROOT%.venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment '.venv' not found in %PROJECT_ROOT%!
    echo Please run setup.bat first to set up your GPU Python environment and models.
    pause
    exit /b 1
)

call "%PROJECT_ROOT%.venv\Scripts\activate.bat"
set "VENV_PYTHON=%PROJECT_ROOT%.venv\Scripts\python.exe"

:: 2. Ensure storage and static directories exist
if not exist "%PROJECT_ROOT%storage\uploads" mkdir "%PROJECT_ROOT%storage\uploads"
if not exist "%PROJECT_ROOT%storage\jobs" mkdir "%PROJECT_ROOT%storage\jobs"
if not exist "%PROJECT_ROOT%storage\outputs" mkdir "%PROJECT_ROOT%storage\outputs"
if not exist "%PROJECT_ROOT%models_cache" mkdir "%PROJECT_ROOT%models_cache"
if not exist "%PROJECT_ROOT%static" mkdir "%PROJECT_ROOT%static"

:: 3. Quick check for CUDA availability via PyTorch inside venv
echo Checking local GPU CUDA environment...
"%VENV_PYTHON%" -c "import torch; print(f'GPU Available: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU Fallback\"})')" 2>nul
if %errorlevel% neq 0 (
    echo Notice: PyTorch check skipped or running CPU mode.
)
echo.

echo Starting FastAPI Uvicorn Server on http://localhost:8000 ...
echo - Web Dashboard: http://localhost:8000/
echo - Interactive Swagger API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server.
echo -------------------------------------------------------------------------------
"%VENV_PYTHON%" -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
