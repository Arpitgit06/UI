@echo off
setlocal enabledelayedexpansion

:: Automatically resolve dynamic project root location
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

echo ===============================================================================
echo                 OmniUI - Local Video-to-UI Setup (Windows GPU)
echo ===============================================================================
echo Project Directory: %PROJECT_ROOT%
echo.

:: 1. Check Python installation and version
echo [1/6] Checking Python installation...
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in your PATH.
    echo Please install Python 3.10+ from https://python.org and make sure to check "Add Python to PATH".
    pause
    exit /b 1
)

for /f "tokens=2 delims=." %%v in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do (
    set PY_MINOR=%%v
)
if !PY_MINOR! lss 10 (
    echo WARNING: Detected Python version older than 3.10. OmniUI requires Python 3.10+.
    echo Continuing anyway, but you may encounter syntax or packaging errors.
) else (
    echo Found compatible Python environment.
)
echo.

:: 2. Create and activate virtual environment (.venv)
echo [2/6] Setting up virtual environment (.venv)...
if not exist "%PROJECT_ROOT%.venv\Scripts\activate.bat" (
    echo Creating new Python virtual environment .venv in project folder...
    python -m venv "%PROJECT_ROOT%.venv"
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment .venv already exists.
)

echo Activating virtual environment (.venv)...
call "%PROJECT_ROOT%.venv\Scripts\activate.bat"
set "VENV_PYTHON=%PROJECT_ROOT%.venv\Scripts\python.exe"
set PIP_REQUIRE_VIRTUALENV=true

"%VENV_PYTHON%" -m pip install --upgrade pip >nul 2>nul
echo.

:: 3. Install GPU wheels strictly inside venv BEFORE requirements.txt
echo [3/6] Installing NVIDIA GPU accelerated wheels inside virtual environment...
echo This will download ~2.5GB+ of GPU wheels directly into .venv to ensure containment. Please be patient...
echo.
echo Installing PyTorch with CUDA 12.1 support...
"%VENV_PYTHON%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (
    echo WARNING: PyTorch CUDA wheel installation encountered an issue or fallback occurred.
)

echo.
echo Installing PaddlePaddle GPU with CUDA 12.6 support...
"%VENV_PYTHON%" -m pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
if %errorlevel% neq 0 (
    echo WARNING: PaddlePaddle GPU wheel installation encountered an issue.
    echo Note: If your GPU CUDA driver does not support cu126, check https://www.paddlepaddle.org.cn/en/install/quick
)
echo.

:: 4. Install requirements.txt strictly inside virtual environment
echo [4/6] Installing OmniUI core dependencies into virtual environment...
"%VENV_PYTHON%" -m pip install -r "%PROJECT_ROOT%requirements.txt"
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies from requirements.txt into virtual environment.
    pause
    exit /b 1
)
echo.

:: 5. Create directories & configuration
echo [5/6] Initializing storage directories and .env configuration...
if not exist "%PROJECT_ROOT%storage\uploads" mkdir "%PROJECT_ROOT%storage\uploads"
if not exist "%PROJECT_ROOT%storage\jobs" mkdir "%PROJECT_ROOT%storage\jobs"
if not exist "%PROJECT_ROOT%storage\outputs" mkdir "%PROJECT_ROOT%storage\outputs"
if not exist "%PROJECT_ROOT%models_cache" mkdir "%PROJECT_ROOT%models_cache"

if not exist "%PROJECT_ROOT%.env" (
    if exist "%PROJECT_ROOT%.env.example" (
        copy /Y "%PROJECT_ROOT%.env.example" "%PROJECT_ROOT%.env" >nul
        echo Created .env file from .env.example.
    ) else (
        echo OMNIUI_CUDA_DEVICE=0 > "%PROJECT_ROOT%.env"
        echo Created default .env file.
    )
) else (
    echo .env configuration already exists.
)
echo.

:: 6. Pre-cache download local AI models strictly inside models_cache
echo [6/6] Pre-caching local AI models into project directory (models_cache)...
echo Downloading local 7B LLMs (Qwen2.5-Coder-7B-Instruct and Qwen2-VL-7B-Instruct) right now so the tool works 100%% offline forever...
echo This will download approximately 30GB of weights. Please be patient depending on your internet speed...
"%VENV_PYTHON%" -c "from app.config import settings; import os; os.environ['HF_HOME'] = str(settings.models_cache_dir); from huggingface_hub import snapshot_download; from transformers import pipeline; print('Downloading Depth-Anything-V2 weights into models_cache...'); pipeline(task='depth-estimation', model=settings.depth_model_name); print('Downloading Qwen2.5-Coder-7B-Instruct...'); snapshot_download(repo_id='Qwen/Qwen2.5-Coder-7B-Instruct'); print('Downloading Qwen2-VL-7B-Instruct...'); snapshot_download(repo_id='Qwen/Qwen2-VL-7B-Instruct'); print('Model pre-caching complete!')"
if %errorlevel% neq 0 (
    echo WARNING: Pre-caching encountered an issue or interruption. Models will be downloaded on-demand upon first run.
)
echo.
echo ===============================================================================
echo Setup completed successfully! All models and packages are contained in %PROJECT_ROOT%
echo You can now start OmniUI by running:
echo        .\start.bat
echo ===============================================================================
pause
