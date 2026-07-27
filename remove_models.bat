@echo off
setlocal enabledelayedexpansion

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

echo ===============================================================================
echo                 OmniUI - Remove Downloaded AI Models
echo ===============================================================================
echo Project Location: %PROJECT_ROOT%
echo.
echo WARNING: This script will remove ONLY the downloaded AI models and weights:
echo   - Hugging Face cache (Qwen2.5-Coder, Qwen2-VL, Depth-Anything-V2)
echo   - YOLO checkpoints
echo   - PaddlePaddle/PaddleOCR cached weights
echo.
echo Your virtual environment (.venv) and storage/jobs will NOT be deleted.
echo.

if /i "%~1"=="-y" goto :START_CLEAN
if /i "%~1"=="/y" goto :START_CLEAN
if /i "%~1"=="--yes" goto :START_CLEAN

set /p CONFIRM="Are you sure you want to delete the models cache? (Y/N): "
if /i not "%CONFIRM%"=="Y" (
    echo Model removal cancelled.
    exit /b 0
)

:START_CLEAN
echo.
echo Removing downloaded AI models and local checkpoints (models_cache)...
if exist "%PROJECT_ROOT%models_cache" (
    echo Deleting models_cache...
    rmdir /s /q "%PROJECT_ROOT%models_cache" 2>nul
)
if exist "%PROJECT_ROOT%yolov10n.pt" (
    del /f /q "%PROJECT_ROOT%yolov10n.pt" 2>nul
)

echo.
echo ===============================================================================
echo Models removed successfully.
echo Next time you start the app, models will either be downloaded on-demand,
echo or you can run setup.bat to pre-cache them again.
echo ===============================================================================
exit /b 0
