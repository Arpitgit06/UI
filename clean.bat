@echo off
setlocal enabledelayedexpansion

:: ===============================================================================
::                   OmniUI - Complete Project Factory Reset Script
:: ===============================================================================
:: Automatically resolves project directory regardless of where it is cloned/placed.
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

echo ===============================================================================
echo                      OmniUI Factory Reset Setup Clean
echo ===============================================================================
echo Project Location: %PROJECT_ROOT%
echo.
echo WARNING: This script will completely remove:
echo   1. Python virtual environments (.venv and legacy omniui_env)
echo   2. Downloaded AI models (Qwen2.5-Coder, Qwen2-VL) and checkpoints (models_cache\)
echo   3. All uploaded videos, job records, and output zips (storage\*)
echo   4. Compiled Python cache directories (__pycache__ and *.pyc files)
echo.

if /i "%~1"=="-y" goto :START_CLEAN
if /i "%~1"=="/y" goto :START_CLEAN
if /i "%~1"=="--yes" goto :START_CLEAN

set /p CONFIRM="Are you sure you want to perform a full factory reset? (Y/N): "
if /i not "%CONFIRM%"=="Y" (
    echo Factory reset cancelled.
    exit /b 0
)

:START_CLEAN
echo.
echo [1/5] Removing Python virtual environments (.venv and legacy omniui_env)...
if exist "%PROJECT_ROOT%.venv" (
    echo Deleting .venv...
    rmdir /s /q "%PROJECT_ROOT%.venv" 2>nul
)
if exist "%PROJECT_ROOT%omniui_env" (
    echo Deleting omniui_env...
    rmdir /s /q "%PROJECT_ROOT%omniui_env" 2>nul
)

echo [2/5] Removing downloaded AI models and local checkpoints (models_cache)...
if exist "%PROJECT_ROOT%models_cache" (
    echo Deleting models_cache...
    rmdir /s /q "%PROJECT_ROOT%models_cache" 2>nul
)
if exist "%PROJECT_ROOT%yolov10n.pt" (
    del /f /q "%PROJECT_ROOT%yolov10n.pt" 2>nul
)

echo [3/5] Cleaning storage directories (uploads, jobs, outputs)...
if exist "%PROJECT_ROOT%storage\uploads" (
    for /d %%i in ("%PROJECT_ROOT%storage\uploads\*") do rmdir /s /q "%%i" 2>nul
    del /f /q "%PROJECT_ROOT%storage\uploads\*" 2>nul
) else (
    mkdir "%PROJECT_ROOT%storage\uploads" 2>nul
)

if exist "%PROJECT_ROOT%storage\jobs" (
    for /d %%i in ("%PROJECT_ROOT%storage\jobs\*") do rmdir /s /q "%%i" 2>nul
    del /f /q "%PROJECT_ROOT%storage\jobs\*" 2>nul
) else (
    mkdir "%PROJECT_ROOT%storage\jobs" 2>nul
)

if exist "%PROJECT_ROOT%storage\outputs" (
    for /d %%i in ("%PROJECT_ROOT%storage\outputs\*") do rmdir /s /q "%%i" 2>nul
    del /f /q "%PROJECT_ROOT%storage\outputs\*" 2>nul
) else (
    mkdir "%PROJECT_ROOT%storage\outputs" 2>nul
)

echo [4/5] Removing compiled Python bytecode (__pycache__ and *.pyc)...
for /d /r "%PROJECT_ROOT%" %%d in (__pycache__) do (
    if exist "%%d" rmdir /s /q "%%d" 2>nul
)
del /s /q "%PROJECT_ROOT%*.pyc" 2>nul

echo [5/5] Reset complete!
echo.
echo ===============================================================================
echo All packages, models, and temporary files have been wiped cleanly.
echo To set up and run the tool again from scratch, run:
echo    .\setup.bat
echo ===============================================================================
exit /b 0
