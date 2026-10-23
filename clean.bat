@echo off
setlocal
if not "%~2"=="" goto :USAGE
set "CLEAN_OPTION="
if "%~1"=="" goto :RUN
if /i "%~1"=="--dry-run" set "CLEAN_OPTION=-DryRun"
if /i "%~1"=="--yes" set "CLEAN_OPTION=-Yes"
if /i "%~1"=="-y" set "CLEAN_OPTION=-Yes"
if /i "%~1"=="/y" set "CLEAN_OPTION=-Yes"
if /i "%~1"=="--help" set "CLEAN_OPTION=-Help"
if /i "%~1"=="/?" set "CLEAN_OPTION=-Help"
if not defined CLEAN_OPTION goto :USAGE
:RUN
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\clean_project.ps1" %CLEAN_OPTION%
exit /b %errorlevel%
:USAGE
echo Usage: clean.bat [--dry-run ^| --yes ^| --help]
exit /b 2
