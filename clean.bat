@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
set "CLEAN_DRY="
set "CLEAN_YES="
set "CLEAN_HELP="
:PARSE
if "%~1"=="" goto :RUN
if /i "%~1"=="--dry-run" (
    set "CLEAN_DRY=-DryRun"
    shift
    goto :PARSE
)
if /i "%~1"=="--yes" goto :SET_YES
if /i "%~1"=="-y" goto :SET_YES
if /i "%~1"=="/y" goto :SET_YES
if /i "%~1"=="--help" goto :SET_HELP
if /i "%~1"=="/?" goto :SET_HELP
goto :USAGE
:SET_YES
set "CLEAN_YES=-Yes"
shift
goto :PARSE
:SET_HELP
set "CLEAN_HELP=-Help"
shift
goto :PARSE
:RUN
if not exist "%PROJECT_ROOT%tools\clean_project.ps1" (
    echo ERROR: tools\clean_project.ps1 is missing. Restore the project files.
    exit /b 1
)
set "CLEAN_POWERSHELL=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%CLEAN_POWERSHELL%" (
    echo ERROR: Windows PowerShell is required to run cleanup.
    exit /b 1
)
"%CLEAN_POWERSHELL%" -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_ROOT%tools\clean_project.ps1" %CLEAN_DRY% %CLEAN_YES% %CLEAN_HELP%
exit /b %errorlevel%
:USAGE
echo Usage: clean.bat [--dry-run] [--yes] [--help]
exit /b 2
