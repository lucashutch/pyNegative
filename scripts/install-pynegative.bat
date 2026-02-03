@echo off
REM ============================================================
REM pyNegative Single-File Installer for Windows
REM ============================================================
REM This script downloads and installs pyNegative using uv.
REM Just double-click this file to install!
REM
REM Silent mode: Add --silent flag
REM Verbose mode: Add --verbose or -v flag
REM ============================================================

title pyNegative Installer

setlocal EnableDelayedExpansion

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Configuration
set REPO=lucashutch/pyNegative
set APP_NAME=pyNegative
set INSTALL_DIR=%USERPROFILE%\%APP_NAME%
set VERSION_FILE=%INSTALL_DIR%\.version
set SCRIPT_URL=https://raw.githubusercontent.com/lucashutch/pyNegative/main/scripts/download_release.py
set TIMEOUT=15
set TEMP_SCRIPT=%TEMP%\download_release.py.%RANDOM%

REM Check for silent mode flags
set SILENT_MODE=0
set VERBOSE_MODE=0
if "%~1"=="--silent" set SILENT_MODE=1
if "%~1"=="-silent" set SILENT_MODE=1
if "%~1"=="--yes" set SILENT_MODE=1
if "%~1"=="-yes" set SILENT_MODE=1
if "%~1"=="-s" set SILENT_MODE=1
if "%~1"=="--verbose" set VERBOSE_MODE=1
if "%~1"=="-v" set VERBOSE_MODE=1

REM Check if already installed
if exist "%INSTALL_DIR%\pyproject.toml" (
    call :show_installed_menu
    exit /b 0
)

REM Show welcome and get confirmation
if %SILENT_MODE%==0 (
    cls
    echo.
    echo ============================================================
    echo     Welcome to pyNegative Installer for Windows
    echo ============================================================
    echo.
    echo This installer will:
    echo   1. Install uv ^(Python package manager^) if needed
    echo   2. Download the latest pyNegative release ^(or main branch^)
    echo   3. Install Python dependencies ^(PySide6, numpy, pillow, etc.^)
    echo   4. Create Start Menu shortcuts
    echo.
    echo Installation location: %INSTALL_DIR%
    echo.
    set /p CONFIRM="Continue with installation? (y/n): "
    if /I not "!CONFIRM!"=="y" (
        echo Installation cancelled.
        exit /b 0
    )
    echo.
)

REM Check if uv is installed
call :check_uv
if %ERRORLEVEL% NEQ 0 (
    call :install_uv
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to install uv
        pause
        exit /b 1
    )
) else (
    if %SILENT_MODE%==0 echo uv is already installed
)

REM Download and install using uv's Python
echo Checking for latest release...
echo.

REM Fetch the download script first
call :fetch_download_script
if %ERRORLEVEL% NEQ 0 (
    exit /b 1
)

REM Run the downloaded script using uv
uv run --python 3 python "%TEMP_SCRIPT%" --repo %REPO% --install-dir %INSTALL_DIR%
set DOWNLOAD_RESULT=%ERRORLEVEL%

if %DOWNLOAD_RESULT% EQU 0 (
    echo Download and extraction complete!
) else if %DOWNLOAD_RESULT% EQU 2 (
    echo Already on latest version, skipping download.
) else (
    echo ERROR: Download failed
    del "%TEMP_SCRIPT%" 2>nul
    pause
    exit /b 1
)

REM Clean up temp script
del "%TEMP_SCRIPT%" 2>nul

echo.
echo Installing dependencies...

cd /d "%INSTALL_DIR%"
uv sync --all-groups
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo Dependencies installed successfully!

REM Create Start Menu shortcuts
call :create_shortcuts

REM Success message
echo.
echo ============================================================
echo     pyNegative installed successfully!
echo ============================================================
echo.
echo You can now launch pyNegative from:
echo   - Start Menu ^> pyNegative
echo   - Or by running: uv run pyneg-ui
echo.

if %SILENT_MODE%==0 (
    set /p LAUNCH="Launch pyNegative now? (y/n): "
    if /I "!LAUNCH!"=="y" (
        start /b uv run pyneg-ui
    )
)

exit /b 0

REM Show installed menu
:show_installed_menu
echo.
echo pyNegative is already installed at:
echo %INSTALL_DIR%

REM Show current version
if exist "%VERSION_FILE%" (
    set /p CURRENT_VERSION=<"%VERSION_FILE%"
    echo Current version: %CURRENT_VERSION%
)

echo.
echo What would you like to do?
echo   1) Update pyNegative to the latest version
echo   2) Uninstall pyNegative
echo   3) Cancel
echo.

if %SILENT_MODE%==1 (
    REM In silent mode, default to update
    call :do_update
    exit /b 0
)

set /p CHOICE="Enter your choice (1-3): "
if "%CHOICE%"=="1" (
    call :do_update
) else if "%CHOICE%"=="2" (
    set /p CONFIRM="Are you sure you want to uninstall? (y/n): "
    if /I not "!CONFIRM!"=="y" (
        echo Uninstall cancelled.
        exit /b 0
    )
    call :uninstall_pynegative
) else if "%CHOICE%"=="3" (
    echo Operation cancelled.
) else (
    echo Invalid choice
    call :show_installed_menu
)
exit /b 0

REM Update pyNegative
:do_update
echo.
echo Updating pyNegative...

if not exist "%INSTALL_DIR%\scripts\install-pynegative.bat" (
    echo ERROR: Installer script not found for updating.
    exit /b 1
)

REM Download (will check version and skip if same)
call :fetch_download_script
if %ERRORLEVEL% NEQ 0 (
    REM Check if already on latest (exit code 2)
    if %ERRORLEVEL% EQU 2 (
        echo Already on latest version!
    ) else (
        echo ERROR: Update failed
    )
    exit /b 1
)

REM Update dependencies
echo Updating dependencies...
cd /d "%INSTALL_DIR%"
uv sync --all-groups
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to update dependencies
    exit /b 1
)

echo Dependencies updated successfully!

REM Copy updated installer files
if not exist "%INSTALL_DIR%\scripts" mkdir "%INSTALL_DIR%\scripts"
copy /Y "%~f0" "%INSTALL_DIR%\scripts\install-pynegative.bat" >nul 2>&1

echo.
echo pyNegative has been updated successfully!
exit /b 0

REM Uninstall pyNegative
:uninstall_pynegative
echo.
echo Uninstalling pyNegative...

REM Remove installation directory
if exist "%INSTALL_DIR%" (
    rmdir /s /q "%INSTALL_DIR%"
    echo Removed installation directory: %INSTALL_DIR%
)

REM Remove Start Menu entry
if exist "%APPDATA%\Microsoft\Windows\Start Menu\Programs\pyNegative.lnk" (
    del "%APPDATA%\Microsoft\Windows\Start Menu\Programs\pyNegative.lnk"
    echo Removed application menu entry
)

REM Remove Start Menu directory
if exist "%APPDATA%\Microsoft\Windows\Start Menu\Programs\%APP_NAME%" (
    rmdir /s /q "%APPDATA%\Microsoft\Windows\Start Menu\Programs\%APP_NAME%"
    echo Removed Start Menu directory
)

echo.
echo pyNegative has been uninstalled successfully!
exit /b 0

REM ============================================================
REM Functions
REM ============================================================

:fetch_download_script
if %SILENT_MODE%==0 echo Downloading installer script...
if %VERBOSE_MODE%==1 (
    echo DEBUG: Script URL: %SCRIPT_URL%
    echo DEBUG: Temp file: %TEMP_SCRIPT%
    echo DEBUG: Timeout: %TIMEOUT% seconds
)

REM Create PowerShell command to download the script
if %VERBOSE_MODE%==1 echo DEBUG: Starting PowerShell download...
powershell -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -Uri '%SCRIPT_URL%' -OutFile '%TEMP_SCRIPT%' -TimeoutSec %TIMEOUT%; exit 0 } catch { exit 1 }"
set DOWNLOAD_RESULT=%ERRORLEVEL%

if %VERBOSE_MODE%==1 echo DEBUG: PowerShell exit code: %DOWNLOAD_RESULT%

if %DOWNLOAD_RESULT% NEQ 0 (
    if %SILENT_MODE%==0 echo ERROR: Failed to download installer script. Please check your internet connection and try again.
    if %VERBOSE_MODE%==1 echo DEBUG: Download failed with error code %DOWNLOAD_RESULT%
    exit /b 1
)

REM Simple validation: Check if file exists
if %VERBOSE_MODE%==1 echo DEBUG: Checking if downloaded file exists...

if not exist "%TEMP_SCRIPT%" (
    if %SILENT_MODE%==0 echo ERROR: Downloaded script file not found.
    if %VERBOSE_MODE%==1 echo DEBUG: File does not exist: %TEMP_SCRIPT%
    exit /b 1
)

if %VERBOSE_MODE%==1 echo DEBUG: File exists, proceeding to execution.

if %SILENT_MODE%==0 echo Installer script downloaded successfully!
if %VERBOSE_MODE%==1 (
    echo DEBUG: Script validation complete
    echo DEBUG: Ready to execute Python script
)
exit /b 0

:check_uv
uv --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    if %SILENT_MODE%==0 echo uv is already installed
    exit /b 0
) else (
    exit /b 1
)

:install_uv
if %SILENT_MODE%==0 echo Installing uv...
powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
if %ERRORLEVEL% NEQ 0 (
    exit /b 1
)
REM Add uv to PATH for this session
set PATH=%USERPROFILE%\.local\bin;%PATH%
if %SILENT_MODE%==0 echo uv installed successfully!
exit /b 0

:create_shortcuts
echo.
echo Creating shortcuts...

REM Create Start Menu directory
set START_MENU_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\%APP_NAME%
if not exist "%START_MENU_DIR%" mkdir "%START_MENU_DIR%"

REM Create PowerShell script for making shortcuts
echo $wsh = New-Object -ComObject WScript.Shell > %TEMP%\create_shortcuts.ps1
echo $installDir = '%INSTALL_DIR%' >> %TEMP%\create_shortcuts.ps1
echo $appName = '%APP_NAME%' >> %TEMP%\create_shortcuts.ps1
echo $startMenuDir = '%START_MENU_DIR%' >> %TEMP%\create_shortcuts.ps1
echo. >> %TEMP%\create_shortcuts.ps1
echo # Main app shortcut >> %TEMP%\create_shortcuts.ps1
echo $shortcut = $wsh.CreateShortcut("$startMenuDir\$appName.lnk") >> %TEMP%\create_shortcuts.ps1
echo $shortcut.TargetPath = "uv" >> %TEMP%\create_shortcuts.ps1
echo $shortcut.Arguments = "run pyneg-ui" >> %TEMP%\create_shortcuts.ps1
echo $shortcut.WorkingDirectory = $installDir >> %TEMP%\create_shortcuts.ps1
echo $shortcut.Description = "pyNegative - RAW Image Processor" >> %TEMP%\create_shortcuts.ps1
echo # Try to use generated icon, fall back to main icon >> %TEMP%\create_shortcuts.ps1
echo $iconPath = "$installDir\scripts\icons\pynegative.ico" >> %TEMP%\create_shortcuts.ps1
echo if (-not (Test-Path $iconPath)) { $iconPath = "$installDir\pynegative_icon.png" } >> %TEMP%\create_shortcuts.ps1
echo if (Test-Path $iconPath) { $shortcut.IconLocation = $iconPath } >> %TEMP%\create_shortcuts.ps1
echo $shortcut.Save() >> %TEMP%\create_shortcuts.ps1
echo. >> %TEMP%\create_shortcuts.ps1
echo # Uninstaller shortcut >> %TEMP%\create_shortcuts.ps1
echo $uninstall = $wsh.CreateShortcut("$startMenuDir\Uninstall $appName.lnk") >> %TEMP%\create_shortcuts.ps1
echo $uninstall.TargetPath = "$installDir\scripts\install-pynegative.bat" >> %TEMP%\create_shortcuts.ps1
echo $uninstall.WorkingDirectory = "$installDir\scripts" >> %TEMP%\create_shortcuts.ps1
echo $uninstall.Description = "Uninstall or Update pyNegative" >> %TEMP%\create_shortcuts.ps1
echo $uninstall.Save() >> %TEMP%\create_shortcuts.ps1

powershell -ExecutionPolicy Bypass -File %TEMP%\create_shortcuts.ps1

REM Copy installer files to install dir for future updates
if not exist "%INSTALL_DIR%\scripts" mkdir "%INSTALL_DIR%\scripts"
copy /Y "%~f0" "%INSTALL_DIR%\scripts\install-pynegative.bat" >nul 2>&1

echo Shortcuts created!
exit /b 0