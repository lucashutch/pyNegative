@echo off
setlocal EnableDelayedExpansion

title pyNegative Installer

set REPO=lucashutch/pyNegative
set GITHUB_URL=https://github.com/%REPO%.git

echo ============================================================
echo     pyNegative Installer for Windows
echo ============================================================
echo.
echo This installer will:
echo   1. Install uv (Python package manager) if needed
echo   2. Resolve the latest pyNegative release
echo   3. Install pyNegative using 'uv tool install'
echo   4. Download the lens database
echo   5. Create Start Menu shortcuts
echo.

REM Check for uv
uv --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing uv...
    powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

REM Check if already installed
uv tool list | findstr /C:"pynegative" >nul
if %ERRORLEVEL% EQU 0 (
    echo pyNegative is already installed.
    echo.
    echo What would you like to do?
    echo   1) Update to the latest version
    echo   2) Reinstall pyNegative
    echo   3) Uninstall pyNegative
    echo   4) Cancel
    echo.
    set /p CHOICE="Enter your choice (1-4): "
    if "!CHOICE!"=="1" goto :do_install
    if "!CHOICE!"=="2" goto :do_install
    if "!CHOICE!"=="3" goto :do_uninstall
    echo Operation cancelled.
    exit /b 0
)

set /p CONFIRM="Continue with installation? (y/n): "
if /I not "!CONFIRM!"=="y" (
    echo Installation cancelled.
    exit /b 0
)

:do_install
REM Fetch latest tag
echo.
echo Fetching latest release tag...
for /f "tokens=*" %%a in ('powershell -Command "$resp = Invoke-RestMethod -Uri 'https://api.github.com/repos/%REPO%/releases/latest'; if ($resp.tag_name) { echo $resp.tag_name } else { echo 'main' }"') do set TAG=%%a

set INSTALL_URL=git+%GITHUB_URL%
if not "%TAG%"=="main" set INSTALL_URL=%INSTALL_URL%@%TAG%

echo Installing pyNegative (!TAG!) via uv tool...
uv tool install %INSTALL_URL% --force

echo Downloading lens database...
for /f "tokens=*" %%a in ('uv tool run --from %INSTALL_URL% python -c "from platformdirs import user_data_dir; print(user_data_dir('pyNegative'))"') do set DATA_DIR=%%a
set LENS_DIR=%DATA_DIR%\data\lensfun
if not exist "%LENS_DIR%" mkdir "%LENS_DIR%"
uv tool run --from %INSTALL_URL% download-lens-db --output-dir "%LENS_DIR%" --quiet

REM Create shortcuts
echo Creating shortcuts...
set START_MENU_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\pyNegative
if not exist "%START_MENU_DIR%" mkdir "%START_MENU_DIR%"

set SHORTCUT_PATH=%START_MENU_DIR%\pyNegative.lnk
set EXE_PATH=%USERPROFILE%\.local\bin\pynegative.exe

powershell -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%SHORTCUT_PATH%');$s.TargetPath='%EXE_PATH%';$s.Save()"

echo.
echo ============================================================
echo     pyNegative installed successfully!
echo ============================================================
echo You can launch it from the Start Menu or by running 'pynegative'.
pause
exit /b 0

:do_uninstall
echo.
echo Uninstalling pyNegative...
uv tool uninstall pynegative
if exist "%APPDATA%\Microsoft\Windows\Start Menu\Programs\pyNegative" (
    rmdir /s /q "%APPDATA%\Microsoft\Windows\Start Menu\Programs\pyNegative"
)
echo pyNegative has been uninstalled.
pause
exit /b 0
