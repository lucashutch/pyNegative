@echo off
REM ============================================================
REM pyNegative Single-File Installer for Windows
REM ============================================================
REM This script downloads and installs pyNegative using uv.
REM Just double-click this file to install!
REM
REM Silent mode: Add --silent flag
REM ============================================================

title pyNegative Installer

setlocal EnableDelayedExpansion

REM Configuration
set REPO=lucashutch/pyNegative
set APP_NAME=pyNegative
set INSTALL_DIR=%USERPROFILE%\%APP_NAME%
set VERSION_FILE=%INSTALL_DIR%\.version

REM Check for silent mode flags
set SILENT_MODE=0
if "%~1"=="--silent" set SILENT_MODE=1
if "%~1"=="-silent" set SILENT_MODE=1
if "%~1"=="--yes" set SILENT_MODE=1
if "%~1"=="-yes" set SILENT_MODE=1
if "%~1"=="-s" set SILENT_MODE=1

REM Function to check if uv is installed
call :check_uv
if %ERRORLEVEL% NEQ 0 (
    call :install_uv
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to install uv
        pause
        exit /b 1
    )
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
    echo   1. Download the latest pyNegative release ^(or main branch^)
    echo   2. Install all dependencies
    echo   3. Create Start Menu shortcuts
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

REM Download and install using uv's Python
echo Checking for latest release...
echo.

REM Create Python script for download
call :create_download_script

REM Run the download script using uv
uv run --python 3 python %TEMP%\pynegative_download.py %INSTALL_DIR% %REPO%
set DOWNLOAD_RESULT=%ERRORLEVEL%

if %DOWNLOAD_RESULT% EQU 0 (
    echo Download and extraction complete!
) else if %DOWNLOAD_RESULT% EQU 2 (
    echo Already on latest version, skipping download.
) else (
    echo ERROR: Download failed
    pause
    exit /b 1
)

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

REM ============================================================
REM Functions
REM ============================================================

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

:create_download_script
REM Create Python script for downloading and extracting
echo import urllib.request > %TEMP%\pynegative_download.py
echo import json >> %TEMP%\pynegative_download.py
echo import zipfile >> %TEMP%\pynegative_download.py
echo import io >> %TEMP%\pynegative_download.py
echo import os >> %TEMP%\pynegative_download.py
echo import sys >> %TEMP%\pynegative_download.py
echo import shutil >> %TEMP%\pynegative_download.py
echo. >> %TEMP%\pynegative_download.py
echo INSTALL_DIR = sys.argv[1] >> %TEMP%\pynegative_download.py
echo REPO = sys.argv[2] >> %TEMP%\pynegative_download.py
echo VERSION_FILE = os.path.join(INSTALL_DIR, '.version') >> %TEMP%\pynegative_download.py
echo. >> %TEMP%\pynegative_download.py
echo def get_latest_version(): >> %TEMP%\pynegative_download.py
echo     try: >> %TEMP%\pynegative_download.py
echo         req = urllib.request.Request( >> %TEMP%\pynegative_download.py
echo             f'https://api.github.com/repos/{REPO}/releases/latest', >> %TEMP%\pynegative_download.py
echo             headers={'User-Agent': 'pyNegative-Installer', 'Accept': 'application/vnd.github.v3+json'} >> %TEMP%\pynegative_download.py
echo         ) >> %TEMP%\pynegative_download.py
echo         with urllib.request.urlopen(req, timeout=10) as response: >> %TEMP%\pynegative_download.py
echo             data = json.loads(response.read().decode()) >> %TEMP%\pynegative_download.py
echo             return data.get('tag_name', 'main') >> %TEMP%\pynegative_download.py
echo     except Exception as e: >> %TEMP%\pynegative_download.py
echo         print(f'Could not check releases: {e}') >> %TEMP%\pynegative_download.py
echo         return 'main' >> %TEMP%\pynegative_download.py
echo. >> %TEMP%\pynegative_download.py
echo def download_and_extract(version, dest_dir): >> %TEMP%\pynegative_download.py
echo     if version == 'main': >> %TEMP%\pynegative_download.py
echo         url = f'https://github.com/{REPO}/archive/refs/heads/main.zip' >> %TEMP%\pynegative_download.py
echo         print(f'Downloading main branch...') >> %TEMP%\pynegative_download.py
echo     else: >> %TEMP%\pynegative_download.py
echo         url = f'https://github.com/{REPO}/archive/refs/tags/{version}.zip' >> %TEMP%\pynegative_download.py
echo         print(f'Downloading {version}...') >> %TEMP%\pynegative_download.py
echo. >> %TEMP%\pynegative_download.py
echo     try: >> %TEMP%\pynegative_download.py
echo         with urllib.request.urlopen(url, timeout=60) as response: >> %TEMP%\pynegative_download.py
echo             zip_data = io.BytesIO(response.read()) >> %TEMP%\pynegative_download.py
echo. >> %TEMP%\pynegative_download.py
echo         temp_extract = dest_dir + '.extracting' >> %TEMP%\pynegative_download.py
echo         if os.path.exists(temp_extract): >> %TEMP%\pynegative_download.py
echo             shutil.rmtree(temp_extract) >> %TEMP%\pynegative_download.py
echo. >> %TEMP%\pynegative_download.py
echo         with zipfile.ZipFile(zip_data, 'r') as zf: >> %TEMP%\pynegative_download.py
echo             zf.extractall(temp_extract) >> %TEMP%\pynegative_download.py
echo. >> %TEMP%\pynegative_download.py
echo         subdirs = [d for d in os.listdir(temp_extract) if os.path.isdir(os.path.join(temp_extract, d))] >> %TEMP%\pynegative_download.py
echo         if not subdirs: >> %TEMP%\pynegative_download.py
echo             print('Error: No directory found in zip') >> %TEMP%\pynegative_download.py
echo             sys.exit(1) >> %TEMP%\pynegative_download.py
echo. >> %TEMP%\pynegative_download.py
echo         source_dir = os.path.join(temp_extract, subdirs[0]) >> %TEMP%\pynegative_download.py
echo. >> %TEMP%\pynegative_download.py
echo         if os.path.exists(dest_dir): >> %TEMP%\pynegative_download.py
echo             print('Removing old installation...') >> %TEMP%\pynegative_download.py
echo             shutil.rmtree(dest_dir) >> %TEMP%\pynegative_download.py
echo. >> %TEMP%\pynegative_download.py
echo         shutil.move(source_dir, dest_dir) >> %TEMP%\pynegative_download.py
echo         shutil.rmtree(temp_extract) >> %TEMP%\pynegative_download.py
echo. >> %TEMP%\pynegative_download.py
echo         # Generate icons if script exists >> %TEMP%\pynegative_download.py
echo         icon_script = os.path.join(dest_dir, 'scripts', 'generate_icons.py') >> %TEMP%\pynegative_download.py
echo         if os.path.exists(icon_script): >> %TEMP%\pynegative_download.py
echo             print('Generating icons...') >> %TEMP%\pynegative_download.py
echo             import subprocess >> %TEMP%\pynegative_download.py
echo             result = subprocess.run( >> %TEMP%\pynegative_download.py
echo                 ['uv', 'run', '--python', '3', 'python', icon_script], >> %TEMP%\pynegative_download.py
echo                 capture_output=True, >> %TEMP%\pynegative_download.py
echo                 text=True, >> %TEMP%\pynegative_download.py
echo                 cwd=dest_dir >> %TEMP%\pynegative_download.py
echo             ) >> %TEMP%\pynegative_download.py
echo             if result.returncode == 0: >> %TEMP%\pynegative_download.py
echo                 print('Icons generated successfully!') >> %TEMP%\pynegative_download.py
echo             else: >> %TEMP%\pynegative_download.py
echo                 print(f'Warning: Icon generation failed') >> %TEMP%\pynegative_download.py
echo. >> %TEMP%\pynegative_download.py
echo         with open(VERSION_FILE, 'w') as f: >> %TEMP%\pynegative_download.py
echo             f.write(version) >> %TEMP%\pynegative_download.py
echo. >> %TEMP%\pynegative_download.py
echo         print(f'Successfully installed {version} to {dest_dir}') >> %TEMP%\pynegative_download.py
echo         return True >> %TEMP%\pynegative_download.py
echo. >> %TEMP%\pynegative_download.py
echo     except Exception as e: >> %TEMP%\pynegative_download.py
echo         print(f'Error downloading: {e}') >> %TEMP%\pynegative_download.py
echo         sys.exit(1) >> %TEMP%\pynegative_download.py
echo. >> %TEMP%\pynegative_download.py
echo if __name__ == '__main__': >> %TEMP%\pynegative_download.py
echo     latest = get_latest_version() >> %TEMP%\pynegative_download.py
echo     current = None >> %TEMP%\pynegative_download.py
echo. >> %TEMP%\pynegative_download.py
echo     if os.path.exists(VERSION_FILE): >> %TEMP%\pynegative_download.py
echo         try: >> %TEMP%\pynegative_download.py
echo             with open(VERSION_FILE, 'r') as f: >> %TEMP%\pynegative_download.py
echo                 current = f.read().strip() >> %TEMP%\pynegative_download.py
echo         except: >> %TEMP%\pynegative_download.py
echo             pass >> %TEMP%\pynegative_download.py
echo. >> %TEMP%\pynegative_download.py
echo     if current == latest and os.path.exists(os.path.join(INSTALL_DIR, 'pyproject.toml')): >> %TEMP%\pynegative_download.py
echo         print(f'Already on latest version: {latest}') >> %TEMP%\pynegative_download.py
echo         sys.exit(2) >> %TEMP%\pynegative_download.py
echo     elif current: >> %TEMP%\pynegative_download.py
echo         print(f'Update available: {current} -> {latest}') >> %TEMP%\pynegative_download.py
echo     else: >> %TEMP%\pynegative_download.py
echo         print(f'Installing {latest}...') >> %TEMP%\pynegative_download.py
echo. >> %TEMP%\pynegative_download.py
echo     download_and_extract(latest, INSTALL_DIR) >> %TEMP%\pynegative_download.py
echo     sys.exit(0) >> %TEMP%\pynegative_download.py
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

REM Copy this batch file to install dir for future updates
if not exist "%INSTALL_DIR%\scripts" mkdir "%INSTALL_DIR%\scripts"
copy /Y "%~f0" "%INSTALL_DIR%\scripts\install-pynegative.bat" >nul 2>&1

echo Shortcuts created!
exit /b 0
