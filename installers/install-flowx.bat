@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "REPO_URL=https://github.com/rafsanlab/flowx.git"
set "ZIP_URL=https://github.com/rafsanlab/flowx/archive/refs/heads/main.zip"
if defined FLOWX_INSTALL_DIR (
    set "DEFAULT_INSTALL_DIR=%FLOWX_INSTALL_DIR%"
) else (
    set "DEFAULT_INSTALL_DIR=%USERPROFILE%\flowx"
)

echo === FlowX Installer (Windows) ===

REM 1. Ensure uv is installed
set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
where uv >nul 2>&1
if errorlevel 1 (
    echo [1/4] Installing uv...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 goto :error
    set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
) else (
    echo [1/4] uv is already installed.
)

REM 2. Locate or download FlowX repository
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%I in ("%SCRIPT_DIR%\..") do set "PARENT_DIR=%%~fI"

if exist "%PARENT_DIR%\pyproject.toml" (
    findstr /C:"name = \"flowx\"" "%PARENT_DIR%\pyproject.toml" >nul 2>&1
    if not errorlevel 1 (
        set "FLOWX_DIR=%PARENT_DIR%"
        echo [2/4] Using local FlowX repository at: !FLOWX_DIR!
        goto :setup_venv
    )
)

set "FLOWX_DIR=%DEFAULT_INSTALL_DIR%"
if exist "%FLOWX_DIR%\pyproject.toml" (
    echo [2/4] FlowX directory already exists at: %FLOWX_DIR%
) else (
    where git >nul 2>&1
    if not errorlevel 1 (
        echo [2/4] Cloning FlowX into %FLOWX_DIR%...
        git clone "%REPO_URL%" "%FLOWX_DIR%"
        if errorlevel 1 goto :error
    ) else (
        echo [2/4] Downloading FlowX archive into %FLOWX_DIR%...
        powershell -NoProfile -ExecutionPolicy Bypass -Command ^
            "$zip = Join-Path $env:TEMP 'flowx-main.zip'; " ^
            "$ext = Join-Path $env:TEMP 'flowx-extract'; " ^
            "if (Test-Path $ext) { Remove-Item -Recurse -Force $ext }; " ^
            "Invoke-WebRequest -Uri '%ZIP_URL%' -OutFile $zip; " ^
            "Expand-Archive -Path $zip -DestinationPath $ext -Force; " ^
            "$sub = Get-ChildItem -Path $ext -Directory | Select-Object -First 1; " ^
            "Move-Item -Path $sub.FullName -Destination '%FLOWX_DIR%'; " ^
            "Remove-Item -Force $zip; " ^
            "Remove-Item -Recurse -Force $ext"
        if errorlevel 1 goto :error
    )
)

:setup_venv
cd /d "%FLOWX_DIR%" || goto :error

REM 3. Create .venv (uv downloads Python 3.11 automatically if needed) & install core flowx
echo [3/4] Setting up virtual environment and installing core FlowX...
uv venv --python 3.11 .venv
if errorlevel 1 goto :error

uv pip install --python .venv\Scripts\python.exe -e .
if errorlevel 1 goto :error

REM 4. Create Desktop and Start Menu shortcuts with custom icons
echo [4/4] Creating FlowX Terminal and FlowX Folder shortcuts...
.venv\Scripts\python.exe installers\setup_desktop.py "%FLOWX_DIR%"
if errorlevel 1 goto :error

echo.
echo FlowX installation complete!
echo   Directory: %FLOWX_DIR%
echo   Shortcuts: 'FlowX Terminal' and 'FlowX Folder' added to your Desktop and Start Menu.
echo.
pause
exit /b 0

:error
echo.
echo Installation failed. Please check the error output above.
pause
exit /b 1
