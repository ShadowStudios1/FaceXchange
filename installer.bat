@echo off
chcp 65001 >nul
title FaceXchange Setup

echo.
echo   === FaceXchange Installer ===
echo   Open-source face swap for everyone
echo   Local . Private . Free . Unlimited
echo.

:: Check Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found!
    echo.
    echo FaceXchange requires Python 3.10 or higher.
    echo Download from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

:: Check Python version
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python 3.10+ required
    python --version
    echo.
    echo Please install Python 3.10 or newer from:
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Bootstrap pip packages
echo [..] Installing setup tools (rich, questionary, requests)...
python -m pip install --quiet --upgrade pip >nul 2>&1
python -m pip install --quiet rich questionary requests
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install setup dependencies.
    echo.
    echo Try running manually:
    echo   python -m pip install rich questionary requests
    echo.
    pause
    exit /b 1
)

echo [OK] Ready to install!
timeout /t 1 /nobreak >nul

:: Launch installer
python install.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo Installer exited with an error.
    pause
)
