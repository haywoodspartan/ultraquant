@echo off
setlocal EnableDelayedExpansion
rem ===========================================================================
rem  UltraQuant - one-click launcher
rem  Double-click this file to open the desktop app.
rem  Optional argument: a session folder (defaults to .\uq_home).
rem ===========================================================================

cd /d "%~dp0"

rem -- Find a Python. Prefer pythonw.exe so no console window lingers behind
rem -- the GUI; fall back to python.exe if the windowed launcher is missing.
set "PYW="
set "PY="

where pythonw.exe >nul 2>&1 && set "PYW=pythonw.exe"
where python.exe  >nul 2>&1 && set "PY=python.exe"

if not defined PYW (
    where py.exe >nul 2>&1 && (
        set "PYW=py.exe -3w"
        if not defined PY set "PY=py.exe -3"
    )
)

if not defined PYW if not defined PY (
    echo.
    echo   Python was not found on this machine.
    echo.
    echo   Install Python 3.11 or newer from https://www.python.org/downloads/
    echo   and be sure to tick "Add python.exe to PATH" during setup.
    echo.
    pause
    exit /b 1
)

rem -- Verify Tkinter is present before launching windowed, otherwise the app
rem -- would fail silently with no console to explain why.
if defined PY (
    %PY% -c "import tkinter" >nul 2>&1
    if errorlevel 1 (
        echo.
        echo   This Python has no Tkinter, so the desktop app cannot start.
        echo   Re-run the Python installer and enable "tcl/tk and IDLE",
        echo   or use the terminal interface instead:
        echo.
        echo       python -m ultraquant.interpreter.chat
        echo.
        pause
        exit /b 1
    )
)

if defined PYW (
    start "" %PYW% -m ultraquant.gui %*
) else (
    %PY% -m ultraquant.gui %*
)
exit /b 0
