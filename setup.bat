@echo off
setlocal EnableExtensions EnableDelayedExpansion
rem One-time environment setup for Windows.
rem
rem Creates .venv and installs pressurekeeper into it (editable, with dev
rem extras). After this script finishes, no further setup is needed -- just
rem invoke the venv's executables directly, e.g.:
rem
rem   .venv\Scripts\pressurekeeper --config config\default.yaml --sim --target 1.0
rem
rem Usage:
rem   setup.bat          dev extras only (matches README's default flow)
rem   setup.bat --gui     also install GUI extras (PyQt6/pyqtgraph)

cd /d "%~dp0"

set "INSTALL_GUI=0"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--gui" (
    set "INSTALL_GUI=1"
    shift
    goto parse_args
)
if /I "%~1"=="-h" goto usage
if /I "%~1"=="--help" goto usage
echo Unknown argument: %~1
goto usage

:usage
echo Usage: %~nx0 [--gui]
echo   --gui   also install GUI extras ^(PyQt6/pyqtgraph^)
exit /b 1

:args_done

rem This project requires Python 3.11+. Try the py launcher pinned to 3.11
rem first, then fall back to whatever python resolves to, version-checked.
set "PYTHON_CMD="

py -3.11 -c "import sys" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.11"
    goto have_python
)

for %%C in (py python3.11 python3 python) do (
    if not defined PYTHON_CMD (
        %%C -c "import sys" >nul 2>nul
        if not errorlevel 1 (
            for /f "tokens=2 delims= " %%V in ('%%C --version 2^>^&1') do set "PYVER=%%V"
            for /f "tokens=1,2 delims=." %%A in ("!PYVER!") do (
                set "PYMAJOR=%%A"
                set "PYMINOR=%%B"
            )
            if !PYMAJOR! GEQ 3 if !PYMINOR! GEQ 11 (
                set "PYTHON_CMD=%%C"
            )
        )
    )
)

if not defined PYTHON_CMD (
    echo Error: could not find a Python 3.11+ interpreter.
    echo Install Python 3.11 or newer ^(https://www.python.org/downloads/^) and re-run this script.
    exit /b 1
)

:have_python
echo Using: !PYTHON_CMD!
!PYTHON_CMD! --version

if exist ".venv\" (
    echo .venv already exists, reusing it.
) else (
    !PYTHON_CMD! -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        exit /b 1
    )
    echo Created .venv
)

set "VENV_PY=.venv\Scripts\python.exe"

rem A pre-existing .venv created by a tool other than the stdlib venv module
rem (e.g. `uv venv`) may not ship pip. Bootstrap it if missing.
"%VENV_PY%" -m pip --version >nul 2>nul
if errorlevel 1 (
    echo pip not found in .venv, bootstrapping via ensurepip...
    "%VENV_PY%" -m ensurepip --upgrade
)

"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip.
    exit /b 1
)

if "%INSTALL_GUI%"=="1" (
    echo Installing pressurekeeper with [dev,gui] extras...
    "%VENV_PY%" -m pip install -e ".[dev,gui]"
) else (
    echo Installing pressurekeeper with [dev] extras...
    "%VENV_PY%" -m pip install -e ".[dev]"
)
if errorlevel 1 (
    echo Installation failed.
    exit /b 1
)

echo.
echo Setup complete. From now on, a single command is enough:
echo.
echo   Simulator (no hardware/network required):
echo     .venv\Scripts\pressurekeeper --config config\default.yaml --sim --target 1.0
echo.
echo   Dry-run against real APIs (reads real ruby data, never writes to PACE5000):
echo     .venv\Scripts\pressurekeeper --config config\default.yaml --target 1.0
echo.
echo   Test suite:
echo     .venv\Scripts\pytest -q
echo.

endlocal
