
@echo off
echo ============================================================
echo TDS RAG Application - EXE Builder
echo ============================================================
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found. Using system Python.
)

echo.
echo Starting build process...
python build_single_exe.py

echo.
echo Build process completed.
pause