@echo off
REM ============================================================
REM BankAI Pro - ML Service Startup Script (Windows)
REM ============================================================
REM Usage: Double-click this file OR run from Command Prompt
REM ============================================================

echo.
echo ============================================================
echo   BankAI Pro - ML Service
echo   Starting on http://localhost:8000
echo ============================================================
echo.

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Run: python -m venv venv
    echo Then: venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM Check if .env exists
if not exist ".env" (
    echo [WARNING] .env file not found. Copying from .env.example...
    copy ".env.example" ".env"
    echo Please edit .env and add your GOOGLE_API_KEY before using AI features.
    echo.
)

REM Activate venv and start server
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Starting FastAPI server...
echo API Docs: http://localhost:8000/docs
echo.

venv\Scripts\uvicorn main:app --reload --port 8000 --host 0.0.0.0

pause
