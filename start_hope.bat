@echo off
REM Hope DuGan AI System Startup Script

echo Starting Hope DuGan AI System...
echo Initializing...

REM Set the working directory to the Hope DuGan folder
cd /d "%~dp0"

REM Activate the Python virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Warning: No virtual environment found. Using system Python.
)

REM Install required packages if not already installed
echo Installing required packages...
pip install -r requirements.txt

REM Start the Hope DuGan AI GUI application
echo Launching Hope DuGan AI Interface...
python -m AI_Python.main_gui

pause