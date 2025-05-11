@echo off
REM Activate the virtual environment
call env\Scripts\activate.bat

REM Run the Python script
python app.py

REM Keep the command prompt open
pause
