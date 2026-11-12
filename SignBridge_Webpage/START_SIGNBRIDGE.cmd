@echo off
setlocal
cd /d "%~dp0."
set "PYTHON=%LocalAppData%\Programs\Python\Python311\python.exe"
set "COMVISION_DIR=%~dp0..\SignBridge_V2 - Copy\hand-gesture-recognition-mediapipe"
if not exist "%PYTHON%" (
    echo Python environment not found: %PYTHON%
    pause
    exit /b 1
)
if not exist "%COMVISION_DIR%\web_api.py" (
    echo API not found: %COMVISION_DIR%\web_api.py
    pause
    exit /b 1
)
echo SignBridge website and LSTM use one server.
echo Wait for "SignBridge LSTM API ready", then open http://127.0.0.1:5011.
echo Keep this window open. Press Ctrl+C to stop.
cd /d "%COMVISION_DIR%"
"%PYTHON%" -B -u web_api.py
pause
