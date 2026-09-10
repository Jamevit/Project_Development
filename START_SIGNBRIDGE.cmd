@echo off
cd /d "%~dp0"
echo Open http://127.0.0.1:5010 after the ASL-HG model is ready.
echo Keep this window open during practice. Press Ctrl+C to stop.
"%USERPROFILE%\Documents\Codex\aslhg-env\Scripts\python.exe" -B server.py
pause
