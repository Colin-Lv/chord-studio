@echo off
setlocal
cd /d "%~dp0"
if exist ".tmp\harmony_web_env\Scripts\python.exe" (
  ".tmp\harmony_web_env\Scripts\python.exe" app\start_web.py --open
) else (
  python app\start_web.py --open
)
if errorlevel 1 pause
