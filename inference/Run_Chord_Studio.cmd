@echo off
"C:\Users\33573\miniconda3\python.exe" -I -B "%~dp0test_audio.py"
if errorlevel 1 echo Test failed. See the preserved logs.
pause
