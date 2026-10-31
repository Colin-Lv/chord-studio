@echo off
setlocal
if "%~1"=="" (
  echo Usage: recognize_harmony.bat "audio-file.mp3" [options]
  exit /b 1
)
python "%~dp0app\analyze_with_madmom.py" %* --open
