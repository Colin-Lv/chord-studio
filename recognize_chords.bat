@echo off
setlocal

if "%~1"=="" (
  echo Usage:
  echo   recognize_chords.bat "audio-file.m4a"
  echo.
echo Example:
  echo   recognize_chords.bat "projects\panopticon\song.m4a"
  exit /b 1
)

python "%~dp0app\chord_studio.py" %*
