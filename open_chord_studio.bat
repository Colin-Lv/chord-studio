@echo off
setlocal
if "%~1"=="" (
  echo Drag an analysis JSON onto this file to open the current workbench.
  echo Usage: open_chord_studio.bat "song_analysis.json"
  exit /b 1
)
python "%~dp0app\open_studio.py" "%~1" --open
