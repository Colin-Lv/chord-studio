@echo off
cd /d "%~dp0.."
"tools\fl-studio-mcp\.venv312\Scripts\python.exe" "fl_studio_integration\verify_connection.py"
echo.
echo Report: fl_studio_integration\verification.json
pause
