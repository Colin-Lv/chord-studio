param([string]$ModelPython = "python")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$modelExecutable = (& $ModelPython -c "import sys; print(sys.executable)").Trim()
if ($LASTEXITCODE -ne 0) { throw "Model Python is unavailable." }
& $modelExecutable -m venv --system-site-packages .tmp/harmony_web_env
if ($LASTEXITCODE -ne 0) { throw "Cannot create the web environment." }
$webExecutable = Join-Path $PSScriptRoot '.tmp/harmony_web_env/Scripts/python.exe'
& $webExecutable -m pip install -r requirements_web.txt
if ($LASTEXITCODE -ne 0) { throw "Cannot install web dependencies." }
Set-Content -LiteralPath '.tmp/harmony_web_model_python.txt' -Value $modelExecutable -Encoding utf8
Write-Host 'Ready. Run start_harmony_web.bat, then open http://127.0.0.1:8765'
