$ErrorActionPreference = "Stop"

$py39 = "$env:LOCALAPPDATA\Programs\Python\Python39\python.exe"
$envPath = "$env:USERPROFILE\.codex\acr_envs\madmom39"

if (-not (Test-Path -LiteralPath $py39)) {
    throw "Python 3.9 not found at $py39"
}

& $py39 -m venv $envPath
& "$envPath\Scripts\python.exe" -m pip install --upgrade pip "setuptools<81" wheel
& "$envPath\Scripts\python.exe" -m pip install "Cython<3" "numpy==1.23.5" "scipy==1.10.1" mido
& "$envPath\Scripts\python.exe" -m pip install madmom==0.16.1 --no-build-isolation

Write-Host "madmom environment ready: $envPath"
