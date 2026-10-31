param([string]$Python = "python", [string]$Environment = ".tmp/structure_env")
$ErrorActionPreference = "Stop"
& $Python -m venv --system-site-packages $Environment
if ($LASTEXITCODE -ne 0) { throw "Unable to create the structure environment" }
$StructurePython = Join-Path $Environment "Scripts/python.exe"
& $StructurePython -m pip install "all-in-one-infer==3.1.0"
if ($LASTEXITCODE -ne 0) { throw "Unable to install the optional structure expert" }
& $StructurePython -c "import allin1_infer, torch; print('Structure expert ready; torch', torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Structure expert import check failed" }
Write-Output "Run: $StructurePython app/structure_expert.py <analysis.json> -o <output.json> --html"
