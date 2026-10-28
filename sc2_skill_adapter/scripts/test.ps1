$ErrorActionPreference = "Stop"
$AdapterRoot = Split-Path $PSScriptRoot -Parent
Push-Location $AdapterRoot
try {
    & ".\.venv\Scripts\python.exe" -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "Offline tests failed" }
} finally { Pop-Location }
