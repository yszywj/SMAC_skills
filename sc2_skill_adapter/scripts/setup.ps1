param(
    [string]$PythonExe = "",
    [string]$IndexUrl = "https://pypi.org/simple"
)
$ErrorActionPreference = "Stop"
$AdapterRoot = Split-Path $PSScriptRoot -Parent
if (-not $PythonExe) {
    $PythonExe = Join-Path (Split-Path $AdapterRoot -Parent) "game_skill\.venvs\micro\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Supply -PythonExe pointing to a Python 3.10 executable."
}
& $PythonExe -c "import sys; assert sys.version_info[:2] == (3, 10), 'Use Python 3.10'"
if ($LASTEXITCODE -ne 0) { throw "Python version check failed" }
$VenvDir = Join-Path $AdapterRoot ".venv"
if (-not (Test-Path -LiteralPath (Join-Path $VenvDir "Scripts\python.exe"))) {
    & $PythonExe -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}
$Runtime = Join-Path $VenvDir "Scripts\python.exe"
$Requirements = Join-Path $AdapterRoot "requirements-lock.txt"
if (-not (Test-Path -LiteralPath $Requirements)) {
    $Requirements = Join-Path $AdapterRoot "requirements.txt"
}
& $Runtime -m pip install --index-url $IndexUrl -r $Requirements
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
& $Runtime (Join-Path $AdapterRoot "run.py") check
if ($LASTEXITCODE -ne 0) { throw "Preflight failed" }
