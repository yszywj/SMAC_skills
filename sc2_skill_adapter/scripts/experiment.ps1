param(
    [ValidateSet(5, 10)][int]$Size = 5,
    [int]$Episodes = 3,
    [int]$Seed = 1,
    [ValidateSet("nearest", "random", "jev", "astra-jev", "suite")][string]$Method = "nearest",
    [string]$Sc2Path = "C:\Program Files (x86)\StarCraft II",
    [string]$Manifest = "",
    [string]$OutputDir = "",
    [string]$PlannerModel = "gpt-6-astra",
    [string]$Model = "jev-1.13.0",
    [ValidateSet("original", "native")][string]$UnitBalance = "original"
)
$ErrorActionPreference = "Stop"
$AdapterRoot = Split-Path $PSScriptRoot -Parent
$Runtime = Join-Path $AdapterRoot ".venv\Scripts\python.exe"
$Entry = Join-Path $AdapterRoot "run.py"
if (-not $Manifest) {
    $Manifest = Join-Path $AdapterRoot "configs\protoss_${Size}_vs_${Size}_${UnitBalance}_n${Episodes}_seed${Seed}.json"
    if (-not (Test-Path -LiteralPath $Manifest)) {
        & $Runtime $Entry scenarios --size $Size --episodes $Episodes --seed $Seed --unit-balance $UnitBalance --output $Manifest
        if ($LASTEXITCODE -ne 0) { throw "Scenario generation failed" }
    }
}
if (-not $OutputDir) {
    $RunStamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
    $OutputDir = Join-Path $AdapterRoot "runs\${Size}v${Size}-${Method}-${RunStamp}"
}
$RunArgs = @("--manifest", $Manifest, "--output-dir", $OutputDir,
    "--sc2-path", $Sc2Path, "--planner-model", $PlannerModel, "--model", $Model)
if ($Method -eq "suite") {
    & $Runtime $Entry suite @RunArgs
} else {
    & $Runtime $Entry run --method $Method @RunArgs
}
if ($LASTEXITCODE -ne 0) { throw "Run failed; inspect $OutputDir\summary.json" }
Write-Output "Results: $OutputDir"
