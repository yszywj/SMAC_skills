param(
    [Parameter(Mandatory=$true)][ValidateSet('A', 'B', 'C', 'D')][string]$Arm,
    [Parameter(Mandatory=$true)][string]$Manifest,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [string]$CollectDir,
    [string]$Sc2Path = 'C:\Program Files (x86)\StarCraft II'
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$adapterRoot = Join-Path $repoRoot 'sc2_skill_adapter'
$budget = Get-Content (Join-Path $repoRoot 'tree_skills/configs/default_budget.json') -Raw | ConvertFrom-Json
$manifestPath = (Resolve-Path -LiteralPath $Manifest).Path
$manifestData = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifestData.episodes.Count -gt $budget.sc2.max_episodes_per_run) { throw 'Episode budget exceeded.' }
if (Get-Process SC2_x64 -ErrorAction SilentlyContinue) { throw 'SC2 already running; one instance required.' }
$checkpoint = Join-Path $repoRoot 'tree_skills/teacher/checkpoints/marl-gpt-main.pt'
$commonArgs = @('run', '--method', 'adaptive-skill', '--manifest', $manifestPath,
    '--sc2-path', $Sc2Path, '--race-profile', 'terran', '--terran-keep-heal-order',
    '--unit-value-mode', 'contextual', '--adaptive-window-fire',
    '--adaptive-range-preserving-move', '--adaptive-target-scoring',
    '--adaptive-coordinated-approach', '--adaptive-safe-reentry',
    '--adaptive-accurate-damage', '--adaptive-critical-escape',
    '--adaptive-planned-reentry', '--adaptive-engagement-commitment',
    '--adaptive-approach-progress-guard', '--output-dir', $OutputDir)
if ($Arm -eq 'B') {
    if (-not $CollectDir) { throw 'B requires a new CollectDir.' }
    $commonArgs += @('--pair-mode', 'teacher-score', '--pair-failure', 'strict',
        '--pair-teacher-checkpoint', $checkpoint, '--pair-last-token', 'on', '--pair-collect', $CollectDir)
}
if ($Arm -in @('C', 'D')) {
    $control = if ($Arm -eq 'C') { 'medivac' } else { 'all' }
    $commonArgs += @('--native-teacher-control', $control,
        '--native-teacher-checkpoint', $checkpoint, '--native-teacher-last-token', 'on')
}
Push-Location $adapterRoot
try {
    & (Join-Path $adapterRoot '.venv/Scripts/python.exe') 'run.py' @commonArgs
    if ($LASTEXITCODE -ne 0) { throw "Experiment failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
