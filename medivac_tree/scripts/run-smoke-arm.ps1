param(
    [Parameter(Mandatory=$true)][ValidateSet('legacy_off', 'tree', 'legacy_on', 'shadow')][string]$Arm,
    [Parameter(Mandatory=$true)][string]$StudyDir,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [string]$Tree,
    [string]$Sc2Path = 'C:\Program Files (x86)\StarCraft II'
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$adapterRoot = Join-Path $repoRoot 'sc2_skill_adapter'
$study = (Resolve-Path -LiteralPath $StudyDir).Path
$protocol = Get-Content -LiteralPath (Join-Path $study 'protocol.json') -Raw | ConvertFrom-Json
$manifestPath = Join-Path $study 'manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($protocol.max_instances -ne 1 -or $manifest.episodes.Count -ne 4 -or
    $manifest.episodes.Count -gt $protocol.max_episodes_per_run -or
    $protocol.max_episodes_per_run -gt 40 -or $manifest.sha256 -ne $protocol.manifest_sha256) {
    throw 'Smoke protocol requires exactly four frozen episodes and one SC2 instance.'
}
$exitDeadline = (Get-Date).AddSeconds(30)
while (Get-Process SC2_x64 -ErrorAction SilentlyContinue) {
    if ((Get-Date) -ge $exitDeadline) { throw 'SC2 is still running; no concurrent instance will be started.' }
    Start-Sleep -Milliseconds 250
}
$policy = switch ($Arm) { 'tree' { 'tree' }; 'shadow' { 'tree-shadow' }; default { 'legacy' } }
$coordination = if ($Arm -eq 'legacy_on') { 'on' } else { 'off' }
$resolvedOutput = if ([IO.Path]::IsPathRooted($OutputDir)) {
    [IO.Path]::GetFullPath($OutputDir)
} else { [IO.Path]::GetFullPath((Join-Path $adapterRoot $OutputDir)) }
if (Test-Path -LiteralPath $resolvedOutput) { throw "Refusing to overwrite $resolvedOutput" }
$runArgs = @('run', '--method', 'adaptive-skill', '--manifest', $manifestPath,
    '--sc2-path', $Sc2Path, '--race-profile', 'terran', '--terran-keep-heal-order',
    '--unit-value-mode', 'contextual', '--adaptive-window-fire',
    '--adaptive-range-preserving-move', '--adaptive-target-scoring',
    '--adaptive-coordinated-approach', '--adaptive-safe-reentry',
    '--adaptive-accurate-damage', '--adaptive-critical-escape',
    '--adaptive-planned-reentry', '--adaptive-engagement-commitment',
    '--adaptive-approach-progress-guard', '--pair-mode', 'off',
    '--native-teacher-control', 'off', '--medivac-policy', $policy,
    '--medical-patient-coordination', $coordination, '--output-dir', $resolvedOutput)
if ($policy -ne 'legacy') {
    if (-not $Tree) { throw 'Tree and shadow arms require an explicit frozen JSON tree path.' }
    $runArgs += @('--medivac-tree', (Resolve-Path -LiteralPath $Tree).Path)
}
Push-Location $adapterRoot
try {
    & (Join-Path $adapterRoot '.venv/Scripts/python.exe') 'run.py' @runArgs
    if ($LASTEXITCODE -ne 0) { throw "Smoke arm failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
