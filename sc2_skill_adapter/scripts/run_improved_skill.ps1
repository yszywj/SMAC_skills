param(
    [Parameter(Mandatory=$true)][ValidateSet(5,10)][int]$Size,
    [int]$Episodes = 20,
    [int]$Seed = 150,
    [string]$OutputName = "",
    [ValidateSet('v4','v5','v6','v7')][string]$PolicyVersion = 'v6',
    [switch]$EnableRegroup,
    [switch]$EnableCohesiveSearch,
    [switch]$EnableCombatPositioning,
    [switch]$AutoSpatialDispatch,
    [ValidateSet('compact','normal','spread')][string]$PositioningSpacing = 'normal',
    [ValidateSet('free','front_back')][string]$PositioningLayout = 'free'
)
$ErrorActionPreference = 'Stop'
if ($Episodes -le 0) { throw 'Episodes must be positive' }
if ($OutputName -and $OutputName -notmatch '^[a-zA-Z0-9][a-zA-Z0-9_-]*$') { throw 'OutputName must be a single directory name' }
$adapterRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $adapterRoot '.venv\Scripts\python.exe'
$manifest = Join-Path $adapterRoot "configs\protoss_${Size}_vs_${Size}_native_n${Episodes}_seed${Seed}.json"
if (-not $OutputName) {
    $OutputName = "native-${Size}v${Size}-${PolicyVersion}-skill-n${Episodes}-seed${Seed}"
}
$output = Join-Path $adapterRoot "runs\$OutputName"
if (Test-Path -LiteralPath $output) { throw "Run directory already exists: $output" }
Push-Location $adapterRoot
try {
    if (-not (Test-Path -LiteralPath $manifest)) {
        & $python run.py scenarios --size $Size --episodes $Episodes --seed $Seed --unit-balance native --output $manifest
        if ($LASTEXITCODE -ne 0) { throw 'Scenario creation failed' }
    }
    $manifestData = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
    if ($manifestData.episodes.Count -ne $Episodes -or $manifestData.seed -ne $Seed -or
        $manifestData.env_args.capability_config.n_units -ne $Size -or
        $manifestData.env_args.capability_config.n_enemies -ne $Size -or
        $manifestData.env_args.unit_balance -ne 'native') {
        throw 'Existing manifest does not match the requested protocol'
    }
    $runArguments = @(
        'run.py', 'run', '--method', 'adaptive-skill', '--manifest', $manifest, '--output-dir', $output,
        '--adaptive-posture', 'balanced', '--skill-max-steps', '4',
        '--adaptive-window-fire', '--adaptive-range-preserving-move', '--adaptive-target-scoring',
        '--adaptive-coordinated-approach', '--adaptive-safe-reentry',
        '--unit-value-mode', 'contextual', '--unit-value-spatial-context'
    )
    if ($PolicyVersion -in @('v5', 'v6', 'v7')) {
        $runArguments += @('--adaptive-accurate-damage', '--adaptive-critical-escape', '--adaptive-planned-reentry')
    }
    if ($PolicyVersion -in @('v6', 'v7')) {
        $runArguments += '--adaptive-engagement-commitment'
    }
    if ($PolicyVersion -eq 'v7') {
        $runArguments += '--adaptive-approach-progress-guard'
    }
    if ($EnableRegroup) { $runArguments += '--enable-regroup' }
    if ($EnableCohesiveSearch) { $runArguments += '--enable-cohesive-search' }
    if ($EnableCombatPositioning) { $runArguments += '--enable-combat-positioning' }
    # These local modes require a selector; enabling either also enables dispatch.
    if ($AutoSpatialDispatch -or $EnableRegroup -or $EnableCohesiveSearch) {
        $runArguments += '--auto-spatial-dispatch'
    }
    if ($EnableCombatPositioning -or $PositioningSpacing -ne 'normal' -or $PositioningLayout -ne 'free') {
        $runArguments += @('--positioning-spacing', $PositioningSpacing, '--positioning-layout', $PositioningLayout)
    }
    & $python @runArguments
    if ($LASTEXITCODE -ne 0) { throw 'Skill run failed; preserve its output for diagnosis' }
    $summary = Get-Content -LiteralPath (Join-Path $output 'summary.json') -Raw | ConvertFrom-Json
    if ($summary.status -ne 'complete' -or $summary.completed_episodes -ne $Episodes) {
        throw 'Skill run did not complete every requested episode'
    }
} finally {
    Pop-Location
}
