param(
    [Parameter(Mandatory=$true)][ValidateSet(5,10)][int]$Size,
    [Parameter(Mandatory=$true)][ValidateSet('current','baseline_v2_70pct','baseline_v6_20261024')][string]$SkillVersion,
    [int]$Episodes = 20,
    [int]$Seed = 170,
    [Parameter(Mandatory=$true)][string]$OutputName,
    [switch]$EnableRegroup,
    [switch]$EnableCohesiveSearch,
    [switch]$EnableCombatPositioning,
    [switch]$AutoSpatialDispatch,
    [ValidateSet('compact','normal','spread')][string]$PositioningSpacing = 'normal',
    [ValidateSet('free','front_back')][string]$PositioningLayout = 'free'
)
$ErrorActionPreference = 'Stop'
if ($Episodes -le 0) { throw 'Episodes must be positive' }
if ($SkillVersion -ne 'current' -and ($EnableRegroup -or $EnableCohesiveSearch -or
    $EnableCombatPositioning -or $AutoSpatialDispatch -or $PositioningSpacing -ne 'normal' -or
    $PositioningLayout -ne 'free')) { throw 'Historical skill versions do not support spatial switches' }
if ($OutputName -notmatch '^[a-zA-Z0-9][a-zA-Z0-9_-]*$') { throw 'OutputName must be a single directory name' }
$adapterRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $adapterRoot '.venv\Scripts\python.exe'
$manifest = Join-Path $adapterRoot "configs\protoss_${Size}_vs_${Size}_native_n${Episodes}_seed${Seed}.json"
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
        'run.py', 'run', '--method', 'adaptive-skill', '--skill-version', $SkillVersion,
        '--manifest', $manifest, '--output-dir', $output,
        '--adaptive-posture', 'balanced', '--skill-max-steps', '4',
        '--adaptive-window-fire', '--adaptive-range-preserving-move', '--adaptive-target-scoring'
    )
    if ($SkillVersion -in @('current', 'baseline_v6_20261024')) {
        $runArguments += @(
            '--adaptive-coordinated-approach', '--adaptive-safe-reentry',
            '--adaptive-accurate-damage', '--adaptive-critical-escape', '--adaptive-planned-reentry',
            '--adaptive-engagement-commitment', '--unit-value-mode', 'contextual',
            '--unit-value-spatial-context'
        )
    }
    if ($EnableRegroup) { $runArguments += '--enable-regroup' }
    if ($EnableCohesiveSearch) { $runArguments += '--enable-cohesive-search' }
    if ($EnableCombatPositioning) { $runArguments += '--enable-combat-positioning' }
    if ($AutoSpatialDispatch -or $EnableRegroup -or $EnableCohesiveSearch) {
        $runArguments += '--auto-spatial-dispatch'
    }
    if ($EnableCombatPositioning -or $PositioningSpacing -ne 'normal' -or $PositioningLayout -ne 'free') {
        $runArguments += @('--positioning-spacing', $PositioningSpacing, '--positioning-layout', $PositioningLayout)
    }
    & $python @runArguments
    if ($LASTEXITCODE -ne 0) { throw 'Skill run failed; preserve its output for diagnosis' }
    $summary = Get-Content -LiteralPath (Join-Path $output 'summary.json') -Raw | ConvertFrom-Json
    if ($summary.status -ne 'complete' -or $summary.completed_episodes -ne $Episodes -or
        $summary.skill_version -ne $SkillVersion) {
        throw 'Skill run did not complete every requested episode with the selected version'
    }
} finally {
    Pop-Location
}
