param(
    [Parameter(Mandatory=$true)][string]$StudyDir,
    [Parameter(Mandatory=$true)][string]$RunPrefix,
    [ValidateSet('A', 'B', 'C', 'D')][string[]]$Arms = @('A', 'B', 'C', 'D')
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$study = (Resolve-Path -LiteralPath $StudyDir).Path
$protocol = Get-Content -LiteralPath (Join-Path $study 'protocol.json') -Raw | ConvertFrom-Json
if ($protocol.max_instances -ne 1 -or $protocol.formal_episodes_per_arm -ne 40) {
    throw 'Unexpected frozen protocol budget.'
}
foreach ($arm in $Arms) {
    # env.close() can return just before the SC2 child process exits.
    # Wait for graceful shutdown; never kill another experiment's process.
    $exitDeadline = (Get-Date).AddSeconds(30)
    while (Get-Process SC2_x64 -ErrorAction SilentlyContinue) {
        if ((Get-Date) -ge $exitDeadline) { throw 'SC2 did not exit within 30 seconds.' }
        Start-Sleep -Milliseconds 250
    }
    $log = Join-Path $study ('formal-' + $arm + '.log')
    if (Test-Path -LiteralPath $log) { throw "Refusing to overwrite $log" }
    $parameters = @{
        Arm = $arm
        Manifest = Join-Path $study 'manifest.json'
        OutputDir = $RunPrefix + $arm
    }
    if ($arm -eq 'B') { $parameters.CollectDir = Join-Path $study 'B-collection' }
    @{phase='running'; arm=$arm; started=(Get-Date -Format o)} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $study 'suite-status.json') -Encoding utf8
    try {
        & (Join-Path $PSScriptRoot 'run-teacher-diagnostic.ps1') @parameters *> $log
    } catch {
        @{phase='failed'; arm=$arm; error=$_.Exception.Message; failed=(Get-Date -Format o)} |
            ConvertTo-Json | Set-Content -LiteralPath (Join-Path $study 'suite-status.json') -Encoding utf8
        throw
    }
}
@{phase='complete'; arms=$Arms; finished=(Get-Date -Format o)} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $study 'suite-status.json') -Encoding utf8
