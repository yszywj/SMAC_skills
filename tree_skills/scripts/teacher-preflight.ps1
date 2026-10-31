# Teacher checkpoint preflight. Downloads are NEVER started here:
# place the .pt files manually under the checkpoint dir first.
# Usage: .\teacher-preflight.ps1 -CheckpointDir <dir-with-.pt-files>
param(
    [Parameter(Mandatory = $true)][string]$CheckpointDir
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent | Split-Path -Parent
& (Join-Path $root "sc2_skill_adapter\.venv\Scripts\python.exe") `
    (Join-Path $PSScriptRoot "..\teacher\preflight.py") --checkpoint-dir $CheckpointDir
