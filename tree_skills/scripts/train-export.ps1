# Distillation training + export. Requires a TRAINING python with scikit-learn
# (never the SMACv2 adapter venv). Refuses to overwrite existing output.
# Usage: .\train-export.ps1 -TrainShard <jsonl> -ValidEpisodes <json-list> -OutputDir <empty-dir> [-Search]
param(
    [Parameter(Mandatory = $true)][string]$TrainShard,
    [Parameter(Mandatory = $true)][string]$ValidEpisodes,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [switch]$Search
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent | Split-Path -Parent
$env:PYTHONPATH = $root
$args = @((Join-Path $PSScriptRoot "..\training\train_tree.py"),
    "--train-shard", $TrainShard, "--valid-episodes", $ValidEpisodes,
    "--output-dir", $OutputDir)
if ($Search) { $args += "--search" }
python @args
