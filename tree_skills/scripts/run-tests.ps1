# tree_skills test entry points. Run from SMAC_agent root.
# Channel/feature tests use the SMACv2 adapter venv (deployment path);
# rule-export equivalence uses a training python with scikit-learn.
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent | Split-Path -Parent   # SMAC_agent
Set-Location $root
$env:PYTHONPATH = $root
$venv = Join-Path $root "sc2_skill_adapter\.venv\Scripts\python.exe"

Write-Host "== channel + features (adapter venv) =="
& $venv -m unittest tree_skills.tests.test_pair_scorer_channel tree_skills.tests.test_features

Write-Host "== rule export equivalence (training python) =="
python -m unittest tree_skills.tests.test_rule_model

Write-Host "== baseline regression (adapter venv) =="
$env:PYTHONPATH = $root
& $venv -m unittest discover -s (Join-Path $root "skills\current\tests") -p "test_*.py"
& $venv -m unittest skills.current.tests.test_race_movement_guard skills.current.tests.test_spatial_review_regressions
