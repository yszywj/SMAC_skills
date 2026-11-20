param(
    [ValidateRange(1, 20)][int]$GamesPerGroup = 20,
    [ValidateSet('protoss', 'terran', 'zerg')][string[]]$Races = @('protoss', 'terran', 'zerg'),
    [string]$BaselinePath = '',
    [string]$OutputName = '',
    [switch]$Resume,
    [switch]$DryRun,
    [Alias('Group')][string]$Composition = ''
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$selectedRaces = @(@('protoss', 'terran', 'zerg') | Where-Object { $_ -in $Races })
if (-not $selectedRaces.Count -or $selectedRaces.Count -ne $Races.Count) {
    throw 'Races must contain one or more distinct race names.'
}
if ($Resume -and (-not $PSBoundParameters.ContainsKey('OutputName') -or -not $OutputName)) {
    throw 'Resume requires an explicit existing OutputName.'
}
$allGroups = @('p_6z4s', 'p_4z5s1c', 'p_2z6s2c', 't_6m4r', 't_4m5r1h',
               't_2m6r2h', 'z_6l4h', 'z_4l5h1b', 'z_2l6h2b')
$raceByPrefix = @{ p = 'protoss'; t = 'terran'; z = 'zerg' }
$groups = @($allGroups | Where-Object { $raceByPrefix[$_.Substring(0, 1)] -in $selectedRaces })
if ($Composition) {
    $groups = @($groups | Where-Object { $_ -eq $Composition })
    if ($groups.Count -eq 0) { throw "Unknown composition or composition outside selected races: $Composition" }
    $selectedRaces = @($raceByPrefix[$groups[0].Substring(0, 1)])
}
if (-not $OutputName) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    if ($Composition) { $OutputName = "tactical_${GamesPerGroup}x1_$($groups[0])_$stamp" }
    else { $OutputName = ('tactical_{0}x{1}x3_{2}' -f $GamesPerGroup, $selectedRaces.Count, $stamp) }
}
if ($OutputName -notmatch '^[a-zA-Z0-9][a-zA-Z0-9_-]*$') {
    throw 'OutputName must be a single directory name.'
}
$adapterRoot = Split-Path -Parent $PSScriptRoot
$launcherPath = $PSCommandPath
$repoRoot = Split-Path -Parent $adapterRoot
$python = Join-Path $adapterRoot '.venv\Scripts\python.exe'
$prior = Join-Path $repoRoot 'runs\tactical_all_races_10v10_9x20_mechanisms_20261115'
if (-not $BaselinePath) { $BaselinePath = Join-Path $repoRoot 'runs\tactical_10x3x3_20261115_212459' }
$comparisonRoot = (Resolve-Path -LiteralPath $BaselinePath -ErrorAction Stop).ProviderPath
$outputRoot = Join-Path $repoRoot ('runs\' + $OutputName)
function Read-Json { param([string]$Path)
    return (Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json)
}
function Write-Json { param($Value, [string]$Path)
    [System.IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 30),
        (New-Object System.Text.UTF8Encoding($false)))
}
function Get-Sha { param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
# Pure standard-library auditing; this helper never imports or starts the game.
# Python uses exactly bridge.config.digest's canonical JSON, including floats.
$auditSource = @'
import hashlib, json, struct, sys
from pathlib import Path

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(',', ':')).encode()).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def native_replay_header(path):
    # SC2 normally prefixes its archive with MPQ user data (1B). Follow its
    # declared offset to the archive header (1A), as mpyq does; do not scan for
    # arbitrary magic bytes or accept a truncated signature-only file.
    size = path.stat().st_size
    def check(condition, reason):
        require(condition, 'Invalid native replay header: ' + path.name + ': ' + reason)
    with path.open('rb') as stream:
        prefix = stream.read(16)
        magic = prefix[:4]
        offset = 0
        if magic == b'MPQ\x1b':
            check(len(prefix) == 16, 'truncated user-data header')
            _, user_size, offset, content_size = struct.unpack('<4sIII', prefix)
            check(16 <= offset <= size - 32, 'archive offset outside file')
            check(content_size <= user_size and 16 + content_size <= offset,
                  'user data overlaps archive')
        else:
            check(magic == b'MPQ\x1a', 'unrecognized signature')
        stream.seek(offset)
        header = stream.read(32)
        check(len(header) == 32, 'truncated archive header')
        archive_magic, header_size, archive_size, *_ = struct.unpack('<4sIIHHIIII', header)
        check(archive_magic == b'MPQ\x1a', 'missing archive at declared offset')
        check(32 <= header_size <= archive_size and offset + archive_size <= size,
              'archive size outside file')
    return dict(user_data=magic == b'MPQ\x1b', archive_offset=offset,
                header_size=header_size, archive_size=archive_size)

def manifest(path, count):
    value = read(path)
    require(value.get('version') == 1 and value.get('sha256') ==
            digest({k: v for k, v in value.items() if k != 'sha256'}), 'Manifest version/hash mismatch')
    require(len(value['episodes']) == 20, 'Expected original 20-scenario manifest')
    for i, row in enumerate(value['episodes']):
        require(row['index'] == i and row['sc2_seed'] == 170 + i, 'Seed mismatch')
        require(row['sha256'] == digest(row['episode_config']), 'Scenario hash mismatch')
    value['episodes'] = value['episodes'][:count]
    value['sha256'] = digest({k: v for k, v in value.items() if k != 'sha256'})
    return value

def rows_by_id(rows, scenarios):
    result = {}
    for row in rows:
        i = row.get('episode')
        require(type(i) is int and 1 <= i <= len(scenarios) and i not in result,
                'Invalid or duplicate episode')
        require(row.get('outcome') in ('win', 'loss', 'draw', 'timeout') and
                row.get('initial_state_sha256') and row.get('scenario_sha256') == scenarios[i-1]['sha256'],
                'Invalid episode evidence/scenario')
        result[i] = row
    require(sorted(result) == list(range(1, len(result) + 1)), 'Non-contiguous completed episodes')
    return result

def metadata(directory, frozen, expected_manifest, summary=None):
    run = read(directory / 'run.json')
    require(run.get('env_args') == expected_manifest['env_args'] and
            run.get('scenario_count') == len(expected_manifest['episodes']), 'Run environment/scenario count mismatch')
    expected_skills = {k: v for k, v in frozen.items() if k.startswith('skills/tactical_v1/') and k.endswith('.py')}
    require(bool(expected_skills), 'Empty frozen skill source')
    for label, value in [('run', run)] + ([('summary', summary)] if summary else []):
        require(value.get('manifest_sha256') == expected_manifest['sha256'], label + ' manifest hash mismatch')
        require(value.get('method') == 'adaptive-skill' and value.get('skill_version') == 'tactical_v1',
                label + ' policy mismatch')
        params = value.get('skill_parameters', {})
        require(params.get('posture') == 'balanced' and params.get('max_steps') == 4,
                label + ' skill parameters mismatch')
        require(value.get('skill_source_hashes') == expected_skills, label + ' skill source hash mismatch')
    require(bool(run.get('source_hashes')), 'Missing runner source hashes')
    for key, value in run['source_hashes'].items():
        key = key.replace(chr(92), '/')
        if '/smacv2/env/starcraft2/' in key:
            key = 'dependencies/smacv2/' + key.split('/smacv2/env/starcraft2/', 1)[1]
        require(frozen.get(key) == value, 'Runner source hash mismatch: ' + key)

def baseline(directory, source, count, freeze):
    # The immediately preceding version may have fewer scheduled episodes.
    # Compare only its signed prefix; never backfill with a different policy.
    if not directory.exists():
        # A failed batch may never have started a later group. Preserve the
        # absence as unpaired evidence instead of mixing an older policy in.
        batch_path = directory.parent.parent / 'matrix_summary.json'
        require(batch_path.is_file(), 'Missing baseline batch evidence')
        recorded = next((row for row in read(batch_path).get('results', [])
                         if row.get('group') == directory.name), None)
        require(recorded is None or recorded.get('completed_games', 0) == 0,
                'Baseline group directory missing despite recorded completed games')
        return dict(rows=[], available=0, missing_seeds=list(range(170, 170+count)),
                    status='not_started', evidence_hashes={name: None for name in
                    ('episodes.json', 'summary.json', 'run.json')})
    scheduled = read(directory / 'run.json').get('scenario_count')
    require(type(scheduled) is int and 1 <= scheduled <= 20,
            'Invalid baseline scenario count')
    original = manifest(source, scheduled)
    summary = read(directory / 'summary.json')
    rows = read(directory / 'episodes.json')
    by_id = rows_by_id(rows, original['episodes'])
    require(summary.get('completed_episodes') == len(rows) and
            (summary.get('status') == 'failed' or
             (summary.get('status') == 'complete' and len(rows) == scheduled)), 'Invalid baseline completion evidence')
    metadata(directory, read(freeze)['source_hashes'], original, summary)
    selected = [row for row in rows if row['episode'] <= count]
    return dict(rows=selected, available=len(selected), missing_seeds=[169+i for i in range(1, count+1) if i not in by_id],
                evidence_hashes={name: sha(directory / name) for name in ('episodes.json', 'summary.json', 'run.json')})

def audit(directory, selected, freeze, evidence, group):
    scenarios = selected['episodes']
    rows = read(directory / 'episodes.json') if (directory / 'episodes.json').exists() else []
    by_id = rows_by_id(rows, scenarios)
    summary = read(directory / 'summary.json') if (directory / 'summary.json').exists() else None
    if summary:
        require(summary.get('completed_episodes') == len(rows), 'Summary/completed episode mismatch')
    if rows or (directory / 'run.json').exists():
        metadata(directory, read(freeze)['source_hashes'], selected, summary)
    starts, ends, steps, invalid = {}, {}, {}, set()
    events = directory / 'events.jsonl'
    if events.exists():
        with events.open(encoding='utf-8') as stream:
            for line in stream:
                event = json.loads(line)
                kind, i = event.get('event'), event.get('episode')
                if kind == 'episode_started':
                    require(type(i) is int and 1 <= i <= len(scenarios) and i not in starts,
                            'Invalid or duplicate episode_started')
                    starts[i] = event
                elif kind == 'episode_completed':
                    require(i not in ends, 'Duplicate episode_completed')
                    ends[i] = event
                elif kind == 'step':
                    feedback = event.get('feedback', {})
                    before, after = feedback.get('from_game_loop'), feedback.get('to_game_loop')
                    prior_steps = steps.setdefault(i, [])
                    good = (type(before) is int and type(after) is int and after - before == 8 and
                            event.get('game_loop') == before and event.get('step') == len(prior_steps)+1 and
                            (not prior_steps or (prior_steps[-1][1] == before and not prior_steps[-1][2])))
                    if not good:
                        invalid.add(i)
                    prior_steps.append((before, after, event.get('terminated')))
    require(set(ends) == set(by_id), 'Completed event/record mismatch')
    require(sorted(starts) == list(range(1, len(starts)+1)) and len(starts) <= len(rows)+1 and
            set(steps) <= set(starts), 'Unexpected episode event sequence')
    for i, row in by_id.items():
        require(i in starts and i not in invalid and len(steps.get(i, [])) == row.get('steps') and
                bool(steps[i]) and steps[i][-1][2] is True, 'Invalid completed episode clock/termination')
        require(all(ends[i].get(k) == v for k, v in row.items()), 'Completed event differs from record')
        require(all(starts[i].get(k) == row[k] for k in ('scenario_sha256', 'initial_state_sha256')),
                'Initial event differs from record')
        replay = Path(row['replay'])
        require(replay.resolve().is_relative_to(directory.resolve()) and replay.is_file(), 'Missing run-local replay')
        native_replay_header(replay)
    old = {row['episode']: row for row in read(evidence)[group]}
    paired, unpaired, gained, lost = [], [], [], []
    for i, row in by_id.items():
        if i not in old:
            unpaired.append(169+i)
            continue
        require(all(row[k] == old[i][k] for k in ('scenario_sha256', 'initial_state_sha256')),
                'Paired episode mismatch')
        paired.append(i)
        if row['outcome'] == 'win' and old[i]['outcome'] != 'win': gained.append(169+i)
        if row['outcome'] != 'win' and old[i]['outcome'] == 'win': lost.append(169+i)
    counts = {plural: sum(row['outcome'] == name for row in rows) for name, plural in
              [('win', 'wins'), ('loss', 'losses'), ('draw', 'draws'), ('timeout', 'timeouts')]}
    new_wins = sum(by_id[i]['outcome'] == 'win' for i in paired)
    old_wins = sum(old[i]['outcome'] == 'win' for i in paired)
    complete = bool(summary and summary.get('status') == 'complete' and len(rows) == len(scenarios))
    return dict(group=group, status='complete' if complete else 'partial', completed_games=len(rows),
                **counts, win_rate=counts['wins']/len(rows) if rows else None,
                paired_games=len(paired), paired_current_wins=new_wins, baseline_wins=old_wins,
                win_delta=new_wins-old_wins, gained_seeds=gained, lost_seeds=lost,
                unpaired_seeds=unpaired, excluded_incomplete_seeds=[169+i for i in sorted(set(starts)-set(by_id))],
                completed_seeds=[169+i for i in sorted(by_id)], runner_error=summary.get('error') if summary else None)

try:
    mode, *args = sys.argv[1:]
    if mode == 'manifest':
        source, count, destination = args
        value = manifest(source, int(count))
        data = json.dumps(value, ensure_ascii=False, indent=2).encode('utf-8')
        if destination != '-':
            with Path(destination).open('xb') as stream: stream.write(data)
        result = dict(sha256=value['sha256'], file_sha256=hashlib.sha256(data).hexdigest())
    elif mode == 'baseline':
        directory, source, count, freeze = args
        result = baseline(Path(directory), source, int(count), freeze)
    elif mode == 'audit':
        directory, selected, freeze, evidence, group = args
        result = audit(Path(directory), read(selected), freeze, evidence, group)
    else:
        raise ValueError('Unknown audit operation')
    print(json.dumps(dict(ok=True, result=result)))
except Exception as exc:
    print(json.dumps(dict(ok=False, error=type(exc).__name__ + ': ' + str(exc))))
'@
function Invoke-Audit { param([string[]]$Arguments)
    $json = $auditSource | & $python -B - @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'Offline evidence helper failed.' }
    $answer = $json | ConvertFrom-Json
    if (-not $answer.ok) { throw $answer.error }
    return $answer.result
}
if (-not (Test-Path -LiteralPath $python)) { throw "Python environment missing: $python" }
if ($Resume) {
    if (-not (Test-Path -LiteralPath $outputRoot -PathType Container)) { throw "Resume output directory does not exist: $outputRoot" }
} elseif (Test-Path -LiteralPath $outputRoot) {
    throw "Output already exists; preserve it and choose a new OutputName: $outputRoot"
}
$design = Read-Json (Join-Path $prior 'design.json')
$priorFreeze = Read-Json (Join-Path $prior 'source_freeze.json')
$comparisonFreeze = Read-Json (Join-Path $comparisonRoot 'source_freeze.json')
$manifestHashes = @{}
$baseline = @{}
$baselineAudit = @{}
$nativeFiles = @{}
foreach ($group in $groups) {
    $manifestPath = Join-Path $prior ('manifests\' + $group + '.json')
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $expected = $design.groups.$group
    $hash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $expected.manifest_file_sha256) { throw "Paired manifest drift: $group" }
    if ($manifest.episodes.Count -ne 20 -or $manifest.seed -ne 170 -or
        $manifest.env_args.step_mul -ne 8 -or [string]$manifest.env_args.difficulty -ne '7' -or
        $manifest.env_args.unit_balance -ne 'native' -or
        $manifest.env_args.capability_config.n_units -ne 10 -or
        $manifest.env_args.capability_config.n_enemies -ne 10) { throw "Protocol mismatch: $group" }
    for ($index = 0; $index -lt 20; $index++) {
        if ($manifest.episodes[$index].index -ne $index -or
            $manifest.episodes[$index].sc2_seed -ne (170 + $index)) { throw "Seed mismatch: $group" }
    }
    if ((Get-FileHash -LiteralPath $expected.map -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected.map_sha256) {
        throw "Native map changed: $group"
    }
    $manifestHashes[$group] = Invoke-Audit @('manifest', $manifestPath, [string]$GamesPerGroup, '-')
    $mapRoot = Split-Path -Parent (Split-Path -Parent $expected.map)
    $nativeRoot = Split-Path -Parent $mapRoot
    foreach ($name in @('starcraft2.py', 'wrapper.py', 'distributions.py')) {
        $key = 'dependencies/smacv2/' + $name
        $path = Join-Path $nativeRoot $name
        if ($nativeFiles.ContainsKey($key) -and $nativeFiles[$key] -ne $path) { throw 'Mixed native dependency roots.' }
        $nativeFiles[$key] = $path
    }
    $nativeFiles['dependencies/smacv2/maps/SMAC_Maps/' + (Split-Path -Leaf $expected.map)] = $expected.map
    $baselineDir = Join-Path $comparisonRoot ('current\' + $group)
    $baselineAudit[$group] = Invoke-Audit @('baseline', $baselineDir, $manifestPath,
        [string]$GamesPerGroup, (Join-Path $comparisonRoot 'source_freeze.json'))
    $baseline[$group] = @($baselineAudit[$group].rows)
}
foreach ($key in $nativeFiles.Keys) {
    $property = $priorFreeze.source_hashes.PSObject.Properties[$key]
    $comparisonProperty = $comparisonFreeze.source_hashes.PSObject.Properties[$key]
    if ($null -eq $property -or $null -eq $comparisonProperty -or
            $property.Value -ne $comparisonProperty.Value -or (Get-Sha $nativeFiles[$key]) -ne $property.Value) {
        throw "Native dependency differs from paired baseline: $key"
    }
}

function Get-RuntimeFiles {
    $files = @{}
    foreach ($folder in @('skills\tactical_v1', 'skills\current', 'sc2_skill_adapter\bridge', 'sc2_skill_adapter\display_guard')) {
        foreach ($file in Get-ChildItem -LiteralPath (Join-Path $repoRoot $folder) -Recurse -File) {
            if ($file.Extension -notin @('.py', '.json') -or $file.FullName -match '[\\/](tests|__pycache__)[\\/]') { continue }
            $relative = $file.FullName.Substring($repoRoot.Length).TrimStart([char[]]'\/').Replace('\', '/')
            $files[$relative] = $file.FullName
        }
    }
    $files['sc2_skill_adapter/run.py'] = Join-Path $adapterRoot 'run.py'
    $files['sc2_skill_adapter/scripts/run_tactical_matrix.ps1'] = $launcherPath
    foreach ($key in $nativeFiles.Keys) { $files[$key] = $nativeFiles[$key] }
    return $files
}

function Get-RuntimeHashes {
    $hashes = @{}
    $files = Get-RuntimeFiles
    foreach ($key in $files.Keys) { $hashes[$key] = Get-Sha $files[$key] }
    return $hashes
}

function Assert-Unchanged {
    param($ExpectedHashes)
    $current = Get-RuntimeHashes
    if ($current.Count -ne $ExpectedHashes.Count) { throw 'Runtime files changed during the batch.' }
    foreach ($path in $ExpectedHashes.Keys) {
        if ($current[$path] -ne $ExpectedHashes[$path]) { throw "Runtime source changed during the batch: $path" }
    }
}

function Get-SnapshotPath { param([string]$Relative)
    $root = [System.IO.Path]::GetFullPath($outputRoot).TrimEnd([char[]]'\/') + [System.IO.Path]::DirectorySeparatorChar
    $destination = [System.IO.Path]::GetFullPath((Join-Path (Join-Path $outputRoot 'source_snapshot') $Relative))
    if (-not $destination.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { throw 'Snapshot destination escaped output directory.' }
    return $destination
}

$frozen = Get-RuntimeHashes
$sources = Get-RuntimeFiles
foreach ($key in $sources.Keys) { $null = Get-SnapshotPath $key }
$totalGames = $groups.Count * $GamesPerGroup
$lastSeed = 169 + $GamesPerGroup
$availablePairs = @($baseline.Values | ForEach-Object { $_ }).Count
$resumeResults = @()
$resumeSkip = @{}
$resumeEvidence = @{}
$resumeRecord = $null
$statusPath = Join-Path $outputRoot 'matrix_summary.json'
if ($Resume) {
    # The original freeze is immutable. Only this launcher can have changed;
    # metadata for old/new games continues to use the identical runtime freeze.
    $originalFreeze = Read-Json (Join-Path $outputRoot 'source_freeze.json')
    $originalHashes = @{}
    foreach ($property in $originalFreeze.source_hashes.PSObject.Properties) {
        $originalHashes[$property.Name] = [string]$property.Value
    }
    if ($originalHashes.Count -ne $frozen.Count) { throw 'Resume runtime file set changed.' }
    $launcherKey = 'sc2_skill_adapter/scripts/run_tactical_matrix.ps1'
    foreach ($key in $frozen.Keys) {
        if (-not $originalHashes.ContainsKey($key) -or
            ($key -ne $launcherKey -and $frozen[$key] -ne $originalHashes[$key])) {
            throw "Resume runtime/config/dependency drift: $key"
        }
        $snapshotPath = Get-SnapshotPath $key
        if (-not (Test-Path -LiteralPath $snapshotPath -PathType Leaf) -or
            (Get-Sha $snapshotPath) -ne $originalHashes[$key]) {
            throw "Resume original snapshot changed: $key"
        }
        $sourceProperty = $originalFreeze.source_paths.PSObject.Properties[$key]
        if ($null -eq $sourceProperty -or $sourceProperty.Value -ne $sources[$key]) {
            throw "Resume runtime source path changed: $key"
        }
    }
    $originalBatch = Read-Json $statusPath
    # Older batches did not record a selection and always scheduled all nine groups.
    $originalGroups = $allGroups
    if ($null -ne $originalBatch.PSObject.Properties['groups']) { $originalGroups = @($originalBatch.groups) }
    if (($originalGroups -join ',') -ne ($groups -join ',')) {
        throw 'Resume race/group selection changed; Races must match the original batch.'
    }
    if ($originalBatch.games_per_group -ne $GamesPerGroup -or $originalBatch.total_games -ne $totalGames -or
        $originalBatch.seed_start -ne 170 -or $originalBatch.seed_end -ne $lastSeed -or
        $originalBatch.step_mul -ne 8 -or [string]$originalBatch.difficulty -ne '7' -or
        $originalBatch.skill_version -ne 'tactical_v1' -or $originalBatch.baseline -ne $comparisonRoot -or
        $originalBatch.manifest_source -ne $prior -or $originalBatch.available_baseline_pairs -ne $availablePairs) {
        throw 'Resume protocol or baseline changed; GamesPerGroup must match the original batch.'
    }
    $savedManifests = Read-Json (Join-Path $outputRoot 'manifest_audit.json')
    $savedBaselineAudit = Read-Json (Join-Path $outputRoot 'baseline_audit.json')
    $savedBaseline = Read-Json (Join-Path $outputRoot 'baseline_episodes.json')
    if ((Get-Sha (Join-Path $outputRoot 'baseline_source_freeze.json')) -ne
        (Get-Sha (Join-Path $comparisonRoot 'source_freeze.json'))) {
        throw 'Resume baseline source freeze changed.'
    }
    foreach ($group in $groups) {
        if ($savedManifests.$group.sha256 -ne $manifestHashes[$group].sha256 -or
            $savedManifests.$group.file_sha256 -ne $manifestHashes[$group].file_sha256) {
            throw "Resume signed manifest changed: $group"
        }
        foreach ($name in @('episodes.json', 'summary.json', 'run.json')) {
            if ($savedBaselineAudit.$group.evidence_hashes.$name -ne $baselineAudit[$group].evidence_hashes.$name) {
                throw "Resume baseline evidence changed: $group/$name"
            }
        }
        $savedRows = ConvertTo-Json -InputObject @($savedBaseline.$group) -Depth 30 -Compress
        $auditedRows = ConvertTo-Json -InputObject @($baseline[$group]) -Depth 30 -Compress
        if ($savedRows -ne $auditedRows) { throw "Resume selected baseline rows changed: $group" }
    }
    $currentRoot = Join-Path $outputRoot 'current'
    if (Test-Path -LiteralPath $currentRoot) {
        foreach ($entry in Get-ChildItem -LiteralPath $currentRoot) {
            if (-not $entry.PSIsContainer -or $entry.Name -notin $groups) {
                throw "Resume unexpected current output: $($entry.Name)"
            }
        }
    }
    $prefixEnded = $false
    foreach ($group in $groups) {
        $directory = Join-Path $outputRoot ('current\' + $group)
        $selected = Join-Path $outputRoot ('manifests\' + $group + '.json')
        $log = Join-Path $outputRoot ('logs\' + $group + '.log')
        $errorLog = Join-Path $outputRoot ('logs\' + $group + '.stderr.log')
        if (-not (Test-Path -LiteralPath $directory)) {
            $prefixEnded = $true
            if ((Test-Path -LiteralPath $selected) -or (Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $errorLog)) {
                throw "Resume refuses partial/unstarted group artifacts: $group"
            }
            continue
        }
        if ($prefixEnded) { throw "Resume requires a contiguous complete prefix; unexpected group: $group" }
        if (-not (Test-Path -LiteralPath $selected -PathType Leaf) -or
            (Get-Sha $selected) -ne $manifestHashes[$group].file_sha256) {
            throw "Resume subset manifest mismatch: $group"
        }
        $result = Invoke-Audit @('audit', $directory, $selected,
            (Join-Path $outputRoot 'source_freeze.json'), (Join-Path $outputRoot 'baseline_episodes.json'), $group)
        if ($result.status -ne 'complete' -or $result.completed_games -ne $GamesPerGroup -or
            @($result.excluded_incomplete_seeds).Count -ne 0 -or $result.runner_error) {
            throw "Resume refuses partial group: $group"
        }
        $resumeResults += $result
        $resumeSkip[$group] = $true
        $evidence = @{}
        foreach ($file in Get-ChildItem -LiteralPath $directory -Recurse -File) {
            $relative = $file.FullName.Substring($outputRoot.Length).TrimStart([char[]]'\/').Replace('\', '/')
            $evidence[$relative] = Get-Sha $file.FullName
        }
        $evidence['manifests/' + $group + '.json'] = Get-Sha $selected
        if (Test-Path -LiteralPath $log) { $evidence['logs/' + $group + '.log'] = Get-Sha $log }
        if (Test-Path -LiteralPath $errorLog) { $evidence['logs/' + $group + '.stderr.log'] = Get-Sha $errorLog }
        $resumeEvidence[$group] = $evidence
    }
    if (-not $resumeResults.Count) { throw 'Resume requires at least one audited complete prefix group; partial attempts are never overwritten.' }
    $resumeRecord = [ordered]@{
        mode = 'complete_prefix_only'; checked_utc = [DateTime]::UtcNow.ToString('o');
        original_source_freeze_sha256 = Get-Sha (Join-Path $outputRoot 'source_freeze.json');
        original_matrix_summary_sha256 = Get-Sha $statusPath;
        original_launcher_sha256 = $originalHashes[$launcherKey]; launcher_sha256 = $frozen[$launcherKey];
        allowed_changed_source = $launcherKey; runtime_source_hashes = $frozen;
        retained_groups = @($resumeResults | ForEach-Object { $_.group });
        retained_games = $resumeResults.Count * $GamesPerGroup;
        remaining_groups = @($groups | Where-Object { -not $resumeSkip.ContainsKey($_) });
        validated_results = $resumeResults; retained_evidence_hashes = $resumeEvidence
    }
    Write-Host ("Resume validated {0} complete prefix groups ({1} games); skipping them and scheduling {2} games." -f
        $resumeResults.Count, $resumeRecord.retained_games, ($totalGames - $resumeRecord.retained_games))
}
Write-Host "Native 10v10: $($groups.Count) mirror compositions ($($groups -join ', ')) x $GamesPerGroup games = $totalGames games."
Write-Host "Seeds 170..$lastSeed; difficulty 7; step_mul 8; balanced; skill_max_steps 4."
Write-Host "Primary comparison: $comparisonRoot ($availablePairs available pairs; missing seeds remain unpaired)."
Write-Host "Output: $outputRoot"
if ($DryRun) {
    foreach ($group in $groups) {
        Write-Host ("{0}: {1}" -f $group, (Join-Path $prior ('manifests\' + $group + '.json')))
    }
    Write-Host "Dry run validated $($groups.Count) signed subset manifests, $availablePairs baseline rows, native dependencies, maps and $($frozen.Count) snapshot paths. No game was started."
    return
}

if (-not $Resume) {
New-Item -ItemType Directory -Path $outputRoot -ErrorAction Stop | Out-Null
New-Item -ItemType Directory -Path (Join-Path $outputRoot 'manifests') | Out-Null
New-Item -ItemType Directory -Path (Join-Path $outputRoot 'logs') | Out-Null
Write-Json ([ordered]@{ source_hashes = $frozen; source_paths = $sources }) (Join-Path $outputRoot 'source_freeze.json')
foreach ($key in $sources.Keys) {
    $destination = Get-SnapshotPath $key
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath $sources[$key] -Destination $destination
    if ((Get-Sha $destination) -ne $frozen[$key]) { throw "Snapshot copy mismatch: $key" }
}
Write-Json $baseline (Join-Path $outputRoot 'baseline_episodes.json')
Write-Json $baselineAudit (Join-Path $outputRoot 'baseline_audit.json')
Write-Json $manifestHashes (Join-Path $outputRoot 'manifest_audit.json')
Copy-Item -LiteralPath (Join-Path $comparisonRoot 'source_freeze.json') -Destination (Join-Path $outputRoot 'baseline_source_freeze.json')
} else {
    Assert-Unchanged $frozen
    $resumeId = (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
    $resumeDirectory = Join-Path $outputRoot ('resume_audit\' + $resumeId)
    New-Item -ItemType Directory -Path $resumeDirectory -ErrorAction Stop | Out-Null
    Copy-Item -LiteralPath $statusPath -Destination (Join-Path $resumeDirectory 'matrix_summary_before.json')
    Copy-Item -LiteralPath $launcherPath -Destination (Join-Path $resumeDirectory 'run_tactical_matrix.ps1')
    if ((Get-Sha (Join-Path $resumeDirectory 'run_tactical_matrix.ps1')) -ne $frozen[$launcherKey]) {
        throw 'Resume launcher evidence copy mismatch.'
    }
    $resumeRecord['launcher_source'] = 'run_tactical_matrix.ps1'
    Write-Json $resumeRecord (Join-Path $resumeDirectory 'audit.json')
}
$batch = [ordered]@{ status = 'running'; baseline = $comparisonRoot; manifest_source = $prior;
    races = $selectedRaces; groups = $groups;
    total_games = $totalGames; games_per_group = $GamesPerGroup; available_baseline_pairs = $availablePairs;
    seed_start = 170; seed_end = $lastSeed; step_mul = 8; difficulty = '7';
    skill_version = 'tactical_v1'; active_group = $null; completed_games = $resumeResults.Count * $GamesPerGroup; results = @($resumeResults) }
if ($Resume) { $batch['resume_audit'] = 'resume_audit/' + $resumeId + '/audit.json' }
$activeAudited = $true
$runDirectory = $null
$manifestPath = $null
function Write-BatchProgress { param([string]$Message)
    # No console output during the batch: a selected legacy console must not
    # block the next group after Python has already completed successfully.
    $line = '{0} {1}' -f ([DateTime]::Now.ToString('o')), $Message
    Add-Content -LiteralPath (Join-Path $outputRoot 'matrix_activity.log') -Value $line -Encoding UTF8
}
function ConvertTo-NativeArgument { param([string]$Value)
    # Start-Process joins ArgumentList into a Win32 command line. Preserve
    # spaces, quotes and trailing backslashes using CRT argument quoting.
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}
function Save-ActiveResult {
    Write-BatchProgress "Auditing $($batch.active_group)."
    Assert-Unchanged $frozen
    $result = Invoke-Audit @('audit', $runDirectory, $manifestPath,
        (Join-Path $outputRoot 'source_freeze.json'), (Join-Path $outputRoot 'baseline_episodes.json'), $batch.active_group)
    $batch.results += $result
    $batch.completed_games += $result.completed_games
    Write-Json $batch $statusPath
    return $result
}
Write-Json $batch $statusPath
Write-Host 'Batch progress is written to matrix_activity.log and matrix_summary.json; episode output is in logs/<group>.log (errors: <group>.stderr.log).'
Write-Host 'The running batch does not stream to this terminal. View progress from another terminal if needed.'
Push-Location $repoRoot
try {
    foreach ($group in $groups) {
        Assert-Unchanged $frozen
        if ($resumeSkip.ContainsKey($group)) {
            Write-BatchProgress "Skipping audited complete group: $group"
            continue
        }
        $batch.active_group = $group
        Write-Json $batch $statusPath
        $manifestPath = Join-Path $outputRoot ('manifests\' + $group + '.json')
        $subset = Invoke-Audit @('manifest', (Join-Path $prior ('manifests\' + $group + '.json')), [string]$GamesPerGroup, $manifestPath)
        if ($subset.sha256 -ne $manifestHashes[$group].sha256 -or
            (Get-Sha $manifestPath) -ne $manifestHashes[$group].file_sha256) {
            throw "Subset manifest mismatch: $group"
        }
        $runDirectory = Join-Path $outputRoot ('current\' + $group)
        if (Test-Path -LiteralPath $runDirectory) { throw "Refusing to overwrite group output: $group" }
        $activeAudited = $false
        $runArgs = @('-u', '-B', (Join-Path $adapterRoot 'run.py'), 'run',
            '--method', 'adaptive-skill', '--skill-version', 'tactical_v1',
            '--manifest', $manifestPath, '--output-dir', $runDirectory,
            '--adaptive-posture', 'balanced', '--skill-max-steps', '4')
        Write-BatchProgress "Starting $group ($GamesPerGroup games)"
        $runExitCode = $null
        $child = $null
        try {
            # Both native streams go to files, without a terminal/Tee pipeline.
            # Keep a synchronous process wait so source checks and evidence
            # audit run only after the game process has actually exited.
            $nativeArguments = ($runArgs | ForEach-Object { ConvertTo-NativeArgument $_ }) -join ' '
            $child = Start-Process -FilePath $python -ArgumentList $nativeArguments `
                -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru `
                -RedirectStandardOutput (Join-Path $outputRoot ('logs\' + $group + '.log')) `
                -RedirectStandardError (Join-Path $outputRoot ('logs\' + $group + '.stderr.log'))
            $null = $child.Handle
            while (-not $child.WaitForExit(1000)) { }
            $child.WaitForExit() # Drain the redirected asynchronous streams.
            $runExitCode = $child.ExitCode
        } finally {
            if ($null -ne $child) {
                try {
                    if (-not $child.HasExited) {
                        # Cancelling the launcher must not leave its detached
                        # Python/SC2 children running. Never kill by image name.
                        $stopper = Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\taskkill.exe') `
                            -ArgumentList @('/PID', [string]$child.Id, '/T', '/F') `
                            -WindowStyle Hidden -PassThru -Wait `
                            -RedirectStandardOutput (Join-Path $outputRoot ('logs\' + $group + '.cancel.log')) `
                            -RedirectStandardError (Join-Path $outputRoot ('logs\' + $group + '.cancel.stderr.log'))
                        $stopper.Dispose()
                    }
                } finally { $child.Dispose() }
            }
        }
        Write-BatchProgress "Python finished $group with exit code $runExitCode."
        $result = Save-ActiveResult
        $activeAudited = $true
        if ($null -eq $runExitCode -or $runExitCode -ne 0) { throw "$group failed with exit $runExitCode; validated partial episodes preserved in matrix_summary." }
        if ($result.status -ne 'complete') { throw "Incomplete group: $group" }
        Write-BatchProgress "$group finished: $($result.wins) / $GamesPerGroup wins; $($result.paired_games) exact pairs."
    }
    $batch.status = 'complete'
    $batch.active_group = $null
    Write-BatchProgress "All $totalGames games complete."
} catch {
    $batch.status = 'failed'
    $batch['error'] = $_.Exception.Message
    throw
} finally {
    if (-not $activeAudited) {
        try { $null = Save-ActiveResult } catch { $batch['partial_audit_error'] = $_.Exception.Message }
        $batch.status = 'failed'
    }
    Write-Json $batch $statusPath
    Pop-Location
}
Write-Host "All $totalGames games complete. Summary: $statusPath"
if ('zerg' -in $selectedRaces) {
    & $python -B (Join-Path $PSScriptRoot 'summarize_zerg_damage.py') $outputRoot
    if ($LASTEXITCODE -ne 0) {
        Write-Warning 'Games are complete. Zerg damage analysis failed; rerun summarize_zerg_damage.py on this batch.'
    }
}
