"""Build a paired Zerg contact A/B report from completed runs (no SC2)."""

import hashlib
import json
import math
import re
from pathlib import Path


VERSIONS = ('baseline', 'target', 'formation')
TYPES = ('Zergling', 'Baneling', 'Hydralisk', 'other')
OUTCOMES = ('win', 'loss', 'draw', 'timeout')


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _digest(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(data).hexdigest()


def _games(folder, size, start, hashes, manifest_hash=None):
    records = _read(folder / 'episodes.json')
    if not isinstance(records, list) or len(records) != size:
        raise ValueError(f'{folder}: expected {size} episode records')
    games = []
    for i, scenario in enumerate(hashes, start):
        row = records[i]
        state = row['initial_state_sha256']
        if (row['episode'] != i + 1 or row['scenario_sha256'] != scenario
                or not re.fullmatch('[0-9a-f]{64}', state) or row['outcome'] not in OUTCOMES):
            raise ValueError(f'{folder}: invalid episode {i + 1}')
        path = folder / 'damage_by_type' / f'episode-{i + 1:04d}.json'
        entry = _read(path)
        if (entry['episode'] != i + 1 or entry['scenario_sha256'] != scenario
                or (manifest_hash is not None and entry['manifest_sha256'] != manifest_hash)):
            raise ValueError(f'{path}: episode/manifest mismatch')
        damage = {}
        for player, side in [('1', 'ally'), ('2', 'enemy')]:
            values = entry['players'][player]['damage_dealt_by_source_type']
            if set(values) != set(TYPES) or not all(
                    type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values.values()):
                raise ValueError(f'{path}: invalid {side} Zerg damage')
            damage[side] = values
        games.append(dict(initial_state_sha256=state, outcome=row['outcome'], damage_by_type=damage))
    return games


def _summary():
    return dict(versions={v: dict(games=0, outcomes=dict.fromkeys(OUTCOMES, 0),
                                  damage_by_type={s: dict.fromkeys(TYPES, 0) for s in ('ally', 'enemy')})
                          for v in VERSIONS},
                paired_vs_baseline={v: dict(gained_wins=0, dropped_wins=0) for v in VERSIONS[1:]})


def _count(summary, games):
    baseline_won = games['baseline']['outcome'] == 'win'
    for version, game in games.items():
        stats = summary['versions'][version]
        stats['games'] += 1
        stats['outcomes'][game['outcome']] += 1
        for side, values in game['damage_by_type'].items():
            for unit, damage in values.items():
                stats['damage_by_type'][side][unit] += damage
        won = game['outcome'] == 'win'
        if version != 'baseline' and won != baseline_won:
            summary['paired_vs_baseline'][version]['gained_wins' if won else 'dropped_wins'] += 1


def summarize(batch: Path, previous: Path, jobs: list):
    """Validate complete 10-game groups, write comparison.json and return its content.

    Each old pair needs target and formation jobs; its baseline is historical.
    New pairs need all three jobs. Gains/drops compare win vs any non-win.
    """
    parts, offsets, names = {}, {}, {}
    for label, name, path in jobs:
        path = Path(path)
        match = re.fullmatch(r'(old|new)__(.+)__([02468])', path.stem)
        if not match or label not in VERSIONS:
            raise ValueError(f'Invalid job: {label}/{path}')
        kind, source, offset = match.groups()
        if source in ('.', '..') or not name or names.setdefault((kind, source), name) != name:
            raise ValueError(f'Invalid/inconsistent group for {path}')
        key = kind, source, int(offset)
        offsets.setdefault(key[:2], set()).add(key[2])
        labels = parts.setdefault(key, {})
        if label in labels or (labels and path != next(iter(labels.values()))):
            raise ValueError(f'Duplicate job or conflicting manifest: {label}/{path}')
        labels[label] = path
    if {kind for kind, _ in offsets} != {'old', 'new'}:
        raise ValueError('Expected both old and new scenario groups')
    if any(group != {0, 2, 4, 6, 8} for group in offsets.values()):
        raise ValueError('Expected all five two-game offsets in every supplied group')

    report = {scope: _summary() for scope in ('old', 'new', 'total')}
    report['episodes'] = []
    for (kind, source, offset), labels in sorted(parts.items()):
        expected = VERSIONS[1:] if kind == 'old' else VERSIONS
        if set(labels) != set(expected):
            raise ValueError(f'{kind}/{source}/{offset}: expected versions {expected}, got {list(labels)}')
        path = labels['target']
        manifest = _read(path)
        episodes = manifest['episodes']
        hashes = [row['sha256'] for row in episodes]
        if (len(hashes) != 2 or hashes != [_digest(row['episode_config']) for row in episodes]
                or manifest['sha256'] != _digest({k: v for k, v in manifest.items() if k != 'sha256'})):
            raise ValueError(f'{path}: incomplete or corrupt manifest')
        runs = {label: _games(batch / label / path.stem, 2, 0, hashes, manifest['sha256'])
                for label in expected}
        if kind == 'old':
            runs['baseline'] = _games(previous / 'contact_response' / source, 10, offset, hashes)
        for i, scenario in enumerate(hashes):
            games = {v: runs[v][i] for v in VERSIONS}
            if len({g['initial_state_sha256'] for g in games.values()}) != 1:
                raise ValueError(f'{path}: initial state mismatch at episode {i + 1}')
            report['episodes'].append(dict(scope=kind, name=names[kind, source], source=source,
                                           episode=offset + i + 1, scenario_sha256=scenario, versions=games))
            _count(report[kind], games)
            _count(report['total'], games)
    (batch / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report
