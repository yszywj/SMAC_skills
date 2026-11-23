"""Validate and summarize a completed micro8-off/hydra batch without SC2."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from compare_protoss_parallel import digest, validate_manifest, write_json
from contact_ab_report import OUTCOMES, TYPES, _games

MICRO_LABELS = ('micro8-off', 'micro8-hydra')


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _shard(work, manifest, label, files):
    run, summary = (_read(work / filename) for filename in ('run.json', 'summary.json'))
    records = _read(work / 'episodes.json')
    hashes = [row['sha256'] for row in manifest['episodes']]
    expected = dict(method='adaptive-skill', skill_version='tactical_v1', execution_mode=label,
                    manifest_sha256=manifest['sha256'])
    if (any(meta[k] != v for meta in (run, summary) for k, v in expected.items())
            or (run['macro_loops'], run['engine_step_loops']) != (8, 1)
            or run['damage_by_type_enabled'] is not True or run['env_args'] != manifest['env_args']
            or summary['status'] != 'complete' or summary['completed_episodes'] != len(hashes)
            or _read(work / 'manifest.json') != manifest):
        raise ValueError(f'{work}: incorrect run metadata or incomplete run')
    source = {Path(k).as_posix(): v for k, v in run['source_hashes'].items()}
    skills = {k: v for k, v in files.items() if k.startswith('skills/tactical_v1/') and k.endswith('.py')}
    if (not skills or skills != run['skill_source_hashes'] or not skills.items() <= source.items()
            or not source.items() <= files.items()
            or 'sc2_skill_adapter/bridge/micro_trigger.py' not in source
            or 'sc2_skill_adapter/damage_maps/10gen_zerg.SC2Map' not in source):
        raise ValueError(f'{work}: sources differ from snapshot')
    games = _games(work, len(hashes), 0, hashes, manifest['sha256'])
    for i, (game, record) in enumerate(zip(games, records), 1):
        block = record['micro_trigger']
        audit = _read(work / 'micro_triggers' / f'episode-{i:04d}.json')
        rows = audit['authorizations']
        states = Counter(row['status'] for row in rows)
        if (audit['episode'] != i or any(meta['mode'] != label or
                (meta['macro_loops'], meta['engine_step_loops']) != (8, 1) for meta in (audit, block))
                or block['windows'] != record['steps'] or record['steps'] <= 0
                or block['authorizations'] != len(rows) or block['states'] != dict(states)
                or any(not row['status'] or row['episode'] != i or
                       (row['submitted_loop'] is not None) != (row['status'] == 'submitted') for row in rows)
                or (label == 'micro8-off' and states['submitted'])
                or (label == 'micro8-hydra' and states['legal_control_off'])):
            raise ValueError(f'{work}: invalid micro trigger accounting for episode {i}')
        game['micro_trigger'] = dict(windows=block['windows'], authorizations=len(rows),
                                     states=dict(states), submitted=states['submitted'])
    return games, (source, run['skill_parameters'], run['versions'])


def _stats(pairs):
    versions = {}
    for label in MICRO_LABELS:
        games = [pair['versions'][label] for pair in pairs]
        counts = Counter(g['outcome'] for g in games)
        states = Counter()
        for game in games:
            states.update(game['micro_trigger']['states'])
        damage = {side: {unit: sum(g['damage_by_type'][side][unit] for g in games) for unit in TYPES}
                  for side in ('ally', 'enemy')}
        micro = {k: sum(g['micro_trigger'][k] for g in games) for k in ('windows', 'authorizations', 'submitted')}
        versions[label] = dict(games=len(games), outcomes={k: counts[k] for k in OUTCOMES},
                               win_rate=counts['win'] / len(games), damage_by_type=damage,
                               micro_trigger=dict(micro, states=dict(states)))
    off, hydra = MICRO_LABELS
    transitions = Counter((p['versions'][off]['outcome'], p['versions'][hydra]['outcome']) for p in pairs)
    return dict(pairs=len(pairs), versions=versions,
                gained_wins=sum(n for (a, b), n in transitions.items() if a != 'win' and b == 'win'),
                dropped_wins=sum(n for (a, b), n in transitions.items() if a == 'win' and b != 'win'),
                paired_outcomes={a: {b: transitions[a, b] for b in OUTCOMES} for a in OUTCOMES},
                damage_delta_hydra_minus_off={s: {u: versions[hydra]['damage_by_type'][s][u] -
                    versions[off]['damage_by_type'][s][u] for u in TYPES} for s in ('ally', 'enemy')})


def micro_report(batch):
    design = _read(batch / 'design.json')
    files, shards = design['snapshot_files'], design['shards']
    if digest(files) != design['snapshot_sha256'] or tuple(design['labels']) != MICRO_LABELS:
        raise ValueError('Invalid batch design')
    # Runner hashes tactical .py files but not defaults.json or shared skills/current.
    for folder in ('source_snapshot', *MICRO_LABELS):
        for name, checksum in files.items():
            path = batch / folder / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError(f'{path}: snapshot changed')
    groups = [f'zerg_{n}zergling_0baneling_{10-n}hydralisk' for n in (3, 6)]
    if set(shards) != {f'{name}_{offset:02d}' for name in groups for offset in range(0, 75, 5)}:
        raise ValueError('Expected 15 shared five-game shards per composition')
    pairs, contract = [], None
    for shard, plan in sorted(shards.items()):
        name, offset = shard.rsplit('_', 1)
        manifest = _read(batch / 'manifests' / f'{shard}.json')
        validate_manifest(manifest)
        start = design['seed'] + 75 * groups.index(name) + int(offset)
        if (plan['composition'] != name or manifest['sha256'] != plan['manifest_sha256']
                or [r['sc2_seed'] for r in manifest['episodes']] != list(range(start, start + 5))):
            raise ValueError(f'{shard}: unplanned scenarios')
        runs = {}
        for label in MICRO_LABELS:
            runs[label], current = _shard(batch / label / shard, manifest, label, files)
            if contract is not None and contract != current:
                raise ValueError(f'{label}/{shard}: runner sources/parameters/versions changed')
            contract = current
        for i, row in enumerate(manifest['episodes']):
            games = {label: runs[label][i] for label in MICRO_LABELS}
            states = {g['initial_state_sha256'] for g in games.values()}
            if len(states) != 1:
                raise ValueError(f'{shard} episode {i+1}: initial state mismatch')
            pairs.append(dict(composition=name, shard=shard, episode=i+1, sc2_seed=row['sc2_seed'],
                              scenario_sha256=row['sha256'], initial_state_sha256=states.pop(), versions=games))
    report = dict(snapshot_sha256=design['snapshot_sha256'], total=_stats(pairs), by_composition={
        name: _stats([p for p in pairs if p['composition'] == name]) for name in groups}, episodes=pairs)
    write_json(batch / 'comparison.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    micro_report(parser.parse_args().batch)
