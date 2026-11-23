"""Offline paired report for four upper-level policies using micro8-hydra."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path

from compare_protoss_parallel import digest, validate_manifest, write_json
from contact_early_report import COMPOSITIONS, SIDES
from contact_front_report import _layout
from contact_micro_report import OUTCOMES, TYPES, _read, _shard

LABELS = ('baseline', 'simple_hydra', 'simple_ling', 'simple_both')
MODE = 'micro8-hydra'


def _summary(rows):
    versions, paired = {}, {}
    for label in LABELS:
        games = [r['versions'][label] for r in rows]
        outcomes = Counter(g['outcome'] for g in games)
        triggers = [g['micro_trigger'] for g in games]
        states = Counter()
        for t in triggers:
            states.update(t['states'])
        versions[label] = dict(
            games=len(games), outcomes={o: outcomes[o] for o in OUTCOMES},
            win_rate=outcomes['win'] / len(games),
            damage_by_type={s: {u: sum(g['damage_by_type'][s][u] for g in games) for u in TYPES} for s in SIDES},
            micro_trigger=dict(states=dict(states), **{k: sum(t[k] for t in triggers)
                               for k in ('windows', 'authorizations', 'submitted')}))
        if label != 'baseline':
            transitions = Counter((r['versions']['baseline']['outcome'] == 'win', g['outcome'] == 'win')
                                  for r, g in zip(rows, games))
            paired[label] = dict(gained_wins=transitions[False, True], dropped_wins=transitions[True, False])
    return dict(pairs=len(rows), versions=versions, paired_vs_baseline=paired)


def _by_layout(rows):
    return {layout: _summary([r for r in rows if r['layout'] == layout])
            for layout in sorted({r['layout'] for r in rows})}


def summarize(batch: Path):
    """Validate all 150 four-way matches before writing and returning comparison.json."""
    design = _read(batch / 'design.json')
    files, shards = design['policy_files'], design['shards']
    if tuple(design['labels']) != LABELS or set(files) != set(LABELS) or type(design['seed']) is not int:
        raise ValueError('Invalid factorial batch design')
    if set(shards) != {f'{name}_{i:02d}' for name in COMPOSITIONS for i in range(0, 75, 5)}:
        raise ValueError('Expected 15 five-game shards per composition')
    shared_files = None
    for label in LABELS:
        root = batch / label
        for name, checksum in files[label].items():
            path = root / name
            if Path(name).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f'{path}: snapshot path outside work directory')
            if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError(f'{path}: snapshot changed')
        runtime = {k: v for k, v in files[label].items() if not k.startswith('skills/')}
        if shared_files is not None and shared_files != runtime:
            raise ValueError(f'{label}: non-policy sources differ from baseline')
        shared_files = runtime

    rows, contracts = [], {}
    for group, (name, roster) in enumerate(COMPOSITIONS.items()):
        team = {u.lower(): n for u, n in roster.items()}
        for offset in range(0, 75, 5):
            shard = f'{name}_{offset:02d}'
            plan = shards[shard]
            manifest = _read(batch / 'manifests' / f'{shard}.json')
            validate_manifest(manifest)
            episodes = manifest['episodes']
            start = design['seed'] + group * 75 + offset
            if (plan['composition'] != name or plan['manifest_sha256'] != manifest['sha256'] or
                    [e['sc2_seed'] for e in episodes] != list(range(start, start + 5))):
                raise ValueError(f'{shard}: unexpected manifest or seeds')
            runs = {}
            for label in LABELS:
                runs[label], contract = _shard(batch / label / shard, manifest, MODE, files[label])
                if contract != contracts.setdefault(label, contract) or contract[2] != contracts['baseline'][2]:
                    raise ValueError(f'{label}/{shard}: runner sources/parameters/versions changed')
            for i, episode in enumerate(episodes):
                cfg = episode['episode_config']
                if any(Counter(cfg['team_gen'][side]) != team for side in ('ally_team', 'enemy_team')):
                    raise ValueError(f'{shard} episode {i+1}: composition mismatch')
                games = {label: runs[label][i] for label in LABELS}
                states = {g['initial_state_sha256'] for g in games.values()}
                if len(states) != 1:
                    raise ValueError(f'{shard} episode {i+1}: initial state mismatch')
                rows.append(dict(shard=shard, composition=name, layout=_layout(cfg, shard), episode=i+1,
                                 sc2_seed=episode['sc2_seed'], scenario_sha256=episode['sha256'],
                                 manifest_sha256=manifest['sha256'], initial_state_sha256=states.pop(), versions=games))

    report = dict(seed=design['seed'], execution_mode=MODE, labels=list(LABELS), episodes=rows,
                  policy_sha256={label: digest(files[label]) for label in LABELS}, total=_summary(rows),
                  by_layout=_by_layout(rows), by_composition={})
    for name in COMPOSITIONS:
        group_rows = [r for r in rows if r['composition'] == name]
        report['by_composition'][name] = dict(**_summary(group_rows), by_layout=_by_layout(group_rows))
    write_json(batch / 'comparison.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    summarize(parser.parse_args().batch)
