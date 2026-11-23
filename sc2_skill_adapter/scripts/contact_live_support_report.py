"""Offline audit of paired, single-game baseline/live-support SC2 runs."""
import json
from collections import Counter
from pathlib import Path

if __package__:
    from .contact_early_report import COMPOSITIONS, OUTCOMES, SIDES, TYPES, _check, _digest, _read, _run
    from .contact_front_report import QUOTA, _layout, _triples
else:
    from contact_early_report import COMPOSITIONS, OUTCOMES, SIDES, TYPES, _check, _digest, _read, _run
    from contact_front_report import QUOTA, _layout, _triples

LABELS = ('baseline', 'support')


def _audit(folder, label):
    changes = []
    with (folder / 'events.jsonl').open('rb') as stream:
        for i, (obs, decision, submitted) in enumerate(_triples(stream, 1, folder)):
            where = f'{folder}: decision {i}'
            state = obs['payload']['state']
            loop = state['game_loop']
            _check(obs['step'] == decision['step'] == submitted['step'] - 1 == i and
                   loop == decision['game_loop'] == submitted['game_loop'], where, 'step/clock mismatch')
            actions = submitted['actions']
            allies = state['allies']
            _check(isinstance(actions, list) and all(type(a) is int for a in actions), where,
                   'invalid submitted action vector')
            actual = {f'u{j}': a for j, a in enumerate(actions)}
            _check(actual.keys() == allies.keys() and actual == decision['actions'], where,
                   'skill_step.actions vs step.actions mismatch')
            trace = decision.get('live_support', {})
            _check(isinstance(trace, dict) and (label == 'support' or not trace), where,
                   'invalid or unexpected live_support trace')
            for actor, entry in trace.items():
                _check(actor in allies and isinstance(entry, dict), where, f'invalid trace actor/entry: {actor}')
                before, after = entry.get('before'), entry.get('after')
                _check(type(before) is int and type(after) is int and after == actual[actor], where,
                       f'{actor}: invalid before/after or after differs from step.actions')
                if before == after:
                    continue
                unit = allies[actor]
                _check(unit['type'] == 'Hydralisk' and unit['alive'], where,
                       f'{actor}: support changed {unit["type"]}; only live Hydralisks may change')
                changes.append(dict(step=i, game_loop=loop, actor=actor, unit_type=unit['type'],
                                    before=before, after=after))
    return changes


def _summary(rows):
    table = {o: dict.fromkeys(OUTCOMES, 0) for o in OUTCOMES}
    versions = {}
    for label in LABELS:
        games = [r['versions'][label] for r in rows]
        versions[label] = dict(games=len(games), outcomes={o: sum(g['outcome'] == o for g in games) for o in OUTCOMES},
            damage_by_type={s: {u: sum(g['damage_by_type'][s][u] for g in games) for u in TYPES} for s in SIDES})
    for r in rows:
        table[r['versions']['baseline']['outcome']][r['versions']['support']['outcome']] += 1
    counts = [len(r['versions']['support']['live_support_changes']) for r in rows]
    return dict(pairs=len(rows), versions=versions, paired_outcomes=table,
                gained_wins=sum(table[o]['win'] for o in OUTCOMES if o != 'win'),
                dropped_wins=sum(table['win'][o] for o in OUTCOMES if o != 'win'),
                support_changed_games=sum(n > 0 for n in counts), support_submitted_changes=sum(counts))


def summarize(batch: Path, jobs: list):
    """Validate 30 paired runs in batch/{baseline,support}/<manifest stem> and write comparison.json."""
    batch = Path(batch)
    parts = {}
    for label, name, path in jobs:
        path = Path(path)
        _check(label in LABELS and name in COMPOSITIONS, path, 'unknown policy/composition')
        group, source, labels = parts.setdefault(path.stem, (name, path, set()))
        _check((group, source) == (name, path) and label not in labels, path, 'duplicate/conflicting shard')
        labels.add(label)
    _check(len(parts) == 30, batch, 'expected 30 paired single-game manifests')
    design, counts, seeds = [], Counter(), set()
    for stem, (name, path, labels) in sorted(parts.items()):
        _check(labels == set(LABELS), path, 'missing policy')
        m = _read(path)
        _check(m['version'] == 1 and len(m['episodes']) == 1 and m['sha256'] ==
               _digest({k: v for k, v in m.items() if k != 'sha256'}), path, 'invalid single-game manifest')
        c, = m['episodes']
        cfg, seed = c['episode_config'], c['sc2_seed']
        team = {u.lower(): n for u, n in COMPOSITIONS[name].items()}
        _check(type(seed) is int and (name, seed) not in seeds and _digest(cfg) == c['sha256'] and
               all(Counter(cfg['team_gen'][k]) == team for k in ('ally_team', 'enemy_team')),
               path, 'invalid seed, scenario hash or composition')
        seeds.add((name, seed))
        layout = _layout(cfg, path)
        counts[name, layout] += 1
        design.append((stem, name, layout, m))
    _check(counts == {(n, l): size for n in COMPOSITIONS for l, size in QUOTA.items()}, batch,
           f'expected 10 scattered and 5 center pairs per composition, got {counts}')

    rows = []
    for stem, name, layout, m in design:
        versions = {}
        for label in LABELS:
            folder = batch / label / stem
            game, = _run(folder, name, m)  # Replay-verified damage, roster, terminal deaths and outcome.
            versions[label] = {k: game[k] for k in ('initial_state_sha256', 'outcome', 'damage_by_type')}
            changes = _audit(folder, label)
            if label == 'support':
                versions[label]['live_support_changes'] = changes
        a, b = (versions[v] for v in LABELS)
        _check(a['initial_state_sha256'] == b['initial_state_sha256'], stem, 'paired initial states differ')
        delta = {s: {u: b['damage_by_type'][s][u] - a['damage_by_type'][s][u]
                     for u in ('Zergling', 'Hydralisk')} for s in SIDES}
        c, = m['episodes']
        rows.append(dict(name=name, shard=stem, layout=layout, episode=1, sc2_seed=c['sc2_seed'],
                         scenario_sha256=c['sha256'], manifest_sha256=m['sha256'], versions=versions,
                         damage_delta_support_minus_baseline=delta))

    def by_layout(group):
        return {l: _summary([r for r in group if r['layout'] == l]) for l in QUOTA}

    report = dict(episodes=rows, total=_summary(rows), by_layout=by_layout(rows), by_composition={},
                  methodology='Damage is summed by actual source type and checked against replay ticks. '
                  'Rows of paired_outcomes are baseline, columns support; gains/drops compare win vs non-win. '
                  'Changes count traced actor decisions whose before differs from the submitted after, not hits.')
    for name in COMPOSITIONS:
        group = [r for r in rows if r['name'] == name]
        report['by_composition'][name] = dict(**_summary(group), by_layout=by_layout(group))
    (batch / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report
