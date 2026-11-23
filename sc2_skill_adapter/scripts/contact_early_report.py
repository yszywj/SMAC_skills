"""Offline paired contact report; requires mpyq and s2protocol, not SC2."""
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean

if __package__:
    from .contact_ab_report import OUTCOMES, TYPES, _digest, _read
    from .make_damage_map import replay_resource_damage
else:
    from contact_ab_report import OUTCOMES, TYPES, _digest, _read
    from make_damage_map import replay_resource_damage

LABELS = ('baseline', 'early')
SIDES = ('ally', 'enemy')
COMPOSITIONS = {f'zerg_{n}zergling_0baneling_{10-n}hydralisk':
                {'Zergling': n, 'Hydralisk': 10-n} for n in (3, 6)}


def _check(ok, where, why):
    if not ok:
        raise ValueError(f'{where}: {why}')


def _deaths(path, rosters, start, end):
    import mpyq
    from s2protocol import versions

    expected = {tag: (int(side), unit) for side, roster in rosters.items() for tag, unit in roster.items()}
    born, deaths = set(), {}
    with path.open('rb') as source:
        archive = mpyq.MPQArchive(source, listfile=False)
        header = versions.latest().decode_replay_header(archive.header['user_data_header']['content'])
        proto = versions.build(header['m_version']['m_baseBuild'])
        for e in proto.decode_replay_tracker_events(archive.read_file('replay.tracker.events')):
            kind = e['_event'].rsplit('.', 1)[-1]
            if kind not in ('SUnitBornEvent', 'SUnitDiedEvent'):
                continue
            tag = str((1 << 32) | proto.unit_tag(e['m_unitTagIndex'], e['m_unitTagRecycle']))
            if tag not in expected:
                continue
            loop = e['_gameloop']
            if kind == 'SUnitBornEvent':
                _check(tag not in born and loop <= start and expected[tag] == (
                    e['m_controlPlayerId'], e['m_unitTypeName'].decode()), path, f'birth mismatch {tag}')
                born.add(tag)
            elif loop <= end:  # The replay also contains post-terminal telemetry steps.
                _check(tag in born and tag not in deaths and loop > start, path, f'invalid death {tag}')
                deaths[tag] = loop
    _check(born == expected.keys(), path, 'missing roster births')
    return deaths


def _opening(rosters, deaths, jumps):
    def first(player, kind):
        return min((deaths[t] for t, u in rosters[player].items() if u == kind and t in deaths), default=None)

    dog = first('1', 'Zergling')
    a, b = (first(p, 'Hydralisk') for p in ('1', '2'))
    side = ('none' if a is None and b is None else 'same_frame' if a == b else
            'ally' if b is None or (a is not None and a < b) else 'enemy')
    n = sum(u == 'Hydralisk' for u in rosters['1'].values())
    return dict(first_ally_zergling_death_loop=dog, first_hydra_death_side=side,
                first_hydra_death_loops=dict(ally=a, enemy=b), ally_hydras=n,
                first_cd_restart_loop_by_hydra=jumps,
                participation_before=None if dog is None else sum(k < dog for k in jumps.values()) / n,
                participation_through=None if dog is None else sum(k <= dog for k in jumps.values()) / n)


def _run(folder, name, manifest):
    _check(_read(folder / 'manifest.json') == manifest, folder, 'copied manifest mismatch')
    configs = manifest['episodes']
    records = _read(folder / 'episodes.json')
    _check(len(records) == len(configs), folder, 'episode count mismatch')
    games = []
    completed, active = 0, False
    with (folder / 'events.jsonl').open('rb') as lines:
        for line in lines:
            e = json.loads(line)
            kind = e['event']
            if kind not in ('episode_started', 'skill_observation', 'episode_completed'):
                continue
            ep = completed + 1
            where = f'{folder}/episode-{ep:04d}'
            _check(e['episode'] == ep <= len(configs), where, 'unexpected episode number')
            row, c = records[completed], configs[completed]
            if kind == 'episode_started':
                _check(not active and row['episode'] == ep and row['scenario_sha256'] == c['sha256'] and
                       re.fullmatch('[0-9a-f]{64}', row['initial_state_sha256']) and row['outcome'] in OUTCOMES and
                       all(e[k] == row[k] for k in ('scenario_sha256', 'initial_state_sha256')),
                       where, 'unexpected start or initial state')
                g = dict(initial_state_sha256=row['initial_state_sha256'], outcome=row['outcome'], damage_by_type={})
                stem = f'episode-{ep:04d}'
                damage = _read(folder / 'damage_by_type' / (stem + '.json'))
                _check(damage['episode'] == ep and damage['scenario_sha256'] == c['sha256'] and
                       damage['manifest_sha256'] == manifest['sha256'], where, 'damage record mismatch')
                score = _read(folder / 'score_details' / (stem + '.json'))
                start, end = damage['begin_request_loop'], damage['terminal_loop']
                _check(all(score[k] == row[k] for k in ('episode', 'scenario_sha256', 'steps', 'outcome'))
                       and score['manifest_sha256'] == manifest['sha256'] and score['sc2_seed'] == c['sc2_seed']
                       and start == e['game_loop'] == score['initial']['game_loop']
                       and end == start + 8 * row['steps'] == score['final']['game_loop'], where, 'score/window mismatch')
                rosters, ticks = {}, []
                for p, side in zip(('1', '2'), SIDES):
                    player = damage['players'][p]
                    rosters[p] = player['initial_roster_by_api_tag']
                    tick = player['ticks_by_source_type']
                    values = player['damage_dealt_by_source_type']
                    _check(Counter(rosters[p].values()) == COMPOSITIONS[name] and set(tick) == set(values) == set(TYPES)
                           and damage['ticks_per_damage'] == 4096 and all(type(tick[u]) is int and tick[u] >= 0 and
                           type(values[u]) in (int, float) and tick[u] == 4096 * values[u] for u in TYPES),
                           where, 'invalid roster/ticks')
                    g['damage_by_type'][side] = values
                    ticks.extend(tick[u] for u in TYPES)
                path = folder / 'replays' / (stem + '.SC2Replay')
                _check(not (rosters['1'].keys() & rosters['2'].keys()), where, 'overlapping player tags')
                _check(damage['freeze_loop_range'] == [end, end+2] and replay_resource_damage(path) ==
                       [dict(freeze_loop_range=[end, end+2], ticks=ticks)], where, 'damage/replay mismatch')
                deaths = _deaths(path, rosters, start, end)
                for p, key in [('1', 'dead_allies'), ('2', 'dead_enemies')]:
                    _check(sum(t in deaths for t in rosters[p]) == row['info'][key], where, f'{key} mismatch')
                _check((row['outcome'] == 'win') == bool(row['info'].get('battle_won')), where, 'battle result mismatch')
                active, step, prev, jumps = True, 0, {}, {}
            elif kind == 'skill_observation':
                _check(active and e['step'] == step < row['steps'], where, 'missing/out-of-order observation')
                s = e['payload']['state']
                loop = s['game_loop']
                units = s['allies'].values()
                _check(s['step_mul'] == 8 and loop == start + 8 * step and len(units) == 10 and
                       {u['tag']: u['type'] for u in units} == rosters['1'], where, 'observation roster/clock mismatch')
                for u in units:
                    tag = u['tag']
                    _check(u['alive'] == (deaths.get(tag, math.inf) > loop), where, f'alive mismatch {tag}')
                    if u['alive'] and u['type'] == 'Hydralisk':
                        cd = u['weapon_cooldown_loops']
                        _check(math.isfinite(cd) and cd >= 0, where, 'invalid cooldown')
                        if tag in prev and cd > max(0, prev[tag] - 8) + 0.02:
                            jumps.setdefault(tag, loop)
                        prev[tag] = cd
                step += 1
            else:
                _check(active and step == row['steps'] and all(row[k] == e[k] for k in row), where, 'completion mismatch')
                g.update(_opening(rosters, deaths, jumps))
                games.append(g)
                completed += 1
                active = False
    _check(not active and completed == len(records), folder, 'incomplete event stream')
    return games


def _summary(rows):
    table = {o: dict.fromkeys(OUTCOMES, 0) for o in OUTCOMES}
    for r in rows:
        v = r['versions']
        table[v['baseline']['outcome']][v['early']['outcome']] += 1
    versions = {}
    for label in LABELS:
        games = [r['versions'][label] for r in rows]
        eligible = [g for g in games if g['first_ally_zergling_death_loop'] is not None]
        versions[label] = dict(games=len(games), outcomes={o: sum(g['outcome'] == o for g in games) for o in OUTCOMES},
            first_hydra_death_side={s: sum(g['first_hydra_death_side'] == s for g in games)
                                   for s in (*SIDES, 'same_frame', 'none')},
            eligible_games=len(eligible), no_ally_dog_death=len(games)-len(eligible),
            mean_participation_before=mean(g['participation_before'] for g in eligible) if eligible else None,
            mean_participation_through=mean(g['participation_through'] for g in eligible) if eligible else None,
            damage_by_type={s: {u: sum(g['damage_by_type'][s][u] for g in games) for u in TYPES} for s in SIDES})
    return dict(versions=versions, paired_outcomes=table, gained_wins=sum(table[o]['win'] for o in OUTCOMES if o != 'win'),
                dropped_wins=sum(table['win'][o] for o in OUTCOMES if o != 'win'))


def summarize(batch: Path, jobs: list):
    """Validate 2 x 150 games before writing comparison.json; return the same report."""
    parts = {}
    for label, name, path in jobs:
        path = Path(path)
        _check(name in COMPOSITIONS and label in LABELS, path, 'unknown label/composition')
        group, source, labels = parts.setdefault(path.stem, (name, path, set()))
        _check(group == name and source == path and label not in labels, path, 'duplicate or conflicting shard')
        labels.add(label)
    _check(Counter(n for n, _, _ in parts.values()) == dict.fromkeys(COMPOSITIONS, 15), batch,
           'expected 15 paired five-game shards per composition')
    rows, seeds = [], set()
    for stem, (name, path, labels) in sorted(parts.items()):
        _check(labels == set(LABELS), path, 'missing policy')
        m = _read(path)
        _check(m['version'] == 1 and len(m['episodes']) == 5 and m['sha256'] ==
               _digest({k: v for k, v in m.items() if k != 'sha256'}), path, 'invalid five-game manifest')
        for c in m['episodes']:
            cfg, seed = c['episode_config'], c['sc2_seed']
            counts = {u.lower(): n for u, n in COMPOSITIONS[name].items()}
            _check(type(seed) is int and (name, seed) not in seeds and c['sha256'] == _digest(cfg)
                   and all(Counter(cfg['team_gen'][s]) == counts for s in ('ally_team', 'enemy_team')),
                   path, 'invalid scenario, composition or repeated seed')
            seeds.add((name, seed))
        runs = {label: _run(batch / label / stem, name, m) for label in LABELS}
        for i, c in enumerate(m['episodes']):
            games = {label: runs[label][i] for label in LABELS}
            _check(len({g['initial_state_sha256'] for g in games.values()}) == 1, path, f'initial state mismatch: {i+1}')
            rows.append(dict(name=name, shard=stem, episode=i+1, sc2_seed=c['sc2_seed'], scenario_sha256=c['sha256'],
                             manifest_sha256=m['sha256'], versions=games))
    report = dict(episodes=rows, total=_summary(rows), by_composition={n: _summary([r for r in rows if r['name'] == n])
                                                                      for n in COMPOSITIONS},
                  methodology='Participation is a per-game fraction of initial allied hydras with an observed CD restart '
                  '(cd > max(0, previous_cd - 8) + 0.02). Before uses sample loop < first allied dog death; through '
                  'includes the death frame. No dog death: null, excluded from means. Proxy, not exact shots/hits. '
                  'First hydra side denotes the victim; replay deaths after terminal are ignored. Damage values are sums.')
    (batch / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report
