"""Offline comparison of 150 frozen baseline/muster pairs (requires mpyq and s2protocol)."""
import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path

from compare_protoss_parallel import digest, validate_manifest, write_json
from contact_early_report import COMPOSITIONS, SIDES, _check, _run
from contact_factorial_report import MODE
from contact_front_report import METHODOLOGY as FRONT, WINDOW, _first_contact, _layout, _since, _stats, _triples
from contact_micro_report import OUTCOMES, TYPES, _read, _shard

LABELS = ('baseline', 'muster')
TIMES = ('first_dog_mask', 'first_dog_attack_order', 'first_dog_cd', 'first_ally_cd')
METRICS = ('ally_support_fraction', 'hydra_cd_fraction_48', 'first_visible_since_start',
           *(f'{event}_since_{origin}' for event in TIMES for origin in ('start', 'visible')))


def _changes(payload, decision, step, label, where):
    state = payload['state']
    actors = state['allies']
    actions = step['actions']
    actual = {f'u{i}': a for i, a in enumerate(actions)}
    _check(isinstance(actions, list) and all(type(a) is int for a in actions) and
           actual.keys() == actors.keys() and actual == decision['actions'], where, 'step.actions mismatch')
    trace = decision.get('opening_muster')
    if label == 'baseline':
        _check(trace is None, where, 'unexpected opening_muster on baseline')
        return []
    _check(isinstance(trace, dict) and isinstance(trace.get('changes'), dict), where, 'missing muster audit')
    changes = []
    for actor, change in trace['changes'].items():
        before, after = change['before'], change['after']
        _check(actor in actors and actors[actor]['alive'], where, f'invalid muster actor {actor}')
        menu = payload['questions'][actor]['criteria']
        _check(type(before) is int and type(after) is int and 1 <= after <= 5 and before != after == actual[actor]
               and str(before) in menu and str(after) in menu, where, f'{actor}: invalid action change')
        receipt = step['feedback']['actors'][actor]
        status = receipt['submission']
        code = (receipt['response'] or {}).get('code')
        _check(step['feedback']['from_game_loop'] == state['game_loop'] and
               receipt['requested_action_id'] == str(after) and
               (receipt['command'] or {}).get('kind') == ('stop' if after == 1 else 'move') and
               ((status == 'result_unmapped' and code is None) or
                (type(code) is int and status == ('accepted' if code == 1 else 'rejected'))),
               where, f'{actor}: inconsistent submission receipt')
        changes.append(dict(step=decision['step'], game_loop=state['game_loop'], actor=actor,
                            unit_type=actors[actor]['type'], tag=actors[actor]['tag'], before=before, after=after,
                            receipt=receipt))
    return changes


def _behavior(triples, game, label, where):
    contact = start = seen = last = None
    first = dict.fromkeys(TIMES)
    prev, participants, changes = {}, set(), []
    for i, (obs, decision, step) in enumerate(triples):
        payload = obs['payload']
        state = payload['state']
        loop = state['game_loop']
        if start is None:
            start = loop
        _check(state['step_mul'] == 8 and obs['step'] == decision['step'] == step['step'] - 1 == i and
               loop == decision['game_loop'] == step['game_loop'] == start + 8*i, where, 'step/clock mismatch')
        last = loop
        if seen is None and any(u['alive'] for u in state['visible_enemies'].values()):
            seen = loop
        if contact is None:
            contact = _first_contact(payload, where)
            if contact is not None:
                first['first_dog_mask'] = loop
        changes.extend(_changes(payload, decision, step, label, where))
        for actor, unit in state['allies'].items():
            if not unit['alive']:
                continue
            tag, cd = unit['tag'], unit['weapon_cooldown_loops']
            _check(math.isfinite(cd) and cd >= 0, where, 'invalid cooldown')
            restart = tag in prev and cd > max(0, prev[tag] - 8) + .02
            prev[tag] = cd
            if restart and first['first_ally_cd'] is None:
                first['first_ally_cd'] = loop
            if unit['type'] == 'Zergling':
                if restart and first['first_dog_cd'] is None:
                    first['first_dog_cd'] = loop
                if decision['actions'][actor] >= 6 and first['first_dog_attack_order'] is None:
                    first['first_dog_attack_order'] = loop
            elif unit['type'] == 'Hydralisk' and restart and seen is not None and seen < loop <= seen + WINDOW:
                participants.add(tag)
    end = None if seen is None else seen + WINDOW
    complete = end is not None and last >= end
    game.update(start_loop=start, first_visible_loop=seen, first_visible_since_start=_since(seen, start),
                first_dog_mask=contact, ally_support_fraction=contact['ally_support_fraction'] if contact else None,
                last_observation_loop=last, hydra_cd_window_end=end,
                hydra_cd_window_status='no_visibility' if end is None else 'complete' if complete else 'truncated',
                hydra_cd_participant_tags_48=sorted(participants) if complete else None,
                hydra_cd_participants_48=len(participants) if complete else None,
                hydra_cd_fraction_48=len(participants) / game['ally_hydras'] if complete else None,
                opening_muster_changes=changes)
    for event, loop in first.items():
        game[event + '_loop'] = loop
        game[event + '_since_start'] = _since(loop, start)
        game[event + '_since_visible'] = _since(loop, seen)


def _games(work, manifest, name, label, files):
    games, contract = _shard(work, manifest, MODE, files)
    # _run independently binds the event stream, roster, terminal deaths, engine
    # damage ticks and replay to the same scenario before we interpret behavior.
    opening = _run(work, name, manifest)
    with (work / 'events.jsonl').open('rb') as stream:
        for i, (game, verified) in enumerate(zip(games, opening), 1):
            _check(all(game[k] == verified[k] for k in ('outcome', 'initial_state_sha256', 'damage_by_type')),
                   work, 'replay/episode mismatch')
            game.update({k: verified[k] for k in ('first_hydra_death_side', 'first_hydra_death_loops', 'ally_hydras')})
            where = f'{work}/episode-{i:04d}'
            _behavior(_triples(stream, i, where), game, label, where)
    return games, contract


def _summary(rows):
    pairs = [r['versions'] for r in rows]
    versions = {}
    for label in LABELS:
        games = [v[label] for v in pairs]
        outcomes = Counter(g['outcome'] for g in games)
        changes = [c for g in games for c in g['opening_muster_changes']]
        versions[label] = dict(games=len(games), outcomes={o: outcomes[o] for o in OUTCOMES},
            win_rate=outcomes['win']/len(games), metrics={k: _stats([g[k] for g in games]) for k in METRICS},
            hydra_cd_window_status={s: sum(g['hydra_cd_window_status'] == s for g in games)
                                    for s in ('no_visibility', 'truncated', 'complete')},
            first_hydra_death_side={s: sum(g['first_hydra_death_side'] == s for g in games)
                                    for s in (*SIDES, 'none', 'same_frame')},
            micro_trigger={k: sum(g['micro_trigger'][k] for g in games) for k in ('authorizations', 'windows', 'submitted')},
            damage_by_type={s: {u: sum(g['damage_by_type'][s][u] for g in games) for u in TYPES} for s in SIDES},
            opening_muster=dict(changed_games=sum(bool(g['opening_muster_changes']) for g in games),
                                submitted_changes=len(changes), by_unit=dict(Counter(c['unit_type'] for c in changes)),
                                receipts=dict(Counter(c['receipt']['submission'] for c in changes))))
    table = {a: dict.fromkeys(OUTCOMES, 0) for a in OUTCOMES}
    for v in pairs:
        table[v['baseline']['outcome']][v['muster']['outcome']] += 1
    deltas = {k: _stats([_since(v['muster'][k], v['baseline'][k]) for v in pairs]) for k in METRICS}
    damage = {s: {u: versions['muster']['damage_by_type'][s][u] - versions['baseline']['damage_by_type'][s][u]
                  for u in TYPES} for s in SIDES}
    return dict(pairs=len(pairs), versions=versions, paired_outcomes=table, paired_vs_baseline={'muster': dict(
        gained_wins=sum(table[o]['win'] for o in OUTCOMES if o != 'win'),
        dropped_wins=sum(table['win'][o] for o in OUTCOMES if o != 'win'))},
        paired_delta_muster_minus_baseline=deltas, damage_delta_muster_minus_baseline=damage)


def _by_layout(rows):
    return {layout: _summary([r for r in rows if r['layout'] == layout]) for layout in sorted({r['layout'] for r in rows})}


def summarize(batch: Path):
    """Validate all 30 five-game shards per label, then write and return comparison.json."""
    batch = Path(batch)
    design = _read(batch / 'design.json')
    files, shards = design['policy_files'], design['shards']
    _check(tuple(design['labels']) == LABELS and set(files) == set(LABELS) and type(design['seed']) is int and
           design['execution_mode'] == MODE, batch, 'invalid muster batch design')
    _check(set(shards) == {f'{n}_{i:02d}' for n in COMPOSITIONS for i in range(0, 75, 5)},
           batch, 'expected 15 five-game shards per composition')
    shared = {}
    for label in LABELS:
        root = batch / label
        for name, checksum in files[label].items():
            path = root / name
            _check(not Path(name).is_absolute() and path.resolve().is_relative_to(root.resolve()), path, 'unsafe path')
            _check(hashlib.sha256(path.read_bytes()).hexdigest() == checksum, path, 'snapshot changed')
        shared[label] = {k: v for k, v in files[label].items() if not k.startswith('skills/')}
    _check(shared['baseline'] == shared['muster'], batch, 'non-policy sources differ')

    rows, contracts = [], {}
    for group, (name, roster) in enumerate(COMPOSITIONS.items()):
        team = {u.lower(): n for u, n in roster.items()}
        for offset in range(0, 75, 5):
            stem = f'{name}_{offset:02d}'
            m = _read(batch / 'manifests' / f'{stem}.json')
            validate_manifest(m)
            start = design['seed'] + 75*group + offset
            episodes = m['episodes']
            _check(shards[stem]['composition'] == name and shards[stem]['manifest_sha256'] == m['sha256'] and
                   [e['sc2_seed'] for e in episodes] == list(range(start, start+5)), stem, 'unplanned scenarios')
            runs = {}
            for label in LABELS:
                runs[label], contract = _games(batch / label / stem, m, name, label, files[label])
                _check(contract == contracts.setdefault(label, contract) and contract[1:] == contracts['baseline'][1:],
                       stem, 'runner sources, parameters or versions differ')
            _check({k:v for k,v in contracts['baseline'][0].items() if not k.startswith('skills/')} ==
                   {k:v for k,v in contracts['muster'][0].items() if not k.startswith('skills/')},
                   stem, 'runner source coverage differs between labels')
            for i, e in enumerate(episodes):
                cfg = e['episode_config']
                _check(all(Counter(cfg['team_gen'][s]) == team for s in ('ally_team', 'enemy_team')),
                       stem, 'composition mismatch')
                versions = {label: runs[label][i] for label in LABELS}
                initial = versions['baseline']['initial_state_sha256']
                _check(initial == versions['muster']['initial_state_sha256'], stem, f'initial state mismatch: {i+1}')
                rows.append(dict(shard=stem, episode=i+1, sc2_seed=e['sc2_seed'], scenario_sha256=e['sha256'],
                    initial_state_sha256=initial, manifest_sha256=m['sha256'], layout=_layout(cfg, stem),
                    composition=name, versions=versions))
    methodology = {k: FRONT[k] for k in ('contact', 'support', 'participation', 'aggregation')}
    methodology.update(timing='All timings are macro observation loops relative to episode start or first team visibility. '
                              'First ally CD includes dogs and hydras. A legal target, requested attack, CD proxy and damage are distinct.',
        missing='No dog mask means null support, not zero. The CD window is anchored independently at visibility. '
                'No visibility or no closing macro observation means null, even if some restarts were observed.',
        damage='Damage sums use both players\' actual source-type ticks verified against the replay. First hydra death '
               'is the victim side from the replay tracker through terminal; same_frame is a tie.',
        audit='Only opening_muster.changes with distinct legal before/after, matching skill_step, step.actions and '
              'actor receipt are counted. Held actors, candidates, accepted orders and micro authorizations are not shots.')
    report = dict(seed=design['seed'], execution_mode=MODE, labels=list(LABELS), methodology=methodology,
                  policy_sha256={v: digest(files[v]) for v in LABELS}, episodes=rows, total=_summary(rows),
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
