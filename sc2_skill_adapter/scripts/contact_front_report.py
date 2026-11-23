"""Offline first-contact comparison for paired single-episode Zerg route runs."""
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

if __package__:
    from .contact_early_report import COMPOSITIONS, OUTCOMES, SIDES, TYPES, _check, _digest, _read, _run
else:
    from contact_early_report import COMPOSITIONS, OUTCOMES, SIDES, TYPES, _check, _digest, _read, _run

LABELS = ('baseline', 'route')
QUOTA = {'scattered': 10, 'center': 5}
WINDOW = 48


def _layout(cfg, where):
    a, b = (cfg[f'{s}_start_positions']['item'] for s in SIDES)
    _check(all(len(side) == 10 and all(len(p) == 2 and all(math.isfinite(x) for x in p) for p in side)
               for side in (a, b)), where, 'invalid spawn coordinates')
    if all(p == [16, 16] for p in a):
        return 'center'
    _check(all(x <= 15 and math.isclose(x + ex, 32) and math.isclose(y, ey)
               for (x, y), (ex, ey) in zip(a, b)), where, 'neither center nor reflected scatter')
    return 'scattered'


def _fraction(n, d):
    return n / d if n is not None and d else None


def _since(loop, start):
    return loop - start if start is not None and loop is not None else None


def _audit_step(label, decision, executed, payload, audit, where):
    vector = executed['actions']
    proposed = decision['actions']
    _check({f'u{i}': a for i, a in enumerate(vector)} == proposed, where, 'skill_step vs submitted actions mismatch')
    trace = decision.get('front_route')
    if label == 'baseline':
        _check(trace is None, where, 'unexpected route trace on baseline')
        return
    _check(isinstance(trace, dict), where, 'missing route audit')
    changes = trace['changes']
    _check(isinstance(changes, dict), where, 'invalid route changes')
    for actor, change in changes.items():
        before, after = change['old_action_id'], change['new_action_id']
        menu = payload['questions'][actor]['criteria']
        _check(type(before) is int and type(after) is int and before != after == proposed[actor] and
               str(before) in menu and str(after) in menu, where, f'invalid route change for {actor}')
        feedback = executed['feedback']['actors'][actor]
        _check(int(feedback['requested_action_id']) == after, where, f'wrong submitted action for {actor}')
        audit.append(dict(loop=executed['game_loop'], decision_step=decision['step'], actor=actor,
                          unit_type=payload['state']['allies'][actor]['type'], before=before,
                          submitted=after, submission=feedback['submission'], reason=change['reason'],
                          predicted_old=change['old'], predicted_new=change['new']))


def _covered(relations, key):
    values = [rel[key] for rel in relations]
    if True in values:
        return True
    return None if None in values else False


def _first_contact(payload, where):
    s = payload['state']
    allies = {a: u for a, u in s['allies'].items() if u['alive']}
    enemies = {e: u for e, u in s['visible_enemies'].items() if u['alive']}
    relations = s['relations']
    legal = {}
    for a in allies:
        menu = payload['questions'][a]['criteria']
        legal[a] = {f'e{int(k)-6}' for k in menu if int(k) >= 6}
        _check(all(type(relations[a][e]['smac_attack_available']) is bool for e in enemies)
               and legal[a] == {e for e in enemies if relations[a][e]['smac_attack_available']},
               where, f'{a}: attack mask/menu/visibility mismatch')
    dogs = sorted((a for a in allies if allies[a]['type'] == 'Zergling' and legal[a]), key=lambda a: int(a[1:]))
    if not dogs:
        return None
    dog = dogs[0]
    targets = legal[dog]
    pos = allies[dog]['position']
    local_enemies = {e for e, u in enemies.items() if any(math.dist(u['position'], enemies[t]['position']) <= 3
                                                          for t in targets)}
    local_allies = {a for a, u in allies.items() if math.dist(u['position'], pos) <= 3}
    guns = [a for a in allies if allies[a]['type'] == 'Hydralisk']
    foes = [e for e in enemies if enemies[e]['type'] == 'Hydralisk']
    friendly = {a: _covered((relations[a][e] for e in local_enemies & legal[a]), 'in_weapon_range') for a in guns}
    opposing = {e: _covered((relations[a][e] for a in local_allies), 'in_enemy_weapon_range') for e in foes}
    n, m = (None if None in d.values() else sum(d.values()) for d in (friendly, opposing))
    return dict(loop=s['game_loop'], dog=dog, dog_position=pos, tied_dogs=dogs,
                legal_target_positions={e: enemies[e]['position'] for e in sorted(targets)},
                local_enemies=sorted(local_enemies), local_allies=sorted(local_allies),
                ally_support_by_hydra=friendly, enemy_coverage_by_hydra=opposing,
                live_ally_hydras=len(guns), visible_enemy_hydras=len(foes),
                ally_support_count=n, enemy_cover_count=m, ally_support_fraction=_fraction(n, len(guns)),
                local_advantage=n-m if n is not None and m is not None else None,
                enemy_cover_fraction=_fraction(m, len(foes)), ally_to_enemy_support_ratio=_fraction(n, m))


def _front_game(events, game, label, where):
    """Enrich one _run result from its observation -> skill_step -> submitted step triples."""
    contact = seen = dog_cd = dog_attack = last = None
    prev, participants = {}, set()
    audit = []
    steps = 0
    for obs, decision, executed in events:
        payload = obs['payload']
        s = payload['state']
        loop = s['game_loop']
        _check(obs['step'] == decision['step'] == executed['step'] - 1 == steps and
               decision['game_loop'] == executed['game_loop'] == loop, where, 'decision/step misalignment')
        steps += 1
        last = loop
        if seen is None and any(u['alive'] for u in s['visible_enemies'].values()):
            seen = loop
        if contact is None:
            contact = _first_contact(payload, where)
        actions = executed['actions']
        _check(isinstance(actions, list) and len(actions) == len(s['allies']) and
               all(type(k) is int for k in actions), where, 'invalid submitted action vector')
        for i, action in enumerate(actions):
            a = f'u{i}'
            u = s['allies'][a]
            if not u['alive']:
                continue
            cd = u['weapon_cooldown_loops']
            _check(math.isfinite(cd) and cd >= 0, where, 'invalid cooldown')
            restart = a in prev and cd > max(0, prev[a] - s['step_mul']) + 0.02
            prev[a] = cd
            if u['type'] == 'Zergling':
                if restart and dog_cd is None:
                    dog_cd = loop
                if action >= 6 and dog_attack is None:
                    dog_attack = loop
            elif restart and seen is not None and seen < loop <= seen + WINDOW:
                participants.add(u['tag'])
        _audit_step(label, decision, executed, payload, audit, where)
    full = seen is not None and last >= seen + WINDOW
    game.update(first_visible_loop=seen, first_dog_mask=contact,
                first_dog_mask_since_visible=_since(contact['loop'] if contact else None, seen),
                first_dog_attack_order_loop=dog_attack, first_dog_attack_order_since_visible=_since(dog_attack, seen),
                first_dog_cd_loop=dog_cd, first_dog_cd_since_visible=_since(dog_cd, seen),
                hydra_cd_window_end=None if seen is None else seen + WINDOW,
                last_observation_loop=last, hydra_cd_window_complete=full,
                hydra_cd_participant_tags_48=sorted(participants) if full else None,
                hydra_cd_participants_48=len(participants) if full else None,
                hydra_cd_fraction_48=len(participants) / game['ally_hydras'] if full else None,
                route_changes=audit if label == 'route' else None)
    return steps


def _triples(stream, episode, where, *, with_micro=False):
    obs = decision = window = None
    count = 0
    for line in stream:
        e = json.loads(line)
        kind = e['event']
        if with_micro and kind == 'micro_window_closed':
            _check(decision is not None and window is None and e['episode'] == episode and
                   e['macro_step'] == decision['step'], where, 'unmatched or duplicate micro window')
            window = e
            continue
        if kind not in ('skill_observation', 'skill_step', 'step', 'episode_completed'):
            continue
        _check(e['episode'] == episode, where, 'unexpected episode')
        if kind == 'skill_observation':
            _check(obs is None and decision is None, where, 'incomplete decision')
            obs = e
        elif kind == 'skill_step':
            _check(obs is not None and decision is None, where, 'unmatched skill_step')
            decision = e
        elif kind == 'step':
            _check(obs is not None and decision is not None, where, 'unmatched submitted step')
            if with_micro:
                _check(window is not None, where, 'missing micro window')
                e['micro_window_closed'] = window
            yield obs, decision, e
            obs = decision = window = None
            count += 1
        else:
            _check(obs is None and count == e['steps'] > 0, where, 'incomplete submitted steps')
            return
    raise ValueError(f'{where}: missing episode_completed')


def _stats(values):
    observed = [v for v in values if v is not None]
    return dict(n=len(observed), missing=len(values)-len(observed), mean=mean(observed) if observed else None)


def _metrics(g):
    keys = ('first_visible_loop', 'first_dog_mask_since_visible', 'first_dog_attack_order_since_visible', 'first_dog_cd_since_visible',
            'hydra_cd_participants_48', 'hydra_cd_fraction_48', 'participation_before', 'participation_through')
    contact = g['first_dog_mask'] or {}
    return {**{k: g[k] for k in keys},
            **{k: contact.get(k) for k in ('live_ally_hydras', 'visible_enemy_hydras', 'local_advantage',
                                           'ally_support_count', 'enemy_cover_count', 'ally_support_fraction',
                                           'enemy_cover_fraction', 'ally_to_enemy_support_ratio')},
             **{f'first_{s}_hydra_death_loop': g['first_hydra_death_loops'][s] for s in SIDES}}


def _ling_output(pairs):
    damage = [(p['baseline']['damage_by_type']['ally']['Zergling'],
               p['route']['damage_by_type']['ally']['Zergling']) for p in pairs]
    baseline, route = (sum(d[i] for d in damage) for i in (0, 1))
    delays = [(p['baseline']['first_dog_cd_since_visible'], p['route']['first_dog_cd_since_visible'])
              for p in pairs]
    return dict(pairs=len(pairs), baseline_damage=baseline, route_damage=route,
                relative_damage_change=(route-baseline)/baseline if baseline else None,
                increased=sum(b>a for a,b in damage), unchanged=sum(b==a for a,b in damage),
                decreased=sum(b<a for a,b in damage),
                paired_first_cycle_delay_loops=_stats([b-a if a is not None and b is not None else None
                                                       for a,b in delays]))


def _summary(rows):
    versions, metrics = {}, {}
    for label in LABELS:
        games = [r['versions'][label] for r in rows]
        metrics[label] = [_metrics(g) for g in games]
        versions[label] = dict(games=len(games), outcomes={o: sum(g['outcome'] == o for g in games) for o in OUTCOMES},
            first_hydra_death_side={s: sum(g['first_hydra_death_side'] == s for g in games)
                                    for s in (*SIDES, 'same_frame', 'none')},
            metrics={k: _stats([m[k] for m in metrics[label]]) for k in metrics[label][0]},
            damage_by_type={s: {u: sum(g['damage_by_type'][s][u] for g in games) for u in TYPES} for s in SIDES})
    pairs = [r['versions'] for r in rows]
    table = {o: dict.fromkeys(OUTCOMES, 0) for o in OUTCOMES}
    shifts = []
    for pair in pairs:
        table[pair['baseline']['outcome']][pair['route']['outcome']] += 1
        contacts = [pair[v]['first_dog_mask'] for v in LABELS]
        shifts.append(math.dist(*(c['dog_position'] for c in contacts)) if all(contacts) else None)
    diffs = {k: _stats([b[k] - a[k] if a[k] is not None and b[k] is not None else None
                        for a, b in zip(metrics['baseline'], metrics['route'])]) for k in metrics['baseline'][0]}
    changes = [r for p in pairs for r in p['route']['route_changes']]
    return dict(pairs=len(pairs), versions=versions, paired_outcomes=table,
                ling_output=dict(all_pairs=_ling_output(pairs),
                                 triggered_only_descriptive=_ling_output([p for p in pairs if p['route']['route_changes']])),
                gained_wins=sum(table[o]['win'] for o in OUTCOMES if o != 'win'),
                dropped_wins=sum(table['win'][o] for o in OUTCOMES if o != 'win'),
                route_changed_games=sum(bool(p['route']['route_changes']) for p in pairs),
                route_submitted_changes=len(changes), route_changes_by_type=dict(Counter(r['unit_type'] for r in changes)),
                route_submission_results=dict(Counter(r['submission'] for r in changes)),
                paired_delta_route_minus_baseline=diffs,
                paired_dog_contact_position_distance=_stats(shifts))


METHODOLOGY = {
    'layout': 'Single-episode manifest spawn coordinates, not filename, RNG branch, or final battle position.',
    'contact': 'First sampled live allied Zergling with a visible enemy in its legal attack menu/mask. '
               'If simultaneous, use lowest numeric actor ID and list all ties. World coordinates are from that '
               'observation, not a planned waypoint or evidence of a hit.',
    'support': 'An allied live hydra supports if its legal mask and native range cover any visible live enemy '
               'within 3 world units (center distance) of any first-dog legal target, including those targets. '
               'A visible live enemy hydra covers if its native weapon range includes the first dog or any '
               'live ally within 3 of that dog. No enemy action mask or cooldown is inferred. Count each hydra once; '
               'fractions divide by all live allied / visible live enemy hydras respectively. Ratio divides the '
               'two support counts. Geometric opportunity, not shots; unseen enemies are unknown.',
    'participation': 'Report mask, submitted attack order and first allied dog CD restart separately. CD restart '
                     'is sampled CD > max(0, previous CD - 8) + 0.02, not an exact shot or hit. Fixed window counts '
                     'distinct allied hydras with a restart sampled in (first team visibility, visibility + 48] '
                     'divided by initial allied hydras. Requires the closing observation; no partial windows.',
    'deaths_damage': 'First hydra death side is the victim (replay tracker). Damage by source type summed across '
                     'games is verified against the replay. Before/through fractions refer to first allied dog death.',
    'audit': 'On the same observation loop compare same-policy pre/post route action IDs with the submitted '
             'step.actions vector. A submitted change is not proof of physical displacement or a hit.',
    'aggregation': 'Means weight games equally. Each mean reports non-null n and missing; paired deltas require '
                   'both versions. Gains/drops are paired win vs any non-win. Null is not zero.'}


def summarize(batch: Path, jobs: list):
    """Validate 30 matched scenarios, write batch/comparison.json, return the report."""
    batch = Path(batch)
    parts = {}
    for label, name, path in jobs:
        path = Path(path)
        _check(label in LABELS and name in COMPOSITIONS, path, 'unknown policy/composition')
        group, source, labels = parts.setdefault(path.stem, (name, path, set()))
        _check((group, source) == (name, path) and label not in labels, path, 'duplicate/conflicting shard')
        labels.add(label)
    _check(len(parts) == 30, batch, 'expected 30 one-game pairs')
    design, counts, seeds = [], Counter(), set()
    for stem, (name, path, labels) in sorted(parts.items()):
        _check(labels == set(LABELS), path, 'missing policy')
        m = _read(path)
        _check(m['version'] == 1 and len(m['episodes']) == 1 and m['sha256'] ==
               _digest({k: v for k, v in m.items() if k != 'sha256'}), path, 'invalid single-game manifest')
        c, = m['episodes']
        cfg, seed = c['episode_config'], c['sc2_seed']
        team = {u.lower(): n for u, n in COMPOSITIONS[name].items()}
        _check(c['sha256'] == _digest(cfg) and type(seed) is int and (name, seed) not in seeds and
               all(Counter(cfg['team_gen'][key]) == team for key in ('ally_team', 'enemy_team')),
               path, 'invalid scenario, repeated seed or composition mismatch')
        seeds.add((name, seed))
        layout = _layout(cfg, path)
        counts[name, layout] += 1
        design.append((stem, name, layout, m))
    _check(counts == {(n, l): size for n in COMPOSITIONS for l, size in QUOTA.items()}, batch,
           f'expected 10 scattered and 5 center pairs per composition, got {counts}')
    rows = []
    for stem, name, layout, m in design:
        games = {}
        for label in LABELS:
            folder = batch / label / stem
            g, = _run(folder, name, m)
            with (folder / 'events.jsonl').open('rb') as stream:
                _front_game(_triples(stream, 1, folder), g, label, folder)
            games[label] = g
        _check(games['baseline']['initial_state_sha256'] == games['route']['initial_state_sha256'], stem,
               'paired initial states differ')
        c, = m['episodes']
        rows.append(dict(name=name, shard=stem, layout=layout, episode=1, sc2_seed=c['sc2_seed'],
                         scenario_sha256=c['sha256'], manifest_sha256=m['sha256'], versions=games))
    def by_layout(group):
        return {l: _summary([r for r in group if r['layout'] == l]) for l in QUOTA}
    report = dict(episodes=rows, total=_summary(rows), by_layout=by_layout(rows), by_composition={},
                  methodology=METHODOLOGY)
    for n in COMPOSITIONS:
        group = [r for r in rows if r['name'] == n]
        report['by_composition'][n] = dict(**_summary(group), by_layout=by_layout(group))
    (batch / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report
