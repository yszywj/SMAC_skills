"""Offline four-way comparison for the ROUND-2 hydra switches. The unit of
pairing is a manifest episode (all four labels complete with the same
observed initial hash); the unit of mechanism follow-up is one changed
actor/decision. Adds round-2 specifics: first-divergence analysis against
baseline per scenario, A2 arrival / designated-mate-range / mate-fire
follow-up facts, and the B2 cannot_reissue category.
"""
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path


LABELS = ('baseline', 'supported_escape', 'guard_support_move', 'both')
ENABLED = dict(zip(LABELS, ('', 'A2', 'B2', 'A2B2')))
OUTCOMES = ('win', 'loss', 'draw', 'timeout')
TYPES = ('Hydralisk', 'Zergling', 'Baneling', 'other')


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _design(batch, design):
    labels = design['labels']
    if len(labels) != 4 or set(labels) != set(LABELS):
        raise ValueError('Expected baseline, supported_escape, guard_support_move and both')
    directory = batch / design['manifests']
    scenarios = {}
    for stem, expected in sorted(design['shards'].items()):
        manifest = _read(directory / f'{stem}.json')
        if not expected == manifest['sha256'] == _digest({k: v for k, v in manifest.items() if k != 'sha256'}):
            raise ValueError(f'{stem}: manifest checksum mismatch')
        for ep, item in enumerate(manifest['episodes'], 1):
            sha, cfg = item['sha256'], item['episode_config']
            if sha != _digest(cfg) or sha in scenarios:
                raise ValueError(f'{stem}/{ep}: invalid or duplicate scenario hash')
            units = Counter(cfg['team_gen']['ally_team'])
            lings = units['zergling']
            if (lings not in (3, 6) or units != Counter(zergling=lings, hydralisk=10-lings)
                    or units != Counter(cfg['team_gen']['enemy_team'])):
                raise ValueError(f'{stem}/{ep}: unexpected composition')
            a, b = (cfg[f'{s}_start_positions']['item'] for s in ('ally', 'enemy'))
            if not all(len(p) == 10 and all(len(xy) == 2 and all(map(math.isfinite, xy)) for xy in p)
                       for p in (a, b)):
                raise ValueError(f'{stem}/{ep}: invalid positions')
            layout = 'center' if all(p == [16, 16] for p in a) else 'scattered'
            if layout == 'scattered' and not all(math.isclose(x + ex, 32) and math.isclose(y, ey)
                                                 for (x, y), (ex, ey) in zip(a, b)):
                raise ValueError(f'{stem}/{ep}: unrecognized layout')
            scenarios[sha] = dict(shard=stem, episode=ep, scenario_sha256=sha, manifest_sha256=expected,
                                  sc2_seed=item['sc2_seed'], layout=layout,
                                  composition=f'{lings}ling_{10-lings}hydra')
    return scenarios


def _damage(path, scenario):
    data = _read(path)
    if not all(data[k] == scenario[k] for k in ('episode', 'scenario_sha256', 'manifest_sha256')):
        raise ValueError('engine damage identity mismatch')
    if data['ticks_per_damage'] != 4096:
        raise ValueError('invalid damage tick scale')
    sides = {}
    for side, player in (('ally', '1'), ('enemy', '2')):
        fields = data['players'][player]
        ticks, amounts = fields['ticks_by_source_type'], fields['damage_dealt_by_source_type']
        if (set(ticks) != set(TYPES) or set(amounts) != set(TYPES)
                or not all(type(ticks[u]) is int and ticks[u] >= 0 and amounts[u] == ticks[u]/4096 for u in TYPES)):
            raise ValueError('invalid engine damage channels')
        sides[side] = amounts
    return sides, data


def _replay_deaths(path, damage, info, frames):
    if __package__:
        from .contact_early_report import _deaths
    else:
        from contact_early_report import _deaths
    rosters = {p: damage['players'][p]['initial_roster_by_api_tag'] for p in ('1', '2')}
    deaths = _deaths(path, rosters, damage['begin_request_loop'], damage['terminal_loop'])
    if any(sum(t in deaths for t in rosters[p]) != info[k] for p, k in [('1', 'dead_allies'), ('2', 'dead_enemies')]):
        raise ValueError('replay deaths do not match episode result')
    hydras = {p: sorted((deaths[t], t) for t, u in roster.items() if u == 'Hydralisk' and t in deaths)
              for p, roster in rosters.items()}
    first = min((t for row in hydras.values() for t, _ in row), default=None)
    firsts = {p: [tag for t, tag in row if t == first] for p, row in hydras.items()}
    side = ('none' if first is None else 'both' if all(firsts.values()) else 'ally' if firsts['1'] else 'enemy')
    isolation = 'not_first_side' if side != 'ally' else 'unknown'
    last = max((t for t in frames if first is not None and first-8 <= t < first), default=None)
    if side == 'ally' and last is not None and len(firsts['1']) == 1:
        f = frames[last]
        actor = next((a for a, t in f['live'].items() if t == firsts['1'][0]), None)
        if actor is not None:
            isolation = 'isolated' if actor in f['isolated'] else ('supported' if actor in f['threatened'] else 'no_visible_threat')
    return deaths, dict(ally_hydra_deaths=len(hydras['1']), enemy_hydra_deaths=len(hydras['2']),
                        first_hydra_death_side=side, first_hydra_death_loop=first, prior_macro_isolation=isolation)


def _snapshot(state):
    units = state['allies']
    hydras = {a: u for a, u in units.items() if u['type'] == 'Hydralisk' and u['alive']}
    rel = state['relations']
    enemies = state['visible_enemies']
    lanes = {}
    for e, enemy in enemies.items():
        lanes[str(enemy['tag'])] = {}
        for a in hydras:
            pair = rel.get(a, {}).get(e, {})
            facts = (pair.get('smac_attack_available'), pair.get('in_weapon_range'))
            lanes[str(enemy['tag'])][a] = False if False in facts else True if facts == (True, True) else None
    covered = {e: {a for a, v in lanes[str(enemies[e]['tag'])].items() if v is True} for e in enemies}
    threatened = {a: [e for e in enemies if rel[a][e].get('in_enemy_weapon_range') is True] for a in hydras}
    threats = {e for es in threatened.values() for e in es}
    return dict(state_hash=_digest(state), lanes=lanes, live={a: str(u['tag']) for a, u in hydras.items()},
                 cooldown={a: u['weapon_cooldown_loops'] for a, u in hydras.items()},
                order_kind={a: ((u.get('orders') or [{}])[0].get('kind') or 'none') for a,u in hydras.items()},
                position={a: [round(u['position'][0], 3), round(u['position'][1], 3)] for a, u in hydras.items()},
                attack_target={a: next((o.get('target_tag') for o in u.get('orders', [])[:1]
                                        if o.get('kind') == 'attack' or o.get('ability_id') in (23, 3674)), None)
                               for a, u in hydras.items()},
                dead={a for a, u in units.items() if u['type'] == 'Hydralisk' and u['alive'] is False},
                # Last observed geometry before death, not killer attribution.
                isolated={a for a, es in threatened.items() if es and all(not (covered[e] - {a}) for e in es)},
                threatened={a for a in hydras if threatened[a]},
                visible={str(u['tag']) for u in enemies.values()},
                threats=len(threats), jointly_covered=sum(len(covered[e]) >= 2 for e in threats))


def _events(path):
    """One bounded-memory pass per shard. Keep macro commands, receipts and
    accepted micro submissions for the divergence analysis."""
    steps, snapshots = {}, defaultdict(dict)
    with path.open('rb') as stream:
        for line in stream:
            if not re.match(rb'^\s*\{\s*"event"\s*:\s*"(step|skill_observation)"', line):
                continue
            e = json.loads(line)
            if e['event'] == 'skill_observation':
                state = e['payload']['state']
                frames = snapshots[e['episode']]
                if state['game_loop'] in frames:
                    raise ValueError('duplicate macro observation')
                frames[state['game_loop']] = _snapshot(state)
                frames[state['game_loop']]['step'] = e['step']
                continue
            key = e['episode'], e['game_loop']
            if key in steps:
                raise ValueError(f'{path}: duplicate step {key}')
            steps[key] = e['actions'], e['feedback']
    return steps, snapshots


def _accepted(step):
    actions, feedback = step
    commands = {a: rec.get('command') for a, rec in feedback['actors'].items()
                if rec.get('submission') == 'accepted' and (rec.get('response') or {}).get('code') == 1}
    micro = Counter((m['submitted_loop'], m['actor_tag'], m['target_tag'])
                    for m in feedback['micro_submissions'] if m['sc2_response'] == [1])
    return commands, micro


def _divergence(group_steps, base_steps, group_frames, base_frames, applied, terminal):
    """First observable difference vs baseline in engine order (observed
    state, macro commands, accepted micro submissions), matched against the
    applied switch windows of this group."""
    if not group_steps or not base_steps:
        return dict(first_divergence=None, kind='none_or_unpaired',
                    matches_switch_window=False)
    for index, t in enumerate(sorted(set(group_steps) | set(base_steps))):
        if terminal is not None and t > terminal:
            break
        if t not in group_steps or t not in base_steps:
            return dict(first_divergence=t, kind='unpaired_window',
                        matches_switch_window=False)
        gf, bf = group_frames.get(t), base_frames.get(t)
        if not all(f and f.get('step') == index and f.get('state_hash') for f in (gf, bf)) or (
                index and t != previous + 8):
            return dict(first_divergence=t, kind='incomplete_prefix', matches_switch_window=False)
        previous = t
        if gf['state_hash'] != bf['state_hash']:
            return dict(first_divergence=t, kind='observed_state',
                        matches_switch_window=False)
        for actions, fb in (group_steps[t], base_steps[t]):
            if (fb.get('response_mapping') != 'exact' or not isinstance(fb.get('micro_submissions'), list)
                    or set(fb.get('actors', {})) != {f'u{i}' for i in range(len(actions))}
                    or any(rec.get('requested_action_id') != str(actions[int(actor[1:])]) or
                           (rec.get('submission') != 'dead_noop' and not (rec.get('command') and rec.get('response')))
                           for actor, rec in fb.get('actors', {}).items())
                    or any(not isinstance(m, dict) or not all(m.get(k) is not None for k in
                           ('submitted_loop', 'actor_tag', 'target_tag', 'sc2_response')) for m in fb['micro_submissions'])):
                return dict(first_divergence=t, kind='incomplete_prefix', matches_switch_window=False)
        ga, gm = _accepted(group_steps[t])
        ba, bm = _accepted(base_steps[t])
        diff = sorted(k for k in ga.keys() | ba.keys() if ga.get(k) != ba.get(k))
        if diff:
            rows = [r for r in applied if r['game_loop'] == t]
            requested = {f'u{i}' for i, (a,b) in enumerate(zip(group_steps[t][0], base_steps[t][0])) if a != b}
            matched = set(diff) == requested == {r['actor'] for r in rows} and all(
                base_steps[t][0][int(r['actor'][1:])] == r['original_action'] and r['actor'] in ba
                for r in rows)
            return dict(first_divergence=t, kind='macro_command', actors=diff,
                        matches_switch_window=matched)
        if group_steps[t][0] != base_steps[t][0]:
            return dict(first_divergence=t, kind='requested_actions', matches_switch_window=False)
        if gm != bm:
            return dict(first_divergence=min(x[0] for x in gm.keys() | bm.keys() if gm[x] != bm[x]),
                        kind='micro_command', matches_switch_window=False)
    return dict(first_divergence=None, kind='no_observed_divergence',
                matches_switch_window=False)


def _protection(frames, steps):
    if not frames or len(frames) != steps:
        return None
    ordered = sorted(frames.items())
    if any(b-a != 8 for (a, _), (b, _) in zip(ordered, ordered[1:])):
        return None
    first = 'none_observed_before_terminal'
    for (_, prev), (_, now) in zip(ordered, ordered[1:]):
        losses = prev['live'].keys() & now['dead']
        if losses:
            first = ('multiple_in_one_interval' if len(losses) != 1 else
                     'isolated' if losses <= prev['isolated'] else
                     'supported' if losses <= prev['threatened'] else 'no_visible_threat')
            break
    return dict(first_observed_ally_hydra_death_prior_macro_state=first,
                visible_hydra_threat_target_windows=sum(f['threats'] for f in frames.values()),
                jointly_covered_threat_target_windows=sum(f['jointly_covered'] for f in frames.values()))


def _receipt(steps, ep, row):
    step = steps.get((ep, row['game_loop']))
    if not step:
        return 'missing_receipt'
    actions, feedback = step
    actors = feedback['actors']
    actor, action = row['actor'], row['final_action']
    if not re.fullmatch(r'u\d+', actor) or int(actor[1:]) >= len(actions):
        raise ValueError('invalid actor slot')
    rec = actors.get(actor, {})
    if actions[int(actor[1:])] != action or rec.get('requested_action_id') != str(action):
        return 'not_submitted_as_selected'
    if rec.get('submission') != 'accepted':
        return rec.get('submission', 'missing_receipt')
    cmd = rec.get('command') or {}
    if row['switch'] == 'A2':
        matched = cmd.get('ability_id') == 16
    else:
        target = _target(row)
        matched = (cmd.get('ability_id') in (23, 3674) and target is not None
                   and str(cmd.get('target_tag')) == str(target))
    return 'accepted' if matched and (rec.get('response') or {}).get('code') == 1 else 'receipt_mismatch'


def _target(row):
    # The original B2 action is a move; its target_tag is often null.
    return row.get('target_tag') or (row.get('support_target_tag') if row.get('current_target') is not None and
                                      row.get('current_target') == row.get('support_target') else None)


def _actor_interval(row, before, after, terminal):
    actor = row['actor']
    if not after:
        return 'terminal_censored' if terminal is not None and terminal <= row['game_loop']+8 else 'missing'
    if actor in after['dead']:
        return 'actor_dead'
    if not before or any(f['live'].get(actor) != str(row['actor_tag']) for f in (before, after)):
        return 'missing'
    prev, curr = (f['cooldown'].get(actor) for f in (before, after))
    if prev is None or curr is None:
        return 'missing'
    if curr > max(0, prev-8)+.02:
        return ('restart_matching_end_order' if _target(row) and str(after['attack_target'].get(actor)) == str(_target(row))
                else 'restart_unverified_target')
    return 'no_restart_observed'


def _paired_b2(row, baseline, variant, terminals):
    """Two observed branches, not an estimate for any later decision."""
    t = row['game_loop']
    out = {}
    for label, frames, terminal in zip(('baseline', 'variant'), (baseline, variant), terminals):
        before, after = frames.get(t), frames.get(t+8)
        mates = row.get('supporting_for')
        lanes = {a: _lane(before, after, a, row.get('support_target_tag'), None, t+8) for a in mates or ()}
        actor_lane = _lane(before, after, row['actor'], row.get('support_target_tag'), None, t+8)
        if not after and terminal is not None and terminal <= t+8:
            lanes = dict.fromkeys(lanes, 'terminal_censored')
            actor_lane = 'terminal_censored'
        out[label] = dict(own_next_8=_actor_interval(row, before, after, terminal),
                          own_lane_on_support_target_next_8=actor_lane,
                          original_mate_lanes_next_8=lanes if mates is not None else None)
    return out


def _lane(before, after, mate, tag, deaths, loop):
    if not before or not after or not tag or mate not in before['live']:
        return 'missing'
    if deaths is not None and deaths.get(str(tag), math.inf) <= loop:
        return 'target_dead'
    if mate in after['dead']:
        return 'mate_dead'
    if after['live'].get(mate) != before['live'][mate]:
        return 'missing'
    if str(tag) not in after['visible']:
        return 'not_visible'
    fact = after['lanes'].get(str(tag), {}).get(mate)
    return 'covered' if fact is True else 'outside' if fact is False else 'missing'


def _followup(row, samples, terminal, track, frames, deaths=None):
    start, switch = row['game_loop'], row['switch']
    samples = sorted(samples, key=lambda r: r['game_loop'])
    if any(f['actor'] != row['actor'] or f['game_loop']-start not in (8, 16, 24) or
           (terminal is not None and f['game_loop'] > terminal) for f in samples):
        raise ValueError('invalid follow-up identity or horizon')
    if len({f['game_loop'] for f in samples}) != len(samples):
        raise ValueError('duplicate follow-up observation')
    # A missing policy follow-up is not a missing engine observation. Macro
    # frames can independently supply survival, position and cooldown facts.
    by_loop = {f['game_loop']: dict(f) for f in samples}
    for offset in (8, 16, 24):
        t, actor = start+offset, row['actor']
        frame = frames.get(t)
        if not frame or terminal is not None and t > terminal:
            continue
        if frame['live'].get(actor) == str(row['actor_tag']):
            f = by_loop.setdefault(t, dict(game_loop=t, actor=actor))
            f['alive'] = True
            pos = frame.get('position', {}).get(actor)
            if pos is not None and row.get('candidate_destination') is not None:
                f['arrived_at_predicted'] = math.dist(pos, row['candidate_destination']) <= .35
        elif actor in frame['dead']:
            by_loop[t] = dict(game_loop=t, actor=actor, alive=False, outcome='died_observed')
    samples = [by_loop[t] for t in sorted(by_loop)]
    # A restart can be hidden by ordinary countdown over an 8-loop window.
    # Compare to expected decay, not just to the preceding absolute cooldown.
    for i, f in enumerate(samples):
        t, actor = f['game_loop'], row['actor']
        prev, curr = frames.get(t-8), frames.get(t)
        if prev and curr and all(s['live'].get(actor) == str(row['actor_tag']) and
                                s['cooldown'].get(actor) is not None for s in (prev, curr)):
            samples[i] = dict(f, cooldown_restarted=curr['cooldown'][actor] > max(0, prev['cooldown'][actor]-8)+0.02)
    alive = [f['game_loop'] for f in samples if f.get('alive') is True]
    dead = [f['game_loop'] for f in samples if f.get('alive') is False and f.get('outcome') == 'died_observed']
    death = None if deaths is None else deaths.get(str(row['actor_tag']))
    if death is not None:
        if death <= start or any(loop >= death for loop in alive):
            raise ValueError('replay/switch survival mismatch')
        if death <= start+24:
            dead.append(death)
    track['accepted_interventions'] += 1
    for h in (16, 24):
        if deaths is not None and terminal is not None and terminal >= start+h and (death is None or death > start+h):
            status = 'alive'
        elif any(t >= start + h for t in alive):
            status = 'alive'
        elif any(t <= start + h for t in dead):
            status = 'dead'
        elif terminal is not None and terminal <= start + h:
            status = 'terminal_censored'
        else:
            status = 'missing_observation'
        track[f'survival_{h}:{status}'] += 1
    track['observed_dead_within_24'] += bool(dead)
    # A macro cooldown change is evidence about the actor, not proof of an
    # exact launch, its target, a hit, or an additional shot vs baseline.
    track['macro_cd_restart_proxy'] += any(f.get('cooldown_restarted') is True for f in samples)
    restart = next((f['game_loop'] for f in samples if f.get('cooldown_restarted') is True), None)
    if restart is not None:
        track[f'first_macro_restart_observed_after_{restart-start}_loops'] += 1
    else:
        status = ('dead' if dead else 'terminal_censored' if terminal is not None and terminal <= start+24
                  else 'no_restart_observed' if all(any(f['game_loop'] == start+t and
                      f.get('cooldown_restarted') is False for f in samples) for t in (8, 16, 24)) else 'missing_observation')
        track['first_macro_restart:' + status] += 1
    if switch == 'A2':
        track['route_estimate:' + (row.get('support_class') or 'missing')] += 1
        arrival = ('arrived' if any(f.get('alive') is True and f.get('arrived_at_predicted') is True for f in samples)
                   else 'arrival_unknown' if dead
                   else 'not_at_sampled_destinations' if {f['game_loop']-start for f in samples
                        if f.get('alive') is True and f.get('arrived_at_predicted') is False} == {8, 16, 24}
                   else 'terminal_censored' if terminal is not None and terminal <= start+24
                   else 'missing_observation')
        track['arrival:' + arrival] += 1
        for sample in samples:
            if (sample.get('alive') is True and row.get('close_pressure') is not None
                    and isinstance(sample.get('pressing_threats_now'), list)
                    and all((sample.get('threat_state') or {}).get(e, '').startswith(('visible', 'pressing'))
                            for e in row.get('current_threats', row['close_pressure']))):
                before, now = len(row['close_pressure']), len(sample['pressing_threats_now'])
                track['pressure_relieved_observations'] += now < before
                track['pressure_unchanged_observations'] += now >= before
        mate, threat = row.get('support_mate'), row.get('support_threat_tag')
        initial = frames.get(start)
        lanes, fires, arrival_lanes = [], [], []
        matching_restart = False
        for offset in (8, 16, 24):
            t = start + offset
            end = frames.get(t)
            if dead and min(dead) <= t:
                lane = 'actor_dead'
            elif terminal is not None and t >= terminal and not end:
                lane = 'terminal_censored'
            elif (not end or end['live'].get(row['actor']) != str(row['actor_tag']) or mate == row['actor'] or
                  (row.get('support_mate_tag') and (not initial or initial['live'].get(mate) != str(row['support_mate_tag'])))):
                lane = 'missing'
            else:
                lane = _lane(initial, end, mate, threat, deaths, t)
            track[f'designated_lane_{offset}:{lane}'] += 1
            lanes.append(lane)
            if any(f['game_loop'] == t and f.get('alive') is True and f.get('arrived_at_predicted') is True
                   for f in samples):
                arrival_lanes.append(lane)
            if lane == 'covered':
                prev = frames.get(t-8)
                if prev and prev['live'].get(mate) == end['live'][mate]:
                    old, new = prev['cooldown'].get(mate), end['cooldown'].get(mate)
                    if old is not None and new is not None:
                        fires.append(new > max(0, old-8)+.02)
                        matching_restart |= fires[-1] and str(end['attack_target'].get(mate)) == str(threat)
        track['threat_in_designated_mate_range'] += 'covered' in lanes
        track['at_landing_designated_coverage:' + ('covered' if 'covered' in arrival_lanes else
              'outside' if arrival_lanes and set(arrival_lanes) == {'outside'} else 'unknown')] += 1
        track['designated_mate_fire_proxy_while_covered:' + ('restart' if any(fires) else
              'no_restart_sampled' if fires else 'unobserved')] += 1
        if any(fires):
            track['designated_restart_end_order:' + ('matches_threat' if matching_restart else 'target_unverified')] += 1
        support_status = 'observed_in_designated_mate_range' if 'covered' in lanes else (
            'outside_all_samples' if lanes == ['outside']*3 else lanes[-1] if lanes[-1] != 'outside' else 'partial')
        track['threat_support_24:' + support_status] += 1
        track[f"estimate_vs_observation:{row.get('support_class') or 'missing'}:{support_status}"] += 1
    if switch == 'B2':
        # Restrict the short-term proxy to the very next decision interval.
        # Report later death independently so shot-then-death is not a success-only bin.
        follow = next((f for f in samples if f['game_loop'] == start + 8), None)
        target = _target(row)
        end_frame = frames.get(start+8, {})
        track['own_next_order:' + end_frame.get('order_kind', {}).get(row['actor'], 'unobserved')] += 1
        mates = row.get('supporting_for')
        track['support_mates_at_decision:' + ('missing' if mates is None else str(len(mates)))] += 1
        for mate in mates or ():
            lane = _lane(frames.get(start), frames.get(start+8), mate, row.get('support_target_tag'), deaths, start+8)
            if lane == 'missing' and terminal is not None and terminal <= start+8 and start+8 not in frames:
                lane = 'terminal_censored'
            track['original_mate_lane_next_8:' + lane] += 1
        if any(t <= start + 8 for t in dead):
            status = 'dead_by_next_sample_shot_unknown'
        elif not follow or follow.get('alive') is not True:
            status = _actor_interval(row, frames.get(start), frames.get(start+8), terminal)
        elif follow.get('cooldown_restarted') is True:
            end = frames.get(start+8, {})
            same = str(end.get('attack_target', {}).get(row['actor'])) == str(target)
            status = ('actor_cd_restart_matching_order' if target is not None and same else
                      'actor_cd_restart_target_unverified')
        elif deaths is not None and deaths.get(str(target), math.inf) <= start+8:
            status = 'target_dead_no_restart_observed'
        elif target and (start+8) in frames and str(target) not in frames[start+8]['visible']:
            status = 'target_not_visible_no_restart_observed'
        else:
            status = 'no_restart_observed' if follow.get('cooldown_restarted') is False else 'missing_cooldown'
        track[f'next_8_macro_proxy:{status}'] += 1
        if restart is not None and dead and any(t >= restart for t in dead):
            track['restart_then_died_within_24'] += 1


def _mechanism(path, label, record, steps, frames, terminal, deaths):
    ep = record['episode']
    if not path.exists() and not ENABLED[label] and 'hydra_supported' not in record:
        return {}, {}, []
    data = _read(path)
    rows = data['decisions']
    if data['episode'] != ep or not isinstance(rows, list):
        raise ValueError('switch artifact identity mismatch')
    summary = record.get('hydra_supported', {})
    if summary.get('decisions') != len(rows):
        raise ValueError('switch artifact row count mismatch')
    counts, focus = Counter(), defaultdict(Counter)
    trials, samples = {}, defaultdict(list)
    states = Counter()
    applied = []
    for row in rows:
        s = row['switch']
        if s not in ('A2', 'B2') or s not in ENABLED[label]:
            raise ValueError(f'unexpected switch {s} in {label}')
        if row.get('followup'):
            samples[s, row['actor_tag'], row['decision_loop']].append(row)
            continue
        result = row['result']
        counts[f'{s}:evaluated'] += 1
        states[f'{s}:{result}'] += 1
        counts[f'{s}:{result}'] += 1
        for field in ('conditions_met', 'selected_for_commit'):
            counts[f'{s}:{field}'] += row.get(field) is True
        counts[f'{s}:candidate_available'] += row.get('candidate_action') is not None
        if result != 'applied' and not row.get('changed'):
            continue
        if (result != 'applied' or row.get('changed') is not True or row.get('candidate_action') is None
                or row['final_action'] != row['candidate_action'] or row['original_action'] == row['final_action']):
            raise ValueError('inconsistent switch change')
        key = s, row['actor_tag'], row['game_loop']
        if key in trials:
            raise ValueError('duplicate switch decision')
        trials[key] = row
    if states != summary.get('states') or not samples.keys() <= trials.keys():
        raise ValueError('switch result counts or follow-up links mismatch')
    for key, row in trials.items():
        s = key[0]
        status = _receipt(steps, ep, row)
        if row['game_loop'] in frames and frames[row['game_loop']]['live'].get(row['actor']) != str(row['actor_tag']):
            status = 'receipt_actor_tag_mismatch'
        if status == 'accepted' and row['game_loop'] not in frames:
            status = 'actor_unverified'
        counts[f'{s}:receipt:{status}'] += 1
        if status == 'accepted':
            applied.append(row)
            counts[f'{s}:changed_commands'] += 1
            _followup(row, samples[key], data.get('terminal_loop', terminal), focus[s], frames, deaths)
    return dict(counts), {s: dict(v) for s, v in focus.items()}, applied


def _summary(rows, label):
    games = [r['versions'][label] for r in rows]
    outcomes = Counter(g['outcome'] for g in games)
    numerator = outcomes['win']
    flips, losses, transitions = Counter(total=0), Counter(total=0), Counter()
    for r, g in zip(rows, games):
        before, after = r['versions']['baseline']['outcome'], g['outcome']
        transitions[f'{before}->{after}'] += 1
        if (before == 'win') != (after == 'win'):
            change = flips if after == 'win' else losses
            change.update(('total', r['layout'], 'comp:' + r['composition']))
    # Keep the same four-way denominator for every damage total and delta.
    damage_rows = [r for r in rows if all(v['damage'] is not None for v in r['versions'].values())]
    def total(side, unit, source):
        return sum(r['versions'][source]['damage'][side][unit] for r in damage_rows)
    def delta(side, unit):
        return total(side, unit, label) - total(side, unit, 'baseline') if damage_rows else None
    decisions, focus = Counter(), defaultdict(Counter)
    divergences = Counter()
    paired_actors, paired_own, paired_mates, paired_support = 0, Counter(), Counter(), Counter()
    for game in games:
        decisions.update(game.get('decisions') or {})
        for switch, counter in game.get('focus', {}).items():
            focus[switch].update(counter)
        if label != 'baseline':
            divergence = (game.get('divergence') or {}).get('kind', 'not_computed')
            divergences['matched_switch_window' if game.get('divergence', {}).get('matches_switch_window')
                        else divergence] += 1
            for pair in game.get('divergence', {}).get('b2_next_8_by_actor', {}).values():
                baseline, variant = pair['baseline'], pair['variant']
                paired_actors += 1
                paired_own[baseline['own_next_8'] + '->' + variant['own_next_8']] += 1
                paired_support[baseline['own_lane_on_support_target_next_8'] + '->' +
                               variant['own_lane_on_support_target_next_8']] += 1
                a, b = (branch['original_mate_lanes_next_8'] for branch in (baseline, variant))
                if a is not None and b is not None:
                    for mate in a:
                        paired_mates[a[mate] + '->' + b[mate]] += 1
    protected = [r['versions'][label]['protection'] for r in rows
                 if all(v.get('protection') is not None for v in r['versions'].values())]
    threats = sum(p['visible_hydra_threat_target_windows'] for p in protected)
    jointly = sum(p['jointly_covered_threat_target_windows'] for p in protected)
    replay_rows = [r for r in rows if all(v.get('replay') is not None for v in r['versions'].values())]
    replays = [r['versions'][label]['replay'] for r in replay_rows]
    death_delta = sum(r['versions'][label]['replay']['ally_hydra_deaths'] -
                      r['versions']['baseline']['replay']['ally_hydra_deaths'] for r in replay_rows)
    isolated_first = sum(r['versions'][label]['replay']['prior_macro_isolation'] == 'isolated'
                         for r in replay_rows)
    return dict(games=len(games), outcomes={k: outcomes[k] for k in OUTCOMES},
                win_rate=numerator/len(games) if games else None,
                vs_baseline=dict(paired_games=len(rows), flipped_up_wins=dict(flips), lost_wins=dict(losses),
                                 transitions=dict(transitions), ally_zergling_damage_delta=delta('ally', 'Zergling'),
                                 enemy_hydralisk_damage_delta=delta('enemy', 'Hydralisk'),
                                 enemy_zergling_damage_delta=delta('enemy', 'Zergling'),
                                 ally_hydra_deaths_delta=death_delta if replay_rows else None),
                damage_paired_games=len(damage_rows),
                **{f'{s}_damage_by_source_type': {u: total(s, u, label) for u in TYPES} if damage_rows else None
                   for s in ('ally', 'enemy')},
                switch_telemetry_games=sum(g.get('decisions') is not None for g in games),
                games_with_accepted_switch={s: sum((g.get('decisions') or {}).get(f'{s}:changed_commands', 0) > 0
                                                   for g in games) for s in ('A2', 'B2')},
                 first_divergence=dict(Counter(divergences)),
                 b2_first_divergence_next_8=dict(actors=paired_actors, own_restart_transitions=dict(paired_own),
                        own_support_lane_transitions=dict(paired_support), original_mate_lane_transitions=dict(paired_mates)),
                 protection=dict(observed_games=len(protected), visible_hydra_threat_target_windows=threats if protected else None,
                                replay_paired_games=len(replays),
                                ally_hydra_deaths=sum(r['ally_hydra_deaths'] for r in replays) if replays else None,
                                isolated_first_ally_hydra_deaths=isolated_first if replays else None,
                                first_hydra_death_side=dict(Counter(r['first_hydra_death_side'] for r in replays)),
                                first_hydra_death_ally_isolation=dict(Counter(r['prior_macro_isolation'] for r in replays)),
                                 jointly_covered_threat_target_windows=jointly if protected else None,
                                joint_coverage_fraction=jointly/threats if threats else None,
                                first_observed_ally_hydra_death_prior_macro_state=dict(Counter(
                                    p['first_observed_ally_hydra_death_prior_macro_state'] for p in protected))),
                switch_decisions=dict(decisions), switch_focus={k: dict(v) for k, v in focus.items()})


def build_report(batch, design, results, missing):
    """Report incomplete observations, never fill them with losses or zero damage."""
    batch = Path(batch)
    scenarios = _design(batch, design)
    expected = {(label, stem) for label in LABELS for stem in design['shards']}
    if not set(results) <= expected:
        raise ValueError('Unexpected result shard')
    missing = dict(missing)
    for label, stem in sorted(expected - results.keys()):
        missing.setdefault(f'{label}/{stem}', 'missing results')
    indexed = defaultdict(dict)
    for (label, stem), records in results.items():
        if f'{label}/{stem}' in missing:
            continue
        for r in records:
            sha = r['scenario_sha256']
            if (sha not in scenarios or scenarios[sha]['shard'] != stem or label in indexed[sha]
                    or r['episode'] != scenarios[sha]['episode']):
                raise ValueError(f'{label}/{stem}: unexpected, duplicate or misnumbered episode {sha}')
            indexed[sha][label] = r

    paired, errors, excluded, issues = [], [], [], []
    load_events = lru_cache(maxsize=4)(_events)
    for sha, scenario in scenarios.items():
        records = indexed[sha]
        absent = [k for k in LABELS if k not in records]
        initials = {k: r.get('initial_state_sha256') for k, r in records.items()}
        reasons = []
        if absent:
            reasons.append('missing group')
        if any(not re.fullmatch(r'[a-f0-9]{64}', str(x)) for x in initials.values()) or len(set(initials.values())) > 1:
            reasons.append('initial state missing, invalid or different')
            errors.append(sha)
        if any(r.get('outcome') not in OUTCOMES for r in records.values()):
            reasons.append('missing or invalid outcome')
        if reasons:
            excluded.append(dict(**scenario, reasons=reasons, missing_labels=absent, initial_states=initials))
            continue
        games, events, applied_by_label, terminals = {}, {}, {}, {}
        for label, r in records.items():
            game = games[label] = dict(outcome=r['outcome'])
            directory = batch / label / scenario['shard']
            filename = f"episode-{scenario['episode']:04d}.json"
            terminal, deaths = None, None
            try:
                steps, all_frames = load_events(directory / 'events.jsonl')
                frames = all_frames.get(scenario['episode'], {})
                game['protection'] = _protection(frames, r.get('steps'))
                events[label] = ({loop: value for (ep, loop), value in steps.items() if ep == r['episode']}, frames)
                ep_steps = events[label][0]
                if ep_steps and len(ep_steps) == r.get('steps'):
                    terminal = ep_steps[max(ep_steps)][1].get('to_game_loop')
            except (OSError, ValueError, KeyError, TypeError) as exc:
                steps, frames, game['protection'] = {}, {}, None
                issues.append(dict(label=label, scenario_sha256=sha, artifact='events', reason=str(exc)))
            for name in ('damage_by_type', 'hydra_supported'):
                path = directory / name / filename
                try:
                    if name == 'damage_by_type':
                        game['damage'], damage = _damage(path, scenario)
                        terminal = damage.get('terminal_loop', terminal)
                        replay = directory / 'replays' / filename.replace('.json', '.SC2Replay')
                        if not replay.exists() and 'replay' in r:
                            issues.append(dict(label=label, scenario_sha256=sha, artifact='replay', reason='missing replay'))
                        if replay.exists():
                            try:
                                deaths, game['replay'] = _replay_deaths(replay, damage, r['info'], frames)
                            except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
                                issues.append(dict(label=label, scenario_sha256=sha, artifact='replay', reason=str(exc)))
                    else:
                        counts, focus, applied = _mechanism(
                            path, label, r, steps, frames, terminal, deaths)
                        game['decisions'], game['focus'] = counts, focus
                        if applied:
                            applied_by_label[label] = applied
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    game['damage' if name == 'damage_by_type' else 'decisions'] = None
                    issues.append(dict(label=label, scenario_sha256=sha, artifact=name, reason=str(exc)))
            terminals[label] = terminal
        base_events = events.get('baseline', ({}, {}))
        for label in LABELS:
            if label == 'baseline':
                continue
            group_steps, group_frames = events.get(label, ({}, {}))
            divergence = games[label]['divergence'] = _divergence(
                group_steps, base_events[0], group_frames, base_events[1],
                applied_by_label.get(label, []), None)
            if divergence['matches_switch_window']:
                divergence['b2_next_8_by_actor'] = {r['actor']: _paired_b2(r, base_events[1], group_frames,
                    (terminals.get('baseline'), terminals.get(label))) for r in applied_by_label[label]
                    if r['switch'] == 'B2' and r['game_loop'] == divergence['first_divergence']}
        del events
        paired.append(dict(**scenario, initial_state_sha256=initials['baseline'], versions=games))

    groups = {}
    for label in LABELS:
        group = groups[label] = _summary(paired, label)
        for key in ('composition', 'layout'):
            group['by_' + key] = {v: _summary([r for r in paired if r[key] == v], label)
                                  for v in sorted({s[key] for s in scenarios.values()})}
        group['by_composition_layout'] = {c: {l: _summary([r for r in paired if r['composition'] == c and r['layout'] == l], label)
                                               for l in group['by_layout']} for c in group['by_composition']}
    for group in groups.values():
        center = group['by_layout'].get('center', {}).get('win_rate')
        group['vs_baseline']['center_win_rate_delta'] = (None if center is None else
            center - groups['baseline']['by_layout']['center']['win_rate'])
    report = dict(labels=list(LABELS), expected_scenarios=len(scenarios), paired_scenarios=len(paired),
                  missing_shards=missing, pairing_initial_state_errors=errors, excluded_scenarios=excluded,
                  telemetry_issues=issues, episodes=paired, groups=groups, definitions=dict(
                      cohort='One game per label per scenario, all four completed with the same observed initial hash.',
                      damage='Both engine players, opposing-player source-type damage; only complete four-way damage pairs.',
                      changed_commands='Selected action differs and the macro step and exact SC2 acceptance receipt agree.',
                       divergence='Within one paired episode, require consecutive observations from step 0 with identical full logged state through the first differing macro command, matching accepted original and substitute actions. B2 next-8 compares branches only there; simultaneous switches are a joint intervention. A shared seed alone or a later decision is not a counterfactual.',
                       mate_lane='Observed legal mask AND native-range geometry on the original support target, not a hit. Losing a lane after a guard is descriptive, not proof the guard caused it. B2 original_mate_lane_next_8 counts mates, while own restart counts actors.',
                      followup='Unit: accepted intervention, not observation. Horizons are 8 loops (next shot) and 16/24 (survival). Arrival uses the predicted destination with a 0.35 world-unit tolerance; death before the next observation is arrival_unknown, never "did not walk".',
                      firing='An observed weapon cooldown restart is a firing proxy, never proof of a particular target, hit, or extra shot vs baseline. Commands and SC2 acceptances do not count as shots. Macro times are interval endpoints.',
                      support_classes="A2 'now' means the named mate's reach already covers the landing point; 'if_pursued' coverage forms only if the enemy advances to engage - a conditional estimate, never realized support.",
                      cannot_reissue='B2 rows whose preserved target is no longer on the command mask are recorded, never fabricated into attacks; STOP is never counted as a kept attack.',
                      protection='At each macro snapshot count distinct visible threats to a live allied hydra. Joint coverage requires two live allied hydras both in range with a legal attack on that threat. Range is not a hit.',
                      isolation='For first exact replay hydra death, use the preceding macro snapshot. Isolated requires a visible threat and no other allied hydra able to return fire on any such threat.',
                      replay='Deaths use roster-verified replay tracker up to terminal, never target disappearance; replay comparisons require four valid replays. Unknown or censored is not zero.'))
    (batch / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report
