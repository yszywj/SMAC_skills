"""Read-only diagnosis of A in a paired hydra batch. Print JSON to stdout.

python diagnose_hydra_escape.py BATCH [--details] (mpyq and s2protocol required)
"""
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

if __package__:
    from .contact_early_report import _deaths
    from .hydra_switch_report import LABELS, _read, _receipt
else:
    from contact_early_report import _deaths
    from hydra_switch_report import LABELS, _read, _receipt


H = 8
ARRIVAL = 0.35  # world units, relative to the projected one-window endpoint


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def load_windows(directory, episodes):
    frames, steps = defaultdict(dict), defaultdict(dict)
    with (directory / 'events.jsonl').open('rb') as stream:
        for line in stream:
            if not re.match(rb'^\s*\{\s*"event"\s*:\s*"(step|skill_observation)"', line):
                continue
            e = json.loads(line)
            ep = e['episode']
            if ep not in episodes:
                continue
            if e['event'] == 'step':
                t, dest = e['game_loop'], steps[ep]
                value = e['actions'], e['feedback']
                require(value[1]['from_game_loop'] == t, 'step receipt clock mismatch')
            else:
                value = e['payload']['state']
                t, dest = value['game_loop'], frames[ep]
                require(value['step_mul'] == H and value['episode'] == ep, 'unexpected observation clock')
            require(t not in dest, 'duplicate step or observation')
            dest[t] = value
    return frames, steps


def replay(directory, item, frames, steps):
    stem = 'episode-%04d' % item['episode']
    damage = _read(directory / 'damage_by_type' / (stem + '.json'))
    score = _read(directory / 'score_details' / (stem + '.json'))
    require(all(item[k] == score[k] == damage[k] for k in ('scenario_sha256', 'manifest_sha256', 'episode'))
            and score['sc2_seed'] == item['sc2_seed'], 'seed/scenario artifact mismatch')
    start, end = damage['begin_request_loop'], damage['terminal_loop']
    require(frames.keys() == steps.keys() == set(range(start, end, H)), 'incomplete macro stream')
    rosters = {p: v['initial_roster_by_api_tag'] for p, v in damage['players'].items()}
    require({u['tag']: u['type'] for u in frames[start]['allies'].values()} == rosters['1'], 'wrong roster')
    deaths = _deaths(directory / 'replays' / (stem + '.SC2Replay'), rosters, start, end)
    return deaths, end


def accepted_commands(step):
    return {a: rec['command'] for a, rec in step[1]['actors'].items()
            if rec['submission'] == 'accepted' and (rec.get('response') or {}).get('code') == 1}


def prefix(escape, baseline, end):
    """First observable difference, in engine order (snapshot, macro, micro)."""
    af, ast = escape
    bf, bst = baseline
    for t in sorted(set(af) | set(bf)):
        if t > end:
            break
        if t not in af or t not in bf:
            return t, 'unpaired_window', []
        # Previous orders are compared as actually accepted, below. Do not
        # compare switch labels, planner decisions or unused authorizations.
        if any(af[t][k] != bf[t][k] for k in ('allies', 'visible_enemies', 'relations', 'movement_candidates')):
            return t, 'observed_state', []
        a, b = accepted_commands(ast[t]), accepted_commands(bst[t])
        diff = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
        if diff:
            return t, 'macro_command', diff
        def micro(st):
            return {(m['submitted_loop'], m['actor_tag'], m['target_tag'])
                    for m in st[t][1]['micro_submissions'] if m['sc2_response'] == [1]}
        delta = micro(ast) ^ micro(bst)
        if delta:
            return min(x[0] for x in delta), 'micro_command', []
    return None, 'no_observed_divergence', []


def status(deaths, terminal, tag, loop):
    if deaths.get(tag, math.inf) <= loop:
        return 'dead'
    return 'alive' if loop <= terminal else 'censored'


def radius(state, enemy, actor):
    profile = state['unit_type_profiles'][str(enemy['unit_type_id'])]
    ranges = [w['range'] for w in profile['weapons'] if w['target_type'] in ('ground', 'any')]
    require(bool(ranges), 'missing native ground reach')
    return max(ranges) + enemy['radius'] + actor['radius']


def previous_beat(state, prev, actor):
    if state is None or prev is None:
        return {'gate': 'missing_previous_beat', 'move': 'unknown'}
    unit = state['allies'][actor]
    require(unit['alive'] and unit['tag'] == prev['actor_tag'], 'prior actor mismatch')
    origin = unit['position']
    enemies = state['visible_enemies']
    reach = {e: radius(state, u, unit) for e, u in enemies.items()}
    def threats(p):
        return {e for e, u in enemies.items() if math.dist(u['position'], p) <= reach[e]}
    now = threats(origin)
    max_dist = H * state['unit_type_profiles'][str(unit['unit_type_id'])]['movement_speed_native'] / 16
    menu = state['movement_candidates'][actor]
    options = []
    for move, candidate in menu.items():
        dest = candidate['destination']
        length = math.dist(origin, dest)
        if not length:
            continue
        point = [o + (d-o)*min(1, max_dist/length) for o, d in zip(origin, dest)]
        mid = [(o+p)/2 for o, p in zip(origin, point)]
        after = threats(point)
        if after < now and threats(mid) <= now:
            options.append(move)
    return dict(gate=prev['result'], veto=prev.get('veto'), recorded_candidate=prev['candidate_action'],
                hp=unit['health'], legal_moves=len(menu), static_shed_moves=options,
                move=('recorded_candidate' if prev['candidate_action'] is not None else
                      'legal_static_relief' if options else 'no_static_relief' if menu else 'no_legal_move'))


def support(state, actor, ids):
    return {e for e in ids if any(a != actor and u['type'] == 'Hydralisk' and u['alive']
                and state['relations'][a][e]['smac_attack_available'] is True
                and state['relations'][a][e]['in_weapon_range'] is True for a, u in state['allies'].items())}


def diagnose_move(r, fs, frames, deaths, terminal):
    t, actor, tag = r['game_loop'], r['actor'], r['actor_tag']
    state = frames[t]
    unit = state['allies'][actor]
    dest = r['candidate_destination']
    tags = r['threat_tags']
    require(set(tags) == set(r['current_threats']) and all(state['visible_enemies'][e]['tag'] == tag0
            for e, tag0 in tags.items()), 'decision threat identity mismatch')
    observed = support(state, actor, tags)
    worth = r['worthwhile_move']
    projection = worth['pursuer_covered_by']
    basis = ('current_threat_in_mate_range' if observed else 'conditional_pursuit' if worth['pursuer_covered_by']
             else 'estimated_shed_only' if worth['shed'] else 'unexplained')
    first_arrival, first_support = None, None
    samples = []
    for dt in (8, 16, 24):
        loop = t+dt
        f, s = fs.get(loop), frames.get(loop)
        life = status(deaths, terminal, tag, loop)
        sample = dict(dt=dt, survival=life, followup_present=f is not None)
        samples.append(sample)
        if f is not None:
            require(f['actor'] == actor and f['submitted_action'] == r['final_action'] and f['threat_tags'] == tags
                    and (f['alive'] is None or f['alive'] == (life == 'alive')), 'followup mismatch')
        if not s or life != 'alive':
            continue
        u = s['allies'][actor]
        require(u['tag'] == tag and u['alive'], 'actor identity/survival mismatch')
        visible = {e for e, target in tags.items() if s['visible_enemies'].get(e, {}).get('tag') == target}
        covered = support(s, actor, visible)
        if f and f['alive'] is True:
            require(f['threat_state'] == {e: ('not_visible' if e not in visible else 'in_ally_range' if e in covered
                    else 'not_in_ally_range') for e in tags}, 'followup range mismatch')
        gap = math.dist(u['position'], dest)
        if gap <= ARRIVAL and first_arrival is None:
            first_arrival = dt
        if covered and first_support is None:
            first_support = dt
        pressing = {e for e in visible if s['relations'][actor][e]['in_enemy_weapon_range']}
        new_pressure = {e for e in s['visible_enemies'] if e not in visible and s['relations'][actor][e]['in_enemy_weapon_range']}
        predicted_pair_covered = None
        if projection:
            mate, e = projection['mate'], projection['threat']
            rel = s['relations'].get(mate, {}).get(e, {})
            predicted_pair_covered = (e in visible and s['allies'][mate]['alive'] and
                                      rel.get('smac_attack_available') is True and rel.get('in_weapon_range') is True)
        # Has the original threat left weapon reach of the *fixed projected point*?
        # Only count targets originally covering that point. Separate death and fog.
        left = []
        for e in visible:
            before, now = state['visible_enemies'][e], s['visible_enemies'][e]
            reach = radius(state, before, unit)
            if math.dist(before['position'], dest) <= reach < math.dist(now['position'], dest):
                left.append(e)
        sample.update(position=u['position'], hp=u['health'], endpoint_gap=round(gap, 4), arrived=gap <= ARRIVAL,
                      observed_mate_coverage=sorted(covered), enemy_left_endpoint=sorted(left),
                      predicted_pair_covered=predicted_pair_covered, originally_covered_now_visible_outside=sorted((observed & visible)-covered),
                      original_threats_now_pressing=sorted(pressing), new_visible_threats=sorted(new_pressure),
                      net_visible_pressure_reduced=len(pressing)+len(new_pressure) < len(tags),
                      original_threat_dead=sorted(e for e in tags if deaths.get(tags[e], math.inf) <= loop),
                      original_threat_unseen=sorted(e for e in tags if e not in visible and deaths.get(tags[e], math.inf) > loop))
    next_sample = samples[0]
    arrival = ('arrived' if first_arrival else 'not_at_sampled_endpoint' if any('arrived' in s for s in samples)
               else 'dead_before_next_sample' if next_sample['survival'] == 'dead' else 'no_live_position')
    return dict(origin=unit['position'], hp=unit['health'], endpoint=dest, basis=basis, current_mate_coverage=sorted(observed),
                gate_window_fire_support=any(r['ally_support'].values()),
                projected_pursuit=worth['pursuer_covered_by'], projected_shed=worth['shed'], samples=samples,
                first_arrival=first_arrival, first_support=first_support, arrival=arrival)


def diagnose(batch, details=False):
    batch = Path(batch)
    report = _read(batch / 'comparison.json')
    cohort = report['episodes']
    require(len(cohort) == report['paired_scenarios'] == len({r['scenario_sha256'] for r in cohort})
            and all(set(r['versions']) == set(LABELS) for r in cohort), 'invalid four-way cohort')
    shards = defaultdict(dict)
    receipts, rows, divergence = Counter(), [], Counter()
    for item in cohort:
        ep = item['episode']
        shard = shards[item['shard']]
        require(ep not in shard, 'duplicate shard episode')
        shard[ep] = item
    for stem, items in shards.items():
        dirs = [batch / label / stem for label in ('escape_only', 'baseline')]
        trials = {}
        for ep, item in items.items():
            data = _read(dirs[0] / 'hydra_switches' / f'episode-{ep:04d}.json')
            require(data['episode'] == ep, 'switch episode mismatch')
            decisions, follow = {}, defaultdict(dict)
            for r in data['decisions']:
                require(r['switch'] == 'A', 'unexpected switch in escape_only')
                t, tag = r['game_loop'], r['actor_tag']
                if r.get('followup'):
                    dest = follow[r['decision_loop'], tag]
                    require(t not in dest and t-r['decision_loop'] in (8, 16, 24), 'invalid followup')
                    dest[t] = r
                else:
                    require((t, tag) not in decisions, 'duplicate decision')
                    decisions[t, tag] = r
            changed = [r for r in decisions.values() if r['result'] == 'applied']
            require(all(r['changed'] == (r['result'] == 'applied') for r in decisions.values())
                    and set(follow) <= {(r['game_loop'], r['actor_tag']) for r in changed}, 'invalid applied/followup links')
            expected = item['versions']['escape_only']['decisions']
            require(expected is not None and len(changed) == expected.get('A:applied', 0), 'comparison application mismatch')
            if changed:
                trials[ep] = decisions, follow, changed
        if not trials:
            continue
        games = [load_windows(d, trials) for d in dirs]
        for ep, (decisions, follow, changed) in trials.items():
            item = items[ep]
            esc, base = [(f[ep], s[ep]) for f, s in games]
            (deaths, terminal), (bd, bt) = [replay(d, item, *g) for d, g in zip(dirs, (esc, base))]
            frames, steps = esc
            when, kind, actors = prefix(esc, base, max(r['game_loop'] for r in changed))
            divergence[kind] += 1
            accepted = 0
            for r in changed:
                t, a, tag = r['game_loop'], r['actor'], r['actor_tag']
                unit = frames[t]['allies'][a]
                require(unit['alive'] and unit['type'] == 'Hydralisk' and unit['tag'] == tag
                        and r['original_kind'] == 'attack' and r['original_action'] != r['final_action']
                        and r['candidate_action'] == r['final_action'], 'invalid A switch identity/action')
                actions, feedback = steps[t]
                rc = _receipt({(ep, t): (actions, feedback['actors'])}, ep, r)
                if rc == 'accepted':
                    target = frames[t]['movement_candidates'][a][str(r['final_action'])]['destination']
                    cmd = feedback['actors'][a]['command']
                    if cmd['kind'] != 'move' or math.dist(cmd['target_position'], target) > .002:
                        rc = 'move_destination_mismatch'
                receipts[rc] += 1
                if rc != 'accepted':
                    continue
                accepted += 1
                row = dict(seed=item['sc2_seed'], scenario=item['scenario_sha256'], shard=stem, episode=ep,
                           loop=t, actor=a, tag=tag, command_destination=target, divergence=[when, kind, actors],
                           prior=previous_beat(frames.get(t-H), decisions.get((t-H, tag)), a),
                           death_after=None if tag not in deaths else deaths[tag]-t,
                           **diagnose_move(r, follow[t, tag], frames, deaths, terminal))
                if when == t and kind == 'macro_command' and actors == [a]:
                    cmd = accepted_commands(base[1][t]).get(a, {})
                    if cmd.get('ability_id') in (23, 3674) and cmd.get('target_tag') == r['target_tag']:
                        row['baseline_survival'] = {s['dt']: status(bd, bt, tag, t+s['dt']) for s in row['samples']}
                        row['baseline_death_after'] = None if tag not in bd else bd[tag]-t
                        row['baseline_positions'] = {dt:base[0][t+dt]['allies'][a]['position'] for dt, life in
                                                     row['baseline_survival'].items() if life == 'alive' and t+dt in base[0]}
                rows.append(row)
            require(accepted == item['versions']['escape_only']['decisions'].get('A:changed_commands', 0),
                    'accepted count differs from comparison')

    def counts(seq):
        return dict(sorted(Counter(seq).items()))
    immediate = [r for r in rows if r['samples'][0]['survival'] == 'dead']
    pairs = [r for r in rows if 'baseline_survival' in r]
    def timing(rs):
        return dict(n=len(rs), previous_gate=counts(r['prior']['gate'] for r in rs),
                    previous_move=counts(r['prior']['move'] for r in rs),
                    previous_gate_by_move=counts(r['prior']['gate']+' / '+r['prior']['move'] for r in rs))
    def next_status(s):
        return ('arrived' if s['arrived'] else 'not_at_endpoint') if 'arrived' in s else s['survival']+'_no_position'
    def support_outcome(r):
        if any(s.get('observed_mate_coverage') for s in r['samples']):
            return 'in_mate_range'
        if any('arrived' in s for s in r['samples']):
            return 'no_mate_range_observed'
        return 'no_live_position'
    result = dict(paired_games=len(cohort), expected_games=report['expected_scenarios'], applied_receipts=dict(receipts),
                  games_with_accepted=len({r['scenario'] for r in rows}), accepted=len(rows),
                  timing=dict(all=timing(rows), dead_by_8=timing(immediate),
                              survival={dt: counts(r['samples'][i]['survival'] for r in rows) for i, dt in enumerate((8,16,24))}),
                  landing=dict(basis=counts(r['basis'] for r in rows), first_observed_arrival=counts(str(r['first_arrival']) for r in rows),
                               gate_window_fire_support=sum(r['gate_window_fire_support'] for r in rows),
                               policy_justification=counts(('shed_and_pursuit' if r['projected_shed'] else 'pursuit_only')
                                                           if r['projected_pursuit'] else 'shed_only' for r in rows),
                               projected_pair_observed=sum(any(s.get('predicted_pair_covered') for s in r['samples']) for r in rows),
                               basis_by_followup=counts(r['basis']+' / '+support_outcome(r) for r in rows),
                               next_beat_among_live=counts(('reduced' if s['net_visible_pressure_reduced'] else 'not_reduced')+
                                   (' / covered' if s['observed_mate_coverage'] else ' / outside_mate_range')
                                   for r in rows for s in r['samples'][:1] if 'arrived' in s),
                               next_beat=counts(next_status(r['samples'][0]) for r in rows), by_24=counts(r['arrival'] for r in rows),
                               actual_support_by_arrival=counts(r['arrival']+' / '+('covered' if r['first_support'] else 'not_observed') for r in rows),
                               any_live_position=sum(any('arrived' in s for s in r['samples']) for r in rows),
                               any_enemy_left_fixed_endpoint=sum(any(s.get('enemy_left_endpoint') for s in r['samples']) for r in rows),
                               initially_covered_target_now_visible_outside=sum(any(s.get('originally_covered_now_visible_outside') for s in r['samples']) for r in rows),
                               any_original_enemy_dead_while_actor_observed=sum(any(s.get('original_threat_dead') for s in r['samples']) for r in rows),
                               any_original_enemy_unseen_alive=sum(any(s.get('original_threat_unseen') for s in r['samples']) for r in rows),
                               any_original_threat_in_mate_range=sum(r['first_support'] is not None for r in rows),
                               followup_samples=sum(s['followup_present'] for r in rows for s in r['samples'])),
                  prefix=dict(first_difference_by_game=dict(divergence), matched=len(pairs), excluded_interventions=len(rows)-len(pairs),
                              survival_baseline_to_escape={dt:counts(r['baseline_survival'][dt]+'->'+r['samples'][i]['survival'] for r in pairs)
                                                           for i, dt in enumerate((8,16,24))}),
                  limits=['One row per accepted A actor/decision; repeated decisions may involve the same actor and death.',
                          'Previous legal static relief: masked move, native speed * 8/16, frozen visible enemies, no new midpoint/end threat; not path, collision, reservation, pursuit or policy validation.',
                          'Support is another living hydra with a legal in-range attack on an original visible threat; not a shot. Estimated shed and conditional pursuit are not observations.',
                          'Current coverage uses observed threat positions, independent of actor endpoint. Gate window fire support additionally requires the policy window-shot estimate. Reduced pressure compares counts of visible enemies in native reach at T/T+8; not damage prevented.',
                          f'Arrival means sampled within {ARRIVAL} world units of the predicted one-window endpoint, not the raw command target. Later orders may intervene; no continuous paths.',
                          'Enemy departure means visible on both samples and crossing out of native reach of the fixed projected endpoint; not death, fog, or proof of being driven off.',
                          'Death is roster-verified replay tracker through terminal. Missing live positions and terminal censoring are not failures.',
                          'Pairs require same four-way scenario/seed, observed macro state and accepted team macro/micro command prefix, then only the same actor differs (original accepted target vs Move). Not equality of hidden engine state or proof of causality.'])
    if details:
        result['interventions'] = rows
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('batch', type=Path)
    p.add_argument('--details', action='store_true', help='also emit each decision, previous gates and follow-up positions')
    args = p.parse_args()
    print(json.dumps(diagnose(args.batch, args.details), indent=2, allow_nan=False))
