"""Offline audit of five local-pack policies on the same 150 frozen scenarios."""
import argparse
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

from compare_protoss_parallel import digest
from contact_early_report import _check
import contact_local_packs_report as packs
from contact_micro_report import _read


MODE_MAP = {'packs_hydra': 'micro8-hydra', 'packs_all': 'micro8-all', 'marginal_all': 'micro8-all',
            'marginal_far': 'micro8-all', 'marginal_near': 'micro8-all'}
LABELS = tuple(MODE_MAP)
RESERVE = {'marginal_far': 4, 'marginal_near': 2}
CONTRASTS = (*zip(LABELS, LABELS[1:]), ('marginal_all', 'marginal_near'))
COUNTS = ('authorizations', 'submitted', 'accepted', 'rejected', 'result_unmapped')
STATES = {'already_ordered', 'submitted', 'actor_dead', 'target_lost', 'never_legal'}


def _micro_window(obs, decision, step, episode, rosters, mode, by_type, where):
    window = step['micro_window_closed']
    rows = window['records']
    state = obs['payload']['state']
    loop = state['game_loop']
    _check(window['episode'] == episode and window['macro_step'] == obs['step'] and
           window['issued_loop'] == loop and window['closed_loop'] == loop + 8 and window['interior_checks'] == 7,
           where, 'incomplete or displaced micro window')
    _check(isinstance(rows, list) and len(rows) == window['authorizations'] and
           dict(Counter(r['status'] for r in rows)) == window['states'], where, 'wrong micro window counts')
    submitted, seen = [], set()
    for r in rows:
        actor, tag = r['actor'], r['actor_tag']
        u = state['allies'].get(actor, {})
        target = state['visible_enemies'].get(r['target'], {})
        species = rosters['1'].get(tag)
        _check(species is not None and tag not in seen and u.get('tag') == tag and u['alive'] and
               species == u['type'] and (mode == 'micro8-all' or species == 'Hydralisk') and
               ('actor_type' not in r or r['actor_type'] == species.lower()) and
               target.get('tag') == r['target_tag'] and r['target_tag'] in rosters['2'] and target['alive'] and
               rosters['2'][r['target_tag']] == target['type'], where, 'invalid micro actor/target identity or type')
        seen.add(tag)
        status, cmd = r['status'], r['macro_action_id']
        _check(r['episode'] == episode and r['macro_step'] == obs['step'] and r['issued_loop'] == loop and
               r['expires_loop'] == loop + 7 and r['kind'] == 'approach_then_attack' and
               type(cmd) is int and cmd == decision['actions'][actor] and
               (2 <= cmd <= 5 or (cmd >= 6 and r['target'] == f'e{cmd-6}')) and status in STATES,
               where, 'invalid micro authorization or macro action')
        n = r['checked_loops']
        legal = status in ('submitted', 'already_ordered')
        _check(type(n) is int and 1 <= n <= 7 and (status != 'never_legal' or n == 7) and
               r['first_legal_loop'] == (loop + n if legal else None) and
               r['submitted_loop'] == (loop + n if status == 'submitted' else None),
               where, 'invalid micro trigger state/clock')
        c = by_type[species]
        c['authorizations'] += 1
        if status == 'submitted':
            submitted.append(r)
            response = r.get('sc2_response')
            _check(isinstance(response, list) and all(type(k) is int for k in response), where, 'invalid SC2 response')
            if 'submitted_action_id' in r:  # New recorder includes the actual interior enemy slot.
                action = r['submitted_action_id']
                _check(type(action) is int and 6 <= action < 6 + len(rosters['2']), where, 'invalid micro attack id')
            # Exactly one raw Attack command; absent/ambiguous results are not acceptance.
            c['submitted'] += 1
            c['result_unmapped' if len(response) != 1 else 'accepted' if response == [1] else 'rejected'] += 1
        else:
            _check('sc2_response' not in r and r.get('submitted_action_id') is None,
                   where, 'receipt without a micro submission')
    _check(step['feedback']['micro_submissions'] == submitted, where, 'wrong submitted micro receipts')
    return rows


def _audit_triples(triples, ep, game, rosters, where, *, work, mode):
    expected = _read(work / 'micro_triggers' / f'episode-{ep:04d}.json')['authorizations']
    by_type = {u: dict.fromkeys(COUNTS, 0) for u in sorted(set(rosters['1'].values()))}
    offset = 0
    for obs, decision, step in triples:
        location = f"{where}: step {obs['step']}"
        rows = _micro_window(obs, decision, step, ep, rosters, mode, by_type, location)
        _check(rows == expected[offset:offset+len(rows)], location, 'window differs from episode authorizations')
        offset += len(rows)
        yield obs, decision, step
    _check(offset == len(expected), where, 'extra episode authorizations')
    game['micro_trigger'].update(mode=mode, by_type=by_type, accepted=sum(c['accepted'] for c in by_type.values()),
                                 records=expected)
    game['initial_roster_by_api_tag'] = rosters


def _games(work, manifest, name, label, files, mode):
    """Single-shard entry point for retries. mode is read by the caller from design.mode_map."""
    _check(label in MODE_MAP and mode == MODE_MAP[label], work, 'invalid label/execution mode')
    audit = partial(_audit_triples, work=work, mode=mode)
    # Local trace shape is independent of the micro execution mode; none is a no-pack baseline.
    trace_label = 'packs_reserve' if label in RESERVE else 'packs'
    games, contract = packs._games(work, manifest, name, trace_label, files, mode=mode, audit_triples=audit)
    triggers = [g['micro_trigger'] for g in games]
    states = Counter()
    for t in triggers:
        states.update(t['states'])
    totals = dict(mode=mode, macro_loops=8, engine_step_loops=1, states=dict(states),
                  windows=sum(t['windows'] for t in triggers), authorizations=sum(states.values()))
    _check(_read(work/'summary.json')['micro_trigger_totals'] == totals, work, 'wrong shard micro totals')
    return games, contract


def _summary(rows):
    versions = packs._versions(rows, LABELS)
    for label, version in versions.items():
        games = [r['versions'][label]['micro_trigger'] for r in rows]
        states = Counter()
        by_type = defaultdict(Counter)
        for micro in games:
            states.update(micro['states'])
            for species, counts in micro['by_type'].items():
                by_type[species].update(counts)
        version['micro_trigger'].update(by_type={u: dict(c) for u, c in by_type.items()}, states=dict(states),
                                        accepted=sum(g['accepted'] for g in games))
    contrasts = {}
    for left, right in CONTRASTS:
        pair = packs._paired(rows, left, right)
        a, b = (versions[label]['micro_trigger']['by_type'] for label in (left, right))
        pair['micro_delta_by_type'] = {u: {k: b[u][k] - a[u][k] for k in COUNTS} for u in a}
        a, b = (versions[label]['local_coord']['counts'] for label in (left, right))
        pair['local_counts_delta'] = {k: b.get(k, 0) - a.get(k, 0) for k in sorted(a.keys() | b.keys())}
        contrasts[right + '_minus_' + left] = pair
    return dict(pairs=len(rows), versions=versions, incremental=contrasts)


def _by(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return groups


def summarize(batch: Path):
    """Validate all 750 games before replacing comparison.json; return the same report."""
    batch = Path(batch)
    design = _read(batch / 'design.json')
    modes = design.get('mode_map')
    _check(modes == MODE_MAP, batch, 'unexpected design.mode_map')

    def load(work, manifest, name, label, files):
        return _games(work, manifest, name, label, files, modes[label])

    rows = packs._matched_rows(batch, design, LABELS, load)
    methodology = {k: v for k, v in packs.METHODOLOGY.items() if k not in ('reserve', 'geometry', 'layout')}
    methodology.update(
        micro='One authorization per registered actor tag/window, joined to the initial allied replay roster. '
              'Submitted = an interior Attack command actually sent; accepted = exactly one SC2 success code [1]. '
              'Neither is a shot or a hit. There is no per-order hit count. Actual damage is independently read '
              'from the replay for both sides, by source type (including other), not allocated to orders.',
        contrasts='Matched right minus left. Win gains/drops compare win to any non-win. First vary execution on '
                  'old packs, then local allocation without dispatch, then enable 4- or 2-beat reserve, and '
                  'compare 2 vs 4. Selection/follow-up counts describe observed behavior, not causal hit attribution.',
        layout='Per-episode frozen spawn coordinates, not the shard name or observed combat geometry.',
        geometry='Multiple fronts and incoming pincers are visible contact geometry at a macro observation, '
                 'not spawn layout or evidence of successful attacks.')
    report = dict(seed=design['seed'], labels=list(LABELS), mode_map=modes, reserve_horizon_beats=RESERVE.copy(),
                  policy_sha256={k: digest(v) for k, v in design['policy_files'].items()}, methodology=methodology,
                  episodes=rows, total=_summary(rows), by_composition={},
                  by_layout={k: _summary(g) for k, g in _by(rows, 'layout').items()})
    for name, group in _by(rows, 'composition').items():
        report['by_composition'][name] = dict(**_summary(group), by_layout={
            k: _summary(g) for k, g in _by(group, 'layout').items()})
    packs._write_report(batch, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    summarize(parser.parse_args().batch)
