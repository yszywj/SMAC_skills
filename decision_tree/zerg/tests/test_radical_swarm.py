"""No SC2 dependencies. Run this module with --archive for the full R2 audit."""
import argparse
from collections import Counter
import json
from pathlib import Path
import unittest

from decision_tree.zerg.radical_swarm import RadicalSwarm


PANELS = Path(__file__).resolve().parents[3] / 'runs/zerg_route_training_r2'


def scene(types, enemies=(), positions=()):
    s = dict(allies={}, movement_candidates={}, visible_enemies={}, search_location=[16, 16])
    masks = []
    for i in range(10):
        alive = i < len(types)
        p = positions[i] if i < len(positions) else (16, 16)
        s['allies'][f'u{i}'] = dict(alive=alive, type=types[i] if alive else 'zergling',
                                     position=p, weapon_ready=True, weapon_cooldown_loops=0, health=30)
        s['movement_candidates'][f'u{i}'] = {str(a): dict(destination=[p[0]+x, p[1]+y]) for a, (x, y)
                                             in enumerate([(0, 2), (0, -2), (2, 0), (-2, 0)], 2)}
        masks.append(([0, 1, 1, 1, 1, 1] if alive else [1, 0, 0, 0, 0, 0]) + [0]*10)
    for j, (kind, hp, pos) in enumerate(enemies):
        s['visible_enemies'][f'e{j}'] = dict(alive=True, type=kind, health=hp, max_health=90, position=pos)
    return {'state': s}, masks


def audit_episode(events):
    controller = RadicalSwarm()
    counts = Counter()
    reasons = Counter()
    lineup = None
    stops = no_contact = passed_targets = 0
    pending = None
    with events.open() as stream:
        for line in stream:
            if '"skill_observation"' not in line[:120] and '"skill_step"' not in line[:120]:
                continue
            event = json.loads(line)
            if event['event'] == 'skill_step':
                assert pending is not None and event['step'] == pending[0]
                old = [event['actions'][f'u{i}'] for i in range(10)]
                counts['different_beats'] += pending[1] != old
                pending = None
                continue
            assert pending is None and event['step'] == counts['beats']
            raw = event['payload']
            s = raw['state']
            u = [s['allies'][f'u{i}'] for i in range(10)]
            if lineup is None:
                lineup = tuple(sum(x['type'].lower() == t for x in u) for t in ('zergling', 'baneling', 'hydralisk'))
            # Archives have the public legal criteria, not env handles. Dead
            # actors have no question and only their SMAC no-op is legal.
            masks = [[int(str(a) in raw['questions'][f'u{i}']['criteria']) for a in range(16)]
                     if x['alive'] else [1] + [0]*15 for i, x in enumerate(u)]
            d = controller.choose(raw, masks)
            actions = d['actions']
            assert len(actions) == len(d['roles']) == len(d['reasons']) == 10
            for i, a in enumerate(actions):
                assert type(a) is int and masks[i][a] and (a == 0) == (not u[i]['alive'])
                if a >= 6:
                    assert f'e{a - 6}' in s['visible_enemies']
                if u[i]['alive'] and d['roles'][i] == 'gunline' and u[i]['weapon_ready'] and any(masks[i][6:]):
                    assert a >= 6
                    counts['ready_gun_shots'] += 1
            pending = event['step'], actions
            counts['beats'] += 1
            counts['phase_' + d['phase']] += 1
            reasons.update(d['reasons'])
            stops = stops + 1 if all(a <= 1 for a in actions) else 0
            counts['all_stop_beats'] += int(stops > 0)
            counts['max_stop_run'] = max(stops, counts['max_stop_run'])
            no_contact = no_contact + 1 if s['visible_enemies'] and max(actions) < 6 else 0
            counts['max_visible_without_attack_run'] = max(no_contact, counts['max_visible_without_attack_run'])
            passed_targets = passed_targets + 1 if no_contact and any(any(row[6:]) for row in masks) else 0
            counts['max_legal_target_without_attack_run'] = max(passed_targets, counts['max_legal_target_without_attack_run'])
            counts['legal_target_without_attack_beats'] += int(passed_targets > 0)
    assert pending is None and counts['beats']
    return lineup, counts, reasons


class TestSwarm(unittest.TestCase):
    def test_constrained_shooters_first_and_overkill_spill(self):
        raw, masks = scene(['Hydralisk']*3, [('zergling', 11, (17, 16)), ('hydralisk', 75, (18, 16))])
        masks[0][6:8] = masks[2][6:8] = [1, 1]
        masks[1][6] = 1
        self.assertEqual(RadicalSwarm().choose(raw, masks)['actions'], [7, 6, 7] + [0]*7)

    def test_burst_splash_and_no_duplicate_spend(self):
        raw, masks = scene(['baneling', 'baneling', 'hydralisk'], [('zergling', 24, (17, 16)),
                                                                    ('zergling', 17, (17, 17))])
        for row in masks[:3]:
            row[6:8] = [1, 1]
        d = RadicalSwarm().choose(raw, masks)
        self.assertEqual(d['reasons'], ['burst_commit', 'burst_conserve', 'volley'] + ['dead']*7)
        self.assertLess(d['actions'][1], 6)

    def test_cooling_gun_improves_distance_from_enemy_bane(self):
        raw, masks = scene(['hydralisk', 'zergling'], [('baneling', 30, (17.5, 16))])
        raw['state']['allies']['u0'].update(weapon_ready=False, weapon_cooldown_loops=10)
        masks[0][6] = masks[1][6] = 1
        d = RadicalSwarm().choose(raw, masks)
        self.assertEqual(d['actions'][:2], [5, 6])
        self.assertEqual(d['reasons'][:2], ['cooldown_evade', 'intercept'])

    def test_fog_mask_and_hidden_metadata(self):
        raw, masks = scene(['zergling']*10, positions=[(1, 4), (31, 28)]*5)
        for row in masks:
            row[14] = 1  # Even if a mask leaks a slot, do not target it.
        first = RadicalSwarm().choose(raw, masks)
        raw['state'].update(seed=12345678, outcome='win', enemies={'e8': dict(position=[1, 4])})
        self.assertEqual(first, RadicalSwarm().choose(raw, masks))
        ctrl = RadicalSwarm()
        ds = [ctrl.choose(raw, masks) for _ in range(25)]
        self.assertEqual([d['phase'] for d in ds[:5]], ['muster']*4 + ['sweep'])
        self.assertTrue(all(max(d['actions']) < 6 for d in ds))
        self.assertGreater(len({tuple(d['actions']) for d in ds[5:]}), 1)

    def test_contact_stall_and_lost_sighting_expire(self):
        raw, masks = scene(['hydralisk']*10, [('hydralisk', 90, (24, 16))])
        ctrl = RadicalSwarm()
        ds = [ctrl.choose(raw, masks) for _ in range(8)]
        self.assertEqual(ds[-1]['phase'], 'break_stall')
        self.assertEqual(ds[-1]['actions'], [4]*10)
        raw['state']['visible_enemies'].clear()
        phases = [ctrl.choose(raw, masks)['phase'] for _ in range(4)]
        self.assertEqual(phases, ['reacquire']*3 + ['sweep'])

    def test_wide_visible_contact_regroups_without_holding_ready_fire(self):
        raw, masks = scene(['hydralisk']*3 + ['zergling'], [('hydralisk', 90, (21, 16))],
                           positions=[(17, 16), (9, 16), (16, 16), (12, 16)])
        masks[0][6] = 1
        ctrl = RadicalSwarm()
        d = ctrl.choose(raw, masks)
        self.assertEqual(d['phase'], 'link_up')
        self.assertEqual(d['actions'][:3], [6, 4, 5])
        for _ in range(4):
            d = ctrl.choose(raw, masks)
        self.assertNotEqual(d['phase'], 'link_up')

    def test_bad_mask_fails_closed(self):
        raw, masks = scene(['hydralisk'])
        masks[0][0] = 1
        with self.assertRaises(ValueError):
            RadicalSwarm().choose(raw, masks)
        with self.assertRaises(ValueError):
            RadicalSwarm().choose(raw, [[1, 2, 3]]*10)

    @unittest.skipUnless(PANELS.is_dir(), 'R2 archive not present')
    def test_common_severe_archived_public_trajectories(self):
        for panel, pair, arm in [('fresh', 'pair-019', 'v2'), ('regression', 'pair-011', 'alternative')]:
            _, counts, _ = audit_episode(PANELS / panel / pair / arm / 'runner/events.jsonl')
            self.assertGreater(counts['ready_gun_shots'], 0)
            self.assertGreater(counts['different_beats'], 1)
            self.assertLess(counts['max_stop_run'], 4)


def full_audit():
    totals, reason_counts, lineups = Counter(), Counter(), Counter()
    for path in sorted(PANELS.glob('*/pair-*/*/runner/events.jsonl')):
        lineup, counts, reasons = audit_episode(path)
        for k, v in counts.items():
            if k.startswith('max_'):
                totals[k] = max(totals[k], v)
            else:
                totals[k] += v
        totals['episodes'] += 1
        totals['different_episodes'] += int(counts['different_beats'] > 0)
        reason_counts.update(reasons)
        lineups[str(lineup)] += 1
    assert totals['episodes'] == 168
    print(json.dumps(dict(counts=totals, reasons=reason_counts, lineups=lineups), sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', action='store_true')
    if parser.parse_args().archive:
        full_audit()
    else:
        unittest.main(argv=['unittest'])
