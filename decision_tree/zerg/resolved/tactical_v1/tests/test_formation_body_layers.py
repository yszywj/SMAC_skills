"""Physical congestion uses body clearance; splash avoidance remains separate."""
import copy
import json
from pathlib import Path
import unittest

from skills.tactical_v1.context import Context, build_snapshot, distance
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.local_formation import LocalFormationSkill
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import enemy, scene, with_questions


def context(raw):
    return Context(build_snapshot(with_questions(raw), raw['game_loop']))


class FormationBodyLayerTests(unittest.TestCase):
    def plans(self, c, config=None):
        return LocalFormationSkill(config or {}).propose(c, TacticalMemory(), Telemetry())

    def test_air_above_ground_is_not_a_physical_pair(self):
        c = context(scene([((0., 0.), 'marine'), ((.1, 0.), 'medivac')], distance=2.))
        skill = LocalFormationSkill({})
        self.assertFalse(self.plans(c))
        self.assertIsNone(skill._opposite_move(c, 'u0', 'u1'))
        legacy = {'common_combat': {'joint_assessment_enabled': False}}
        self.assertTrue(self.plans(c, legacy))

    def test_small_bodies_keep_legal_contact_without_false_spacing(self):
        raw = scene([((0., 0.), 'zergling'), ((.9, 0.), 'zergling')],
                    {'e0': enemy((1.5, 0.))})
        for unit in raw['allies'].values():
            unit['radius'] = .375
        c = context(raw)
        self.assertTrue(all(c.attackable_enemies(a) for a in c.snapshot.allies))
        self.assertFalse(self.plans(c))
        self.assertTrue(self.plans(c, {'common_combat': {'joint_assessment_enabled': False}}))

    def test_large_bodies_need_more_than_legacy_fixed_clearance(self):
        raw = scene([((0., 0.), 'marauder'), ((1.3, 0.), 'marauder')])
        for unit in raw['allies'].values():
            unit['radius'] = .75
        c = context(raw)
        self.assertTrue(self.plans(c))
        self.assertFalse(self.plans(c, {'common_combat': {'joint_assessment_enabled': False}}))

    def test_real_blocking_pair_uses_same_predicate_for_all_paths(self):
        raw = scene([((0., 0.), 'marine'), ((.65, 0.), 'marine'),
                     ((0., .1), 'medivac')])
        for unit in raw['allies'].values():
            unit['radius'] = .375
        c, skill = context(raw), LocalFormationSkill({})
        positions = {a: u.position for a, u in c.snapshot.allies.items()}
        self.assertEqual(skill._crowding(c, 'u0', positions), 1)
        self.assertEqual(skill._crowding_at(c, 'u0', (0., 0.), positions), 1)
        plans = self.plans(c)
        pairs = [p for p in plans if p.atomic]
        self.assertTrue(pairs)
        for plan in pairs:
            self.assertEqual(set(plan.controlled_actors), {'u0', 'u1'})
            positions_after = {a: c.predict_position(a, plan.current_actions[a]).position
                               for a in plan.controlled_actors}
            self.assertGreater(distance(positions_after['u0'], positions_after['u1']), .8)
        move = skill._opposite_move(c, 'u0', 'u1')
        self.assertIsNotNone(move)
        self.assertLess(skill._crowding_at(c, 'u0', c.predict_position('u0', move).position,
                                          positions), 1)

    def test_same_air_layer_can_decongest_without_ground_partner(self):
        raw = scene([((0., 0.), 'marine'), ((.5, 0.), 'marine'), ((0., .1), 'marine')])
        raw['allies']['u0']['is_flying'] = True
        raw['allies']['u1']['is_flying'] = True
        c = context(raw)
        plans = self.plans(c)
        self.assertTrue(any(p.atomic for p in plans))
        self.assertTrue(all('u2' not in p.controlled_actors for p in plans))

    def test_splash_escape_remains_even_without_physical_neighbors(self):
        raw = scene([((0., 0.), 'marine'), ((.1, 0.), 'medivac')],
                    {'e0': enemy((3.4, 0.), 'colossus')})
        c = context(raw)
        plans = self.plans(c)
        self.assertTrue(plans)
        self.assertTrue(all(p.controlled_actors == ('u0',) for p in plans))
        self.assertTrue(all('splash' in p.rationale for p in plans))
        self.assertTrue(all(distance(c.predict_position('u0', p.current_actions['u0']).position,
                                    c.snapshot.enemies['e0'].position) > 3.5 for p in plans))

    def test_contact_margin_is_shared_with_melee_slots(self):
        raw = scene([((0., 0.), 'marine'), ((.85, 0.), 'marine')])
        for unit in raw['allies'].values():
            unit['radius'] = .375
        c = context(raw)
        self.assertFalse(self.plans(c))
        self.assertTrue(self.plans(c, {'coordinated_engage': {'contact_slot_margin': .15}}))

    def test_eleven_recorded_air_only_neighbors_do_not_trigger_ground_spacing(self):
        batch = Path(__file__).resolve().parents[3] / 'runs/tactical_10x3x3_20261115_212459'
        audit_path = batch / 'analysis/common_formation_audit.json'
        if not audit_path.exists():
            self.skipTest('optional recorded formation observations unavailable')
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        checked = 0
        for group, result in audit['groups'].items():
            samples = result['samples']
            if not samples:
                continue
            wanted = {(s['seed'] - 169, s['step']): s for s in samples}
            with (batch / 'current' / group / 'events.jsonl').open(encoding='utf-8') as stream:
                for raw_line in stream:
                    if '"event": "skill_observation"' not in raw_line:
                        continue
                    row = json.loads(raw_line)
                    key = (row['episode'], row['step'])
                    if key not in wanted:
                        continue
                    sample = wanted.pop(key)
                    with self.subTest(group=group, seed=sample['seed'], step=sample['step']):
                        payload = row['payload']
                        c = Context(build_snapshot(payload, payload['state']['game_loop']))
                        actor = sample['actor']
                        self.assertFalse(c.snapshot.allies[actor].is_flying)
                        self.assertTrue(sample['near_air'])
                        self.assertFalse(sample['near_ground'])
                        self.assertFalse(any(cap.splash_attack for cap in c.enemy_capabilities.values()))
                        memory, skill = TacticalMemory(), LocalFormationSkill({})
                        before = copy.deepcopy(memory.__dict__)
                        plans = skill.propose(c, memory, Telemetry())
                        self.assertFalse(any(actor in p.controlled_actors for p in plans))
                        self.assertEqual(before, memory.__dict__)
                        legacy = self.plans(c, {'common_combat': {'joint_assessment_enabled': False}})
                        self.assertTrue(any(actor in p.controlled_actors for p in legacy))
                    checked += 1
                    if not wanted:
                        break
            self.assertFalse(wanted)
        self.assertEqual(checked, 11)


if __name__ == '__main__':
    unittest.main()
