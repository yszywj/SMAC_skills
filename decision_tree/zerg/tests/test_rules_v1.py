"""Offline dual-lane tests. No engine, labels, held-out episodes or resolver writes."""
import copy
import hashlib
import importlib
import json
import pickle
import unittest
from pathlib import Path
from unittest.mock import patch

from decision_tree.zerg import rules_v1 as rule
from decision_tree.zerg.factory import load_resolved_package
from skills.tactical_v1.tests import fixtures as fx

ROOT = Path(__file__).resolve().parents[3]
PKG = load_resolved_package(ROOT / 'decision_tree/zerg/resolved/tactical_v1').__name__


def native(name):
    return importlib.import_module(f'{PKG}.{name}')


def fixture():
    # Baneling's east route cuts through u2's observed body; zergling's south
    # route collides with u3 moving east. Both alternatives close on e0. u4
    # attacks and u5 is an unowned, dead baseline row.
    units = [((0, 0), 'baneling'), ((3, 2), 'zergling'), ((1.4, 0), 'zergling'),
             ((2, 1), 'zergling'), ((9, -2), 'zergling'), ((0, 8), 'zergling')]
    state = fx.scene(units, {'e0': fx.enemy((10, -2))}, attack_range=1.5)
    state['allies']['u5'].update(alive=False, health=0.)
    for i, unit in enumerate(state['allies'].values()):
        unit.update(tag=str(101 + i), radius=.375)
    payload = fx.with_questions(state)
    m = native('skill_manager').SkillManager()
    _, s = m.step(payload, len(units))
    snap, _, _, risk, ctx = m._pending_commit
    contracts, common = native('contracts'), native('skills.common')

    def plan(i, cmd):
        actor = f'u{i}'
        skill = ('burst_engage' if i == 0 else 'focus_fire' if i == 4 else 'coordinated_engage')
        return common.make_plan(skill, [actor], 'e0', {actor: cmd}, [],
            contracts.ResourceClaims(controls=(actor,), space=(common.space_claim(ctx, actor, cmd),)),
            contracts.Commitment('attacking' if i == 4 else 'approaching'), {},
            snap.snapshot_id, m.config['_fingerprint'], goal_kind='enemy')

    # Controlled planner fixture: two genuine scored singleton substitutions
    # retained separately by the old coordinator, never their combination.
    base_plans = [plan(i, cmd) for i, cmd in enumerate((4, 3, 4, 4, 6))]
    m.coordinator._cache.clear()
    entries = []
    for changed in ({}, {0: plan(0, 3)}, {1: plan(1, 4)}):
        plans = [changed.get(i, p) for i, p in enumerate(base_plans)]
        actions = native('baseline').complete_plan(ctx, m.memory, m.skills, plans)
        entry = m.coordinator.evaluate_once(ctx, plans, actions, m.memory, [1], len(units))
        assert entry is not None
        entries.append(entry)
    base = entries[0]
    vector = [base.actions[f'u{i}'] for i in range(len(units))]
    s.update(actions=dict(base.actions), chosen_plans=[{'id': p.proposal_id} for p in base_plans],
             cache_entries=len(m.coordinator._cache), local_coord={'selected': []}, top_alternates=[])
    m._cache = (m._cache[0], vector[:], copy.deepcopy(s), base_plans, snap)
    m._pending_commit = (snap, base_plans, vector[:], risk, ctx)
    return m, s, payload


class DualLane(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = fixture()

    def setUp(self):
        self.m, self.s, self.obs = copy.deepcopy(self.f)

    def entry(self):
        return rule.make_entry(self.m, self.s, self.obs)

    def test_complete_native_plan_determinism_and_isolation(self):
        m = self.m
        before = pickle.dumps((m, self.s, self.obs))
        e = self.entry()
        self.assertIsInstance(e, native('coordinator')._CompletePlan)
        self.assertEqual(e.actions, dict(u0=3, u1=4, u2=4, u3=4, u4=6, u5=0))
        ids = {p.proposal_id for p in e.plans}
        self.assertEqual(len(ids), 5)
        self.assertTrue(ids <= {p.proposal_id for old in m.coordinator._cache.values() for p in old.plans})
        self.assertTrue({p.proposal_id for p in m._pending_commit[1][2:]} <= ids)
        self.assertEqual(pickle.dumps(e), pickle.dumps(self.entry()))
        e.plans[0].current_actions.clear()
        e.execution.clear()
        self.assertEqual(before, pickle.dumps((m, self.s, self.obs)))

    def test_predicate_uses_observable_masks_only(self):
        obs, a = self.obs, self.s['actions']
        self.assertTrue(rule.may_trigger(obs, [a[f'u{i}'] for i in range(6)]))
        self.assertTrue(rule.may_trigger({**obs, 'future_return': -999}, a))
        self.assertFalse(rule.may_trigger(obs, {**a, 'u4': 4}))  # no attack
        obs['questions']['u0']['criteria']['6'] = 'legal'
        self.assertFalse(rule.may_trigger(obs, a))             # bomber can attack
        self.assertFalse(rule.may_trigger(obs, {**a, 'u0': 6}))
        self.assertFalse(rule.may_trigger(obs, {'u0': 4}))
        self.assertFalse(rule.may_trigger({}, a))

    def test_global_dedup_not_topk_or_plan_ids(self):
        e = self.entry()
        self.assertIsNotNone(e)
        self.assertEqual(self.s['top_alternates'], [])
        e.plans = []  # different semantics, same complete vector
        self.m.coordinator._cache['not-a-top-alternate'] = e
        self.m._cache[2]['cache_entries'] = len(self.m.coordinator._cache)
        with patch.object(native('coordinator').Coordinator, 'evaluate_once', side_effect=AssertionError):
            self.assertIsNone(self.entry())

    def test_incomplete_or_unauthorized_evidence_abstains(self):
        m = self.m
        for attr, value in [('snapshot_id', 100), ('chosen_plans', []), ('local_coord', {'selected': ['x']})]:
            with self.subTest(attr=attr), patch.dict(self.s, {attr: value}):
                self.assertIsNone(self.entry())
        with patch.dict(self.obs['state'], {'game_loop': 999}):
            self.assertIsNone(self.entry())
        with patch.dict(m._pending_commit[4].formation_orders, {'u0': 4}):
            self.assertIsNone(self.entry())
        with patch.dict(m._pending_commit[0].legal, {'u0': frozenset((1, 4))}):
            self.assertIsNone(self.entry())
        with patch.object(m._pending_commit[1][2], 'primary_skill', 'unregistered'):
            self.assertIsNone(self.entry())
        with patch.object(m.memory, 'committed', True):
            self.assertIsNone(self.entry())

    def test_native_completion_and_validation_may_veto(self):
        with patch.object(native('baseline'), 'complete_plan', return_value={}):
            self.assertIsNone(self.entry())
        with patch.object(native('coordinator'), 'validate_plans', return_value=['conflict']):
            self.assertIsNone(self.entry())

    def test_no_donor_plan_no_synthetic_fill(self):
        cache = self.m.coordinator._cache
        for k, e in list(cache.items()):
            if e.actions['u1'] == 4:
                del cache[k]
        self.m._cache[2]['cache_entries'] = len(cache)
        self.assertIsNone(self.entry())

    def test_source_binding(self):
        self.assertEqual(rule.rule_hash(), hashlib.sha256(Path(rule.__file__).read_bytes()).hexdigest())


# Additional real-prefix regression if the optional original training archive
# is installed. Never search experiment groups or read an episode's outcome.
ARCHIVE = (ROOT / 'runs/zerg_shadow_ptpy1yzu/workers/zerg_9zergling_1baneling_0hydralisk'
           / 'episode-036-seed-20262358/runner/events.jsonl')


@unittest.skipUnless(ARCHIVE.is_file(), 'original training anchor 023 is not installed')
class TrainPrefix(unittest.TestCase):
    def test_actual_native_replay(self):
        m = native('skill_manager').SkillManager()
        m.micro_all_attackers = True
        with ARCHIVE.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['event'] == 'skill_observation':
                    obs = row['payload']
                    actions, s = m.step(obs, 10)
                    if s['snapshot_id'] == 7:
                        break
                    m.commit_final_actions(actions)
                if row['event'] == 'step':
                    for sub in row['feedback'].get('micro_submissions', []):
                        if sub.get('sc2_response') == [1]:
                            m.record_micro_action(sub['actor'], sub['submitted_action_id'])
            else:
                self.fail('missing pre-action snapshot')
        self.assertEqual((s['snapshot_game_loop'], actions), (50, [4, 3, 3, 4, 15, 15, 4, 4, 4, 4]))
        before = pickle.dumps(m)
        entry = rule.make_entry(m, s, obs)
        self.assertEqual(entry.actions, {**s['actions'], 'u0': 3, 'u1': 4})
        self.assertEqual(len(entry.plans), 8)
        self.assertEqual(set(p.proposal_id for p in entry.plans) - set(p.proposal_id for p in m._pending_commit[1]),
                         {'burst_engage-f096fa83e8', 'coordinated_engage-075db80562-slot:e9:6:4'})
        self.assertNotIn(entry.actions, [e.actions for e in m.coordinator._cache.values()])
        self.assertEqual(pickle.dumps(m), before)


if __name__ == '__main__':
    unittest.main()
