"""Replay frozen contact_response inputs through private managers; no SC2 imports."""
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from contact_response_micro_variant import FROZEN, apply_variant
from contact_simple_units import apply_simple


def frames(n):
    log = FROZEN.parents[1] / 'zerg_3zergling_0baneling_7hydralisk_0/events.jsonl'
    with log.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event['event'] == 'skill_observation':
                payload = event['payload']
            elif event['event'] == 'step':
                yield payload, event['actions']
                n -= 1
                if not n:
                    return


@contextmanager
def private(lings, hydras):
    with tempfile.TemporaryDirectory() as tmp:
        skills = Path(tmp) / 'skills'
        shutil.copytree(FROZEN.parent, skills, ignore=shutil.ignore_patterns('__pycache__', 'tests'))
        root = skills / 'tactical_v1'
        apply_variant(root)
        apply_simple(root, lings, hydras)
        with patch.dict(sys.modules), patch.object(sys, 'path', [tmp, *sys.path]):
            for name in list(sys.modules):
                if name == 'skills' or name.startswith('skills.'):
                    del sys.modules[name]
            yield importlib.import_module('skills.tactical_v1.skill_manager')


@unittest.skipUnless(FROZEN.is_dir(), 'requires recorded contact_response fixture')
class SimpleUnitsTests(unittest.TestCase):
    def test_only_selected_species_changes_recorded_final_actions(self):
        for lings in (True, False):
            with self.subTest(lings=lings), private(lings, not lings) as mod:
                manager = mod.SkillManager()
                changed = False
                # The first frame already changes hydras; frame two first changes a ling.
                for payload, expected in frames(2 if lings else 1):
                    actions, summary = manager.step(payload, 10)
                    for i, (old, new) in enumerate(zip(expected, actions)):
                        row = summary['units'][f'u{i}']
                        self.assertEqual((new, new), (row['action'], summary['actions'][f'u{i}']))
                        if (i < 3) == lings:
                            self.assertEqual(row['skill'], 'simple_units')
                            changed |= old != new
                        else:
                            self.assertEqual(old, new)
                    manager.commit_final_actions(actions)
                    self.assertTrue(all(int(r['actor'][1:]) >= 3 for r in manager.last_micro_intents))
                self.assertTrue(changed)

    def test_approach_grants_commit_override_and_receipt(self):
        with private(True, True) as mod:
            manager = mod.SkillManager()
            payload, _ = next(frames(1))
            actions, summary = manager.step(payload, 10)
            ctx = manager._pending_commit[-1]
            snap = ctx.snapshot
            self.assertEqual(manager.last_micro_intents, [])
            manager.commit_final_actions(actions)
            grants = manager.last_micro_intents
            self.assertEqual({r['actor'] for r in grants}, {f'u{i}' for i in range(3, 10)})
            for r in grants:
                a, e = r['actor'], r['target']
                self.assertEqual(e, min(snap.enemies, key=lambda t: ctx.distance(a, t)))
                self.assertIs(snap.raw_state['relations'][a][e]['smac_center_distance_allowed'], False)
                self.assertEqual((r['issued_loop'], r['kind'], r['phase'], r['macro_action_id']),
                                 (snap.game_loop, 'approach_then_attack', 'approaching', actions[int(a[1:])]))
                self.assertEqual((r['actor_tag'], r['target_tag']),
                                 (snap.raw_state['allies'][a]['tag'], snap.raw_state['visible_enemies'][e]['tag']))
            r = grants[0]
            a, e = r['actor'], r['target']
            for bad in (None, '0', str(2**64), 'invalid'):
                with patch.dict(snap.raw_state['visible_enemies'][e], tag=bad):
                    self.assertNotIn(a, [g['actor'] for g in mod._simple_intents(ctx, summary, [], actions)])
            loop = manager.memory.last_committed_loop
            shot = ctx.decode(a, 6 + int(e[1:])).action
            manager.record_micro_action(a, shot + 1)
            self.assertEqual(manager.memory.last_committed_actions[a], r['macro_action_id'])
            manager.record_micro_action(a, shot)
            self.assertEqual(manager.memory.last_committed_actions[a], shot)
            self.assertEqual(manager.memory.last_committed_loop, loop)
            manager.commit_final_actions(actions)
            self.assertFalse(manager.last_micro_intents)
            manager.reset()
            actions, _ = manager.step(payload, 10)
            actions[0] = 1  # even an unrelated final override invalidates this window
            manager.commit_final_actions(actions)
            self.assertFalse(manager.last_micro_intents)

    def test_current_legal_native_order_only_and_no_visibility(self):
        with private(False, True) as mod:
            payload, _ = list(frames(3))[-1]
            state = payload['state']
            a = 'u9'
            ctx = mod.Context(mod.build_snapshot(payload, 1))
            legal = sorted((e for e in ctx.attackable_enemies(a)
                            if state['relations'][a][e]['smac_center_distance_allowed']), key=lambda e: ctx.distance(a, e))
            near, far = legal
            attack = dict(ability_id=3674, target_unit=far, target_tag=state['visible_enemies'][far]['tag'])
            for orders, goal, phase in (([attack], far, 'keep_attack'),
                                        ([dict(attack, target_tag='0')], near, 'nearest_attack'),
                                        ([{'ability_id': 16}, attack], near, 'nearest_attack')):
                summary = {'actions': {}, 'units': {}}
                actions = [1] * 10
                with patch.dict(state['allies'][a], orders=orders):
                    mod._simple_units(ctx, actions, summary, [], ('hydralisk',))
                self.assertEqual((summary['units'][a]['phase'], summary['units'][a]['goal'], actions[9]),
                                 (phase, goal, ctx.attack_action(a, goal)))
            with patch.dict(state, visible_enemies={}):
                ctx = mod.Context(mod.build_snapshot(payload, 1))
                actions, summary, plans = [4] * 10, {}, [object()]
                self.assertIs(mod._simple_units(ctx, actions, summary, plans, ('hydralisk',)), plans)
                self.assertEqual((actions, summary), ([4] * 10, {}))


if __name__ == '__main__':
    unittest.main()
