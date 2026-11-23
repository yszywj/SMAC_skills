"""Run the copied manager against recorded inputs, without SC2 or live policies."""
import copy
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


def frames(lings, count):
    name = f'zerg_{lings}zergling_0baneling_{10-lings}hydralisk_0'
    with (FROZEN.parents[1] / name / 'events.jsonl').open() as stream:
        for line in stream:
            event = json.loads(line)
            if event['event'] == 'skill_observation':
                payload = event['payload']
            elif event['event'] == 'step':
                yield payload, event['actions']
                count -= 1
                if count == 0:
                    return


@unittest.skipUnless(FROZEN.is_dir(), 'historical contact_response fixture not installed')
class ContactResponseMicroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        skills = Path(tmp.name) / 'skills'
        shutil.copytree(FROZEN.parent, skills)
        cls.private = skills / 'tactical_v1'
        apply_variant(cls.private)
        # Do not resolve frozen shared helpers from an earlier test's imports.
        for p in (patch.dict(sys.modules), patch.object(sys, 'path', [tmp.name, *sys.path])):
            p.start()
            cls.addClassCleanup(p.stop)
        for name in list(sys.modules):
            if name == 'skills' or name.startswith('skills.'):
                del sys.modules[name]
        cls.module = importlib.import_module('skills.tactical_v1.skill_manager')

    def test_private_install_only(self):
        with self.assertRaises(ValueError):
            apply_variant(FROZEN)
        with self.assertRaises(ValueError):
            apply_variant(self.private)  # reapplying or mixing policy versions is not allowed
        self.assertEqual([p.name for p in FROZEN.rglob('*') if p.suffix in ('.py', '.json')
                          and p.read_bytes() != (self.private / p.relative_to(FROZEN)).read_bytes()],
                         ['skill_manager.py'])

    def test_off_replays_original_actions_with_nonzero_authority(self):
        for lings in (3, 6):
            with self.subTest(lings=lings):
                m = self.module.SkillManager()
                total = opened = 0
                previous = []
                for payload, expected in frames(lings, 8):
                    state = payload['state']
                    for r in previous:
                        actor, target = r['actor'], r['target']
                        rel = state['relations'].get(actor, {}).get(target, {})
                        if (state['allies'][actor]['tag'] == r['actor_tag'] and
                                state['visible_enemies'].get(target, {}).get('tag') == r['target_tag'] and
                                rel.get('smac_center_distance_allowed') and rel.get('smac_attack_available')):
                            opened += 1
                    actions, summary = m.step(payload, 10)
                    self.assertEqual(m.last_micro_intents, [])
                    self.assertEqual(actions, expected)  # recorded final commands, not shadow plans
                    m.commit_final_actions(actions)
                    self.assertEqual(m.memory.last_committed_actions, {f'u{i}': a for i, a in enumerate(expected)})
                    seen = set()
                    for r in m.last_micro_intents:
                        actor, target = r['actor'], r['target']
                        self.assertNotIn(actor, seen)
                        seen.add(actor)
                        row = summary['units'][actor]
                        self.assertEqual((r['kind'], r['issued_loop'], r['skill'], r['phase']),
                                         ('approach_then_attack', state['game_loop'], row['skill'], row['phase']))
                        self.assertEqual((r['macro_action_id'], target), (row['action'], row['goal']))
                        self.assertEqual(r['target_tag'], state['visible_enemies'][target]['tag'])
                        self.assertEqual(r['actor_tag'], state['allies'][actor]['tag'])
                        self.assertFalse(state['relations'][actor][target]['smac_center_distance_allowed'])
                    total += len(seen)
                    previous = m.last_micro_intents
                self.assertGreater(total, 0)
                self.assertGreater(opened, 0)  # at L+8, not proof of interior submission

    def test_arbitration_exclusions_and_receipt(self):
        m = self.module.SkillManager()
        payload, expected = next(frames(3, 1))
        actions, summary = m.step(payload, 10)
        ctx = m._pending_commit[-1]
        actor = 'u8'
        row = summary['units'][actor]
        self.assertEqual((row['phase'], row['goal']), ('exclusive_control', 'e8'))

        def grants(s):
            return [r for r in self.module._micro_intents(ctx, s, m._pending_commit[1], actions)
                    if r['actor'] == actor]

        self.assertEqual(len(grants(summary)), 1)
        for change in ({'phase': 'follow_zergling'}, {'phase': 'searching'},
                       {'skill': 'disengage_reenter'}, {'goal': 'e7'}, {'goal_kind': 'ally'}):
            s = copy.deepcopy(summary)
            s['units'][actor].update(change)
            self.assertFalse(grants(s))
        for change in ({'kind': 'rescue'}, {'joint_rescue': {'substituted': True}}, {'satisfied': False}):
            s = copy.deepcopy(summary)
            s['execution_duties']['actors'][actor].update(change)
            self.assertFalse(grants(s))
        with patch.dict(ctx.snapshot.raw_state['visible_enemies']['e8'], tag=None):
            self.assertFalse(grants(summary))
        with patch.object(ctx._ranged_formation_forecast, 'threats_at', side_effect=lambda a, point:
                          {'e7'} if point == ctx.snapshot.allies[a].position else set()):
            self.assertFalse(grants(summary))

        changed = list(expected)
        changed[0] = 1  # even another team member can invalidate a joint approach
        m.commit_final_actions(changed)
        self.assertFalse(m.last_micro_intents)
        m.reset()
        m.step(payload, 10)
        m.commit_final_actions(actions)
        self.assertTrue(m.last_micro_intents)
        loop = m.memory.last_committed_loop
        m.record_micro_action(actor, 13)  # e7, not the authorized e8
        self.assertEqual(m.memory.last_committed_actions[actor], expected[8])
        m.record_micro_action(actor, 14)
        self.assertEqual(m.memory.last_committed_actions[actor], 14)
        self.assertEqual(m.memory.last_committed_loop, loop)
        m.commit_final_actions(actions)  # repeated commit cannot resurrect a grant
        self.assertEqual(m.last_micro_intents, [])
        m.reset()
        self.assertEqual(m.last_micro_intents, [])


if __name__ == '__main__':
    unittest.main()
