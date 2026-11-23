"""Opening decisions and commit scope only. No environment or predicted rollout."""
import copy
import importlib
import json
from math import dist
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from contact_muster_variant import FROZEN, RUN, apply_variant


def frames(dogs, batch, count=1):
    scenario = f'zerg_{dogs}zergling_0baneling_{10-dogs}hydralisk_{batch:02}'
    result = []
    with (RUN / 'micro8-hydra' / scenario / 'events.jsonl').open() as f:
        for line in f:
            event = json.loads(line)
            if event['event'] == 'skill_observation':
                payload = event['payload']
            elif event['event'] == 'step':
                result.append((payload, event['actions']))
                if len(result) == count:
                    return result
    raise AssertionError('incomplete historical fixture')


class OpeningMusterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        skills = Path(tmp.name) / 'skills'
        shutil.copytree(FROZEN.parent, skills)
        cls.private = skills / 'muster'
        shutil.copytree(FROZEN, cls.private)
        apply_variant(cls.private)
        for p in (patch.dict(sys.modules), patch.object(sys, 'path', [tmp.name, *sys.path])):
            p.start()
            cls.addClassCleanup(p.stop)
        for name in list(sys.modules):
            if name == 'skills' or name.startswith('skills.'):
                del sys.modules[name]
        cls.original = importlib.import_module('skills.tactical_v1.skill_manager').SkillManager
        cls.module = importlib.import_module('skills.muster.skill_manager')

    def test_install_scope(self):
        for path in (FROZEN, self.private):
            with self.assertRaises(ValueError):
                apply_variant(path)
        self.assertEqual([p.relative_to(FROZEN).as_posix() for p in FROZEN.rglob('*')
                          if p.suffix in ('.py', '.json') and
                          p.read_bytes() != (self.private / p.relative_to(FROZEN)).read_bytes()],
                         ['skill_manager.py'])

    def test_real_spread_team_commands_and_commit_scope(self):
        for dogs, batch in ((3, 0), (6, 10)):
            with self.subTest(dogs=dogs):
                ref, m = self.original(), self.module.SkillManager()
                (payload, expected), (next_payload, _) = frames(dogs, batch, 2)
                base, _ = ref.step(payload, 10)
                self.assertEqual(base, expected)
                actions, summary = m.step(payload, 10)
                ctx = m._pending_commit[-1]
                info = summary['opening_muster']
                self.assertTrue(info['active'])
                held = set(info['held'])
                self.assertEqual(held | info['released'].keys(), set(ctx.snapshot.allies))
                self.assertTrue(held.isdisjoint(info['released']))
                self.assertGreaterEqual(len(held), 8)
                self.assertGreaterEqual(len(info['changes']), 5)
                self.assertTrue(held - info['changes'].keys())  # ownership even for equal actions
                self.assertEqual(info['changes'], {f'u{i}': dict(before=b, after=a)
                                 for i, (a, b) in enumerate(zip(actions, base)) if a != b})
                self.assertTrue(held.issuperset(info['changes']))
                pos = {a: u.position for a, u in ctx.snapshot.allies.items()}
                anchor = info['anchor']
                self.assertIn(anchor, [pos[f'u{i}'] for i in range(dogs, 10)])
                center = tuple(sum(p[i] for p in pos.values()) / 10 for i in (0, 1))
                for e in ctx.snapshot.enemies.values():
                    self.assertGreaterEqual(dist(e.position, anchor), dist(e.position, center))
                slots = [summary['units'][a]['slot'] for a in held]
                self.assertEqual(len(slots), len(set(slots)))
                self.assertTrue(all(dist(s, anchor) <= 4 for s in slots))
                for a in held:
                    cmd = actions[int(a[1:])]
                    row = summary['units'][a]
                    self.assertEqual((row['plan'], row['skill'], row['action'], summary['actions'][a]),
                                     (None, 'opening_muster', cmd, cmd))
                    self.assertIn(cmd, ctx.legal_actions(a))
                    if cmd != 1:
                        end = ctx.move_destinations(a)[cmd]
                        self.assertLess(dist(end, row['slot']), dist(pos[a], row['slot']))
                # Return and pending commit must agree on whole-plan ownership.
                self.assertTrue(all(held.isdisjoint(p['actors']) for p in summary['chosen_plans']))
                kept = {p.proposal_id for p in m._pending_commit[1]}
                self.assertEqual(kept, {p.proposal_id for p in ref._pending_commit[1]
                                        if held.isdisjoint(p.controlled_actors)})
                # Mutable return telemetry must not affect the cached result or grants.
                info['held'].clear()
                self.assertEqual(set(m.step(payload, 10)[1]['opening_muster']['held']), held)
                ref.commit_final_actions(base)
                m.commit_final_actions(actions)
                grants = {r['actor'] for r in m.last_micro_intents}
                self.assertTrue(grants.isdisjoint(held))
                self.assertEqual(m.memory.last_committed_actions, {f'u{i}': a for i, a in enumerate(actions)})
                if dogs == 3:
                    self.assertEqual(info['released'], {'u9': 'contact', 'u7': 'support_contact'})
                    self.assertTrue(held.intersection(r['actor'] for r in ref.last_micro_intents))
                    self.assertIn('u7', grants)  # local release is not team termination
                _, later = m.step(next_payload, 10)
                self.assertTrue(later['opening_muster']['active'])
                self.assertEqual(anchor, later['opening_muster']['anchor'])

    def test_distant_attack_commands_do_not_latch_team_contact(self):
        p, _ = frames(6, 10)[0]
        ref = self.original()
        original, s = ref.step(p, 10)
        _, plans, _, _, ctx = ref._pending_commit
        muster = self.module._OpeningMuster()
        e = next(iter(ctx.snapshot.enemies.values()))
        target = 6 + e.index
        # Unmask this visible enemy for every actor without moving it. A legal
        # menu is not physical contact. Preserve actual commands only locally.
        for a, menu in ctx.snapshot.legal.items():
            ctx.snapshot.legal[a] = menu | {target}
        cmds = list(original)
        cmds[0] = target
        s1 = copy.deepcopy(s)
        muster.apply(ctx, cmds, s1, plans)
        self.assertEqual(s1['opening_muster']['released'], {'u0': 'attack'})
        self.assertEqual(len(s1['opening_muster']['held']), 9)
        self.assertEqual(cmds[0], target)
        s2 = copy.deepcopy(s)
        self.assertIs(muster.apply(ctx, [target]*10, s2, plans), plans)
        self.assertEqual(s2['opening_muster']['reason'], 'local_release')
        s3 = copy.deepcopy(s)
        muster.apply(ctx, original, s3, plans)
        self.assertTrue(s3['opening_muster']['active'])
        self.assertEqual(s1['opening_muster']['anchor'], s3['opening_muster']['anchor'])

    def test_real_center_immediately_permanently_returns_to_micro8(self):
        for dogs, batch, spread in ((3, 5, 0), (6, 0, 10)):
            with self.subTest(dogs=dogs):
                ref, m = self.original(), self.module.SkillManager()
                for p, recorded in frames(dogs, batch, 2):
                    b, _ = ref.step(p, 10)
                    a, s = m.step(p, 10)
                    self.assertEqual((a, b), (recorded, recorded))
                    info = s['opening_muster']
                    self.assertEqual((info['active'], info['reason'], info['changes']),
                                     (False, 'already_grouped', {}))
                    ref.commit_final_actions(b)
                    m.commit_final_actions(a)
                    self.assertTrue(m.last_micro_intents)
                    self.assertEqual(ref.last_micro_intents, m.last_micro_intents)
                p, _ = frames(dogs, spread)[0]
                p['state']['game_loop'] += 16
                self.assertFalse(m.step(p, 10)[1]['opening_muster']['active'])
                m.reset()
                self.assertTrue(m.step(p, 10)[1]['opening_muster']['active'])

    def test_timeout_reset_and_outside_roster(self):
        m = self.module.SkillManager()
        p, _ = frames(3, 0)[0]
        start = p['state']['game_loop']
        for elapsed, reason in ((0, 'gathering'), (40, 'gathering'), (48, 'timeout'), (56, 'timeout')):
            p['state']['game_loop'] = start + elapsed
            actions, s = m.step(p, 10)
            self.assertEqual(s['opening_muster']['reason'], reason)
            m.commit_final_actions(actions)
        m.reset()
        self.assertTrue(m.step(p, 10)[1]['opening_muster']['active'])
        # Mixed and pure-but-wrong rosters: same decisions and grants as original.
        for name, tid in (('Baneling', 9), ('Hydralisk', 107)):
            modified = copy.deepcopy(p)
            modified['state']['allies']['u0'].update(type=name, unit_type_id=tid)
            ref, m = self.original(), self.module.SkillManager()
            a, s = m.step(modified, 10)
            b, _ = ref.step(modified, 10)
            self.assertEqual(s['opening_muster']['reason'], 'outside_roster')
            self.assertEqual(a, b)
            ref.commit_final_actions(b)
            m.commit_final_actions(a)
            self.assertEqual(m.last_micro_intents, ref.last_micro_intents)

if __name__ == '__main__':
    unittest.main()
