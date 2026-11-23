"""Replay the real macro policies, including the native melee chase branch (no SC2)."""
from contextlib import ExitStack
from dataclasses import replace
import importlib
from inspect import signature
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'sc2_skill_adapter/scripts'))
from contact_all_attackers import FROZEN, PRE_MICRO, apply_variant
from contact_local_packs_variant import apply_variant as install_local
from sc2_skill_adapter.tests.test_contact_local_packs_variant import frames as zerg_frames
from skills.tactical_v1.skill_manager import SkillManager as LiveManager


def historical(stem, episode=1, count=1):
    with (PRE_MICRO.parents[1] / stem / 'events.jsonl').open() as f:
        for line in f:
            event = json.loads(line)
            if event.get('episode') != episode:
                continue
            if event['event'] == 'skill_observation':
                payload = event['payload']
            if event['event'] == 'step':
                yield payload, event['actions']
                count -= 1
                if not count:
                    return
    raise AssertionError('incomplete historical fixture')


def types(rows, p):
    return {p['state']['allies'][r['actor']]['type'] for r in rows}


class RaceChecks:
    def test_actual_protoss_and_terran_skills(self):
        for stem, ep, wanted in (
                ('protoss_2stalker_7zealot_1colossus_0', 3, {'Zealot', 'Stalker', 'Colossus'}),
                ('terran_2marine_7marauder_1medivac_0', 1, {'Marine', 'Marauder'})):
            p, expected = next(historical(stem, ep))
            with self.subTest(stem=stem):
                m = self.manager_cls()
                m.micro_all_attackers = True
                actions, s = m.step(p, 10)
                if self.manager_cls is not LiveManager:
                    self.assertEqual(actions, expected)
                m.commit_final_actions(actions)
                self.assertEqual(types(m.last_micro_intents, p), wanted)
                for g in m.last_micro_intents:
                    self.assertEqual(g['phase'], 'approaching')
                    self.assertEqual(g['target'], s['units'][g['actor']]['goal'])

    def test_bane_only_direct_contact_not_region_or_wait(self):
        m = self.manager_cls()
        m.micro_all_attackers = True
        found = []
        for p, _ in historical('zerg_4zergling_1baneling_5hydralisk_0', count=12):
            acts, s = m.step(p, 10)
            m.commit_final_actions(acts)
            d = s['execution_duties']['actors']['u0']
            for g in m.last_micro_intents:
                if g['actor'] == 'u0':
                    found.append(p['state']['game_loop'])
                    self.assertEqual((g['target'], d['reason']), (d['target'], 'cashout_direct_contact'))
                    self.assertIn(g['macro_action_id'], (2, 3, 4, 5))
        self.assertTrue(found)

    def test_whole_team_commit_and_receipts(self):
        p, _ = next(historical('zerg_3zergling_0baneling_7hydralisk_0'))
        m = self.manager_cls()
        m.micro_all_attackers = True
        actions, _ = m.step(p, 10)
        changed = [*actions[:-1], 5 if actions[-1] != 5 else 4]
        m.commit_final_actions(changed)
        self.assertFalse(m.last_micro_intents)
        m.reset()
        actions, _ = m.step(p, 10)
        m.commit_final_actions(actions)
        g = next(r for r in m.last_micro_intents if r['actor'] == 'u0')
        shot = 6 + int(g['target'][1:])
        loop = m.memory.last_committed_loop
        m.record_micro_action('u0', shot+1)
        self.assertEqual(m.memory.last_committed_actions['u0'], actions[0])
        m.record_micro_action('u0', shot)
        self.assertEqual(m.memory.last_committed_actions['u0'], shot)
        self.assertEqual(m.memory.last_committed_loop, loop)
        m.commit_final_actions(actions)
        self.assertFalse(m.last_micro_intents)


class LiveAllAttackersTest(RaceChecks, unittest.TestCase):
    manager_cls = LiveManager


class AllAttackersTest(RaceChecks, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        skills = Path(tmp.name) / 'skills'
        shutil.copytree(FROZEN.parent, skills, ignore=shutil.ignore_patterns('__pycache__'))
        cls.base = skills / 'tactical_v1'
        cls.all_dir = skills / 'with_all'
        shutil.copytree(cls.base, cls.all_dir)
        apply_variant(cls.all_dir)
        packs = skills / 'with_local'
        shutil.copytree(cls.base, packs)
        install_local(packs, True)
        apply_variant(packs)
        with_path = ExitStack()
        cls.addClassCleanup(with_path.close)
        with_path.enter_context(patch.object(sys, 'path', [tmp.name, *sys.path]))
        with_path.enter_context(patch.dict(sys.modules))
        for name in list(sys.modules):
            if name == 'skills' or name.startswith('skills.'):
                del sys.modules[name]
        cls.ref = importlib.import_module('skills.tactical_v1.skill_manager').SkillManager
        cls.mod = importlib.import_module('skills.with_all.skill_manager')
        cls.manager_cls = cls.mod.SkillManager
        cls.packs = importlib.import_module('skills.with_local.skill_manager')

    def test_install_only_private_recognized_packages(self):
        for path in (FROZEN, PRE_MICRO, ROOT / 'skills/tactical_v1', self.all_dir,
                     ROOT / 'runs/contact_local_packs_20261122_215806_250467/packs/skills/tactical_v1'):
            with self.assertRaises(ValueError):
                apply_variant(path)
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / 'v1'
            shutil.copytree(PRE_MICRO, p)
            link = Path(tmp) / 'link'
            link.symlink_to(p)
            with self.assertRaises(ValueError):
                apply_variant(link)
            (p / 'alias').symlink_to(p / 'config.py')
            with self.assertRaises(ValueError):
                apply_variant(p)
            (p / 'alias').unlink()
            apply_variant(p)
            self.assertEqual((p / 'skill_manager.py').read_bytes(), (self.all_dir / 'skill_manager.py').read_bytes())
            (p / 'skill_manager.py').write_bytes((self.base / 'skill_manager.py').read_bytes())
            (p / 'config.py').write_text('# modified')
            with self.assertRaises(ValueError):
                apply_variant(p)
            self.assertEqual((p / 'skill_manager.py').read_bytes(), (self.base / 'skill_manager.py').read_bytes())

    @unittest.skipUnless('marginal' in signature(install_local).parameters, 'near installer not installed')
    def test_local_first_then_all_marginal_and_near(self):
        for dispatch, horizon in ((False, 4), (True, 4), (True, 2)):
            with self.subTest(dispatch=dispatch, horizon=horizon), TemporaryDirectory() as tmp:
                root = Path(tmp) / 'v1'
                shutil.copytree(self.base, root)
                install_local(root, dispatch, marginal=True, reserve_horizon=horizon)
                original = {p.name: p.read_bytes() for p in root.glob('*.py')}
                apply_variant(root)
                self.assertEqual([p.name for p in root.glob('*.py') if original[p.name] != p.read_bytes()],
                                 ['skill_manager.py'])

    def test_historical_lings_move_and_continue_native_chase(self):
        for dogs in (3, 6):
            base, revised, off = self.ref(), self.mod.SkillManager(), self.mod.SkillManager()
            revised.micro_all_attackers = True
            move = chase = hydra = 0
            for p, expected, historical_grants in zerg_frames(dogs, 0, 10):
                state = p['state']
                for m in (base, revised, off):
                    acts, s = m.step(p, 10)
                    self.assertEqual(acts, expected)
                    self.assertEqual(m.last_micro_intents, [])
                    m.commit_final_actions(acts)
                legacy = base.last_micro_intents
                self.assertEqual(off.last_micro_intents, legacy)
                self.assertEqual(revised.last_micro_intents[:len(legacy)], legacy)
                self.assertEqual(len({g['actor_tag'] for g in revised.last_micro_intents}), len(revised.last_micro_intents))
                hydra += len(legacy)
                for g in revised.last_micro_intents[len(legacy):]:
                    a, e = g['actor'], g['target']
                    self.assertEqual(state['allies'][a]['type'], 'Zergling')
                    self.assertEqual(g['target_tag'], state['visible_enemies'][e]['tag'])
                    self.assertEqual(g['actor_tag'], state['allies'][a]['tag'])
                    self.assertEqual(s['units'][a]['goal'], e)
                    if g['macro_action_id'] >= 6:
                        self.assertEqual(g['macro_action_id'], 6+int(e[1:]))
                        self.assertEqual(s['execution_duties']['actors'][a]['reason'], 'native_contact_commit')
                        self.assertTrue(state['relations'][a][e]['smac_center_distance_allowed'])
                        self.assertFalse(state['relations'][a][e]['in_weapon_range'])
                        chase += 1
                    else:
                        move += 1
                for g in historical_grants:
                    if g['status'] == 'submitted':
                        for m in (base, revised, off):
                            m.record_micro_action(g['actor'], 6+int(g['target'][1:]))
            self.assertTrue(min(move, chase, hydra) > 0, (dogs, move, chase, hydra))

    def test_nonattack_duties_spoofed_targets_and_stale_plans(self):
        p, _ = next(historical('zerg_3zergling_0baneling_7hydralisk_0'))
        m = self.mod.SkillManager()
        acts, s = m.step(p, 10)
        snap, plans, _, _, ctx = m._pending_commit
        actor = 'u0'
        row = s['units'][actor]
        plan = next(plan for plan in plans if plan.proposal_id == row['plan'])
        duty = s['execution_duties']['actors'][actor]
        def granted(pl=plan):
            return any(g['actor'] == actor for g in self.mod.extra_approaches(ctx, s, [pl], acts))
        self.assertTrue(granted())
        for change in ({'kind': 'rescue'}, {'reason': 'synchronize_with_rear_fire'}, {'kind': 'support'},
                       {'target': 'e1'}, {'isolated_shooter_support': {'action': acts[0]}}, {'satisfied': False}):
            with patch.dict(duty, change):
                self.assertFalse(granted())
        for side, key in (('allies', actor), ('visible_enemies', row['goal'])):
            for tag in ('01', 'u0', 2**64, 0, None):
                with patch.dict(snap.raw_state[side][key], tag=tag):
                    self.assertFalse(granted())
        self.assertFalse(granted(replace(plan, snapshot_id=plan.snapshot_id+1)))
        for skill, phase in (('coordinated_engage', 'guarding'), ('ranged_fire_formation', 'exclusive_control'),
                             ('heal_support', 'approaching'), ('reserve_dispatch', 'reinforcing'),
                             ('disengage_reenter', 'ESCAPE'), ('local_formation', 'approaching')):
            with patch.dict(row, skill=skill, phase=phase):
                self.assertFalse(granted(replace(plan, primary_skill=skill,
                                                commitment=replace(plan.commitment, phase=phase))))

    def test_reserve_never_authorizes_transit(self):
        m = self.packs.SkillManager()
        m.micro_all_attackers = True
        reserve = 0
        for p, _, _ in zerg_frames(3, 0, 16):
            acts, s = m.step(p, 10)
            m.commit_final_actions(acts)
            for row in s['local_coord']['selected']:
                if row['skill'] == 'reserve_dispatch':
                    reserve += 1
                    self.assertNotIn(row['actor'], {r['actor'] for r in m.last_micro_intents})
        self.assertGreater(reserve, 0)

    def test_local_member_owns_its_own_target_not_the_group_goal(self):
        # Keep the real movement, tag, menu and duties of this ling. Supply
        # the pack provider's per-actor contract, with a different group goal.
        p, _ = next(historical('zerg_3zergling_0baneling_7hydralisk_0'))
        m = self.mod.SkillManager()
        actions, s = m.step(p, 10)
        snapshot, plans, _, _, ctx = m._pending_commit
        a = 'u0'
        row = s['units'][a]
        t, cmd = row['goal'], actions[0]
        old = next(pl for pl in plans if pl.proposal_id == row['plan'])
        tag = snapshot.raw_state['visible_enemies'][t]['tag']
        data = dict(pack_id='test-pack', roles={a: 'approach_attack'}, actor_target={a: t},
                    target_tags={t: tag}, actor_tags={a: snapshot.allies[a].tag}, reference_actions={a: 1})
        pl = replace(old, primary_skill='local_engagement', goal='e8', goal_data=data,
                     commitment=replace(old.commitment, phase='support'))
        self.assertNotEqual(pl.goal, t)
        row.update(goal=pl.goal, phase='support', skill=pl.primary_skill)
        s['execution_duties']['actors'][a].update(kind='entry', reason='local_contact', target=t)
        change = dict(before=1, after=cmd, pack_id=data['pack_id'], target_tag=tag, skill=pl.primary_skill)
        s['local_coord'] = dict(changes={a: change})
        def grants():
            return self.mod.extra_approaches(ctx, s, [pl], actions)
        grant, = grants()
        self.assertEqual((grant['target'], grant['target_tag']), (t, tag))
        self.assertEqual(self.mod.merge_approaches([grant], [grant]), [grant])
        for mapping, key, bad in ((data['roles'], a, 'reserve_approach'), (data, 'request_id', 'r'),
                                  (data['actor_tags'], a, '1'), (data['target_tags'], t, '2'),
                                  (change, 'before', cmd), (change, 'after', 1), (change, 'pack_id', 'other')):
            with patch.dict(mapping, {key: bad}):
                self.assertFalse(grants())


if __name__ == '__main__':
    unittest.main()
