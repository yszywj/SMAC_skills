"""Exercise the actual private installers, offers, arbitration and commit, offline."""
import importlib
from hashlib import sha256
from pathlib import Path
import shutil
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sc2_skill_adapter.tests import test_contact_local_packs_variant as old
from sc2_skill_adapter.tests import test_local_engagement_template as geometry


def digest(root):
    result = sha256()
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if path.suffix in ('.py', '.json') and 'tests' not in rel.parts:
            result.update(rel.as_posix().encode() + b'\0' + sha256(path.read_bytes()).digest())
    return result.hexdigest()


class OptionTest(unittest.TestCase):
    window = geometry.LocalEngagementTest.window
    commit = geometry.LocalEngagementTest.commit

    @classmethod
    def setUpClass(cls):
        old.LocalPacksTest.setUpClass()
        cls.addClassCleanup(old.LocalPacksTest.doClassCleanups)
        cls.skills = old.LocalPacksTest.skills
        for name, dispatch, marginal, beats in (('value', False, True, 4), ('near', True, False, 2),
                                                ('both', True, True, 2)):
            root = cls.skills / name
            shutil.copytree(old.FROZEN, root)
            old.apply_variant(root, dispatch, marginal=marginal, reserve_horizon=beats)
        cls.L = importlib.import_module('skills.near.local_engagement')
        cls.far = importlib.import_module('skills.reserve.local_engagement')
        cls.value = importlib.import_module('skills.value.local_engagement')
        cls.context = importlib.import_module('skills.near.context')
        cls.memory = importlib.import_module('skills.near.memory').TacticalMemory
        cls.payload = next(old.frames(3, 0, 1))[0]

    def test_released_packages_byte_identical(self):
        # Released 20261122_215806 packages, including generated coordinator and manager.
        self.assertEqual([digest(self.skills / name) for name in ('packs', 'reserve')], [
            'd7a034fc07777ef663402e0d8d81cb86057bf3aa533657dde304d3f78564e621',
            'd6dc56292476dbd53d05570fa735d7ee9c82849b4964686ef6e71b959502dc24'])
        for args, exc in [({'marginal':1}, TypeError), ({'reserve_horizon':True}, ValueError),
                          ({'reserve_horizon':3}, ValueError)]:
            with self.assertRaises(exc):
                old.apply_variant(self.skills / 'tactical_v1', True, **args)

    def scene(self, loop, x, *, e='e0', tag='201', ex=0., home_hp=90., more=False):
        allies = {'u0':(ex,.4), 'u4':(x,0.), 'u6':(10.4,1.4)}
        if more:
            allies.update(u7=(2,0),u8=(0,3))
        return self.window(allies, {e:(ex,0.), 'e1':(13.7,3.)}, loop,
                           health={'u6':home_hp}, tags={e:tag}, enemy_dogs=[e])

    def reserves(self, mod, ctx, ref, memory, tag='201'):
        plans, audit = mod.propose_local(ctx, memory, ref, True)
        return [p for p in plans if p.primary_skill == mod.RESERVE and
                p.goal_data['target_tags'][p.goal] == tag], audit

    def test_two_beats_not_four_and_no_auto_renew(self):
        mem = self.memory()
        ctx, ref = self.scene(2, 8.)
        far, _ = self.reserves(self.far, ctx, ref, mem)
        near, tr = self.reserves(self.L, ctx, ref, mem)
        self.assertTrue(far)
        self.assertFalse(near)
        self.assertTrue(all(p.goal_data['eta_loops'] > 16 for p in far))
        request = next(r for r in tr['requests'] if r['target_tag'] == '201')
        self.assertEqual((request['created_loop'], request['expires_loop'], request['claimed_by']), (2,18,[]))
        self.commit(ctx, mem, ref, tr)
        ctx, ref = self.scene(10, 7.)
        near, tr = self.reserves(self.L, ctx, ref, mem)
        self.assertFalse(near)  # fresh actor, only 8 loops remain on that request
        self.assertEqual(request, next(r for r in tr['requests'] if r['target_tag'] == '201'))
        self.commit(ctx, mem, ref, tr)
        ctx, ref = self.scene(18, 7.)
        self.assertFalse(any(r['target_tag'] == '201' for r in self.reserves(self.L, ctx, ref, mem)[1]['requests']))

    def test_followup_requires_gap_target_progress_and_home_line(self):
        mem = self.memory()
        ctx, ref = self.scene(2, 7.)
        plans, trace = self.reserves(self.L, ctx, ref, mem)
        self.assertEqual([p.controlled_actors for p in plans], [('u4',)])
        plan = plans[0]
        r = next(r for r in trace['requests'] if r['target'] == 'e0')
        self.assertEqual(r['claimed_by'], [])
        self.assertFalse(mem.instances)
        self.commit(ctx, mem, ref, trace, [plan])
        self.assertIn('107', mem.instances[plan.instance_id].extra['home_line'])
        for x, changes, allowed in [(5.95,dict(e='e7'), True), (7.,dict(ex=1.), False),
                                    (5.95,dict(home_hp=88.), False), (5.95,dict(more=True), False),
                                    (5.95,dict(tag='999'), False), (4.9,{},False)]:
            with self.subTest(x=x, changes=changes):
                ctx, ref = self.scene(10, x, **changes)
                plans, tr = self.reserves(self.L, ctx, ref, mem, tag=changes.get('tag','201'))
                self.assertEqual([p.instance_id for p in plans], [plan.instance_id] if allowed else [])
                if allowed:
                    req = next(r for r in tr['requests'] if r['request_id'] == plan.goal_data['request_id'])
                    self.assertEqual((req['expires_loop'],req['claimed_by']), (18,['u4']))
                    self.assertNotIn('e7', next(p for p in tr['packs'] if 'e7' in p['enemies'])['fire_rays']['u4'])
                else:
                    self.assertTrue(any(row['actor']=='u4' for row in tr['releases']))
        self.commit(ctx, mem, ref, tr)
        self.assertNotIn(plan.instance_id, mem.instances)

    def test_marginal_changes_actual_candidates_with_same_pack_geometry(self):
        # Legacy prioritizes the damaged target; every eligible shooter points
        # there. Preserve its existing gun but offer a NEW ray onto the other
        # enemy, without widening the pack or borrowing outsiders.
        ctx, ref = self.window({'u3':(-1,0),'u4':(-2,1),'u5':(-2,-1)},
                               {'e0':(1,0),'e1':(2,0)}, health={'e0':70})
        ref['u3'] = 6
        old_plans, old_tr = self.far.propose_local(ctx, self.memory(), ref, False)
        new_plans, new_tr = self.value.propose_local(ctx, self.memory(), ref, False)
        self.assertEqual(old_tr['packs'], new_tr['packs'])
        def solo(plans):
            return {p.goal for p in plans if len(p.controlled_actors)==1}
        self.assertIn('e1', solo(new_plans))
        self.assertNotEqual([p.proposal_id for p in new_plans], [p.proposal_id for p in old_plans])
        ctx, ref = self.window({'u3':(-1,0),'u4':(-2,1),'u5':(-2,-1)},
                               {'e0':(1,0),'e1':(2,0)}, health={'e0':5})
        ref['u3'] = 6
        ps, _ = self.value.propose_local(ctx, self.memory(), ref, False)
        self.assertFalse(any('u3' in p.controlled_actors for p in ps))

    def test_real_manager_joint_score_and_dispatch(self):
        m = importlib.import_module('skills.both.skill_manager').SkillManager()
        value = importlib.import_module('skills.value.skill_manager').SkillManager()
        legacy = old.LocalPacksTest.packs.SkillManager()
        selected, reserve, switches = 0, 0, []
        for payload, _, _ in old.frames(3, 0, 18):
            av, sv = value.step(payload, 10)
            ao, _ = legacy.step(payload, 10)
            if av[7] != ao[7] and 'u7' in sv['local_coord']['changes']:
                switches.append((payload['state']['game_loop'], ao[7], av[7],
                                 sv['score_breakdown']['pack_marginal_delta']))
            value.commit_final_actions(av)
            legacy.commit_final_actions(ao)
            actions, summary = m.step(payload, 10)
            tr = summary['local_coord']
            self.assertTrue(all(r['expires_loop'] - r['created_loop'] <= 16 for r in tr['requests']))
            for row in tr['selected']:
                if row['skill'] == 'reserve_dispatch':
                    self.assertIn(row['actor'], tr['idle'])
                    reserve += int(row['changed'])
                else:
                    selected += int(row['changed'])
            m.commit_final_actions(actions)
        self.assertGreater(selected, 0)
        self.assertGreater(reserve, 0)
        self.assertIn((26, 8, 13, .9), switches)

    def test_emergency_ling_is_not_displaced_by_ranged_preference(self):
        ctx, ref = self.window({'u3':(0,0),'u4':(-1.5,1),'u5':(-1.5,-1)},
                               {'e0':(.5,0),'e1':(2,0)}, health={'u3':55,'e0':7}, enemy_dogs=['e0'])
        plans, _ = self.value.propose_local(ctx, self.memory(), ref, False)
        singles = [p for p in plans if len(p.controlled_actors)==1]
        self.assertEqual(singles[0].goal, 'e0')
        ref['u4'] = 6  # sole finishing shot; preserve it in ALL branches
        plans, _ = self.value.propose_local(ctx, self.memory(), ref, False)
        self.assertFalse(any('u4' in p.controlled_actors for p in plans))


if __name__ == '__main__':
    unittest.main()
