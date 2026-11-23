"""Complete manager/commit replay against recorded macro observations, no SC2."""
import copy
import importlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from contact_local_packs_variant import FROZEN, RUN, _survival_margin, apply_variant


def frames(dogs, shard, count):
    folder = RUN / 'micro8-hydra' / f'zerg_{dogs}zergling_0baneling_{10-dogs}hydralisk_{shard:02d}'
    grants = json.loads((folder / 'micro_triggers/episode-0001.json').read_text())['authorizations']
    with (folder / 'events.jsonl').open() as f:
        for line in f:
            event = json.loads(line)
            if event['event'] == 'skill_observation':
                payload = event['payload']
            if event['event'] == 'step':
                assert event['episode'] == 1
                yield payload, event['actions'], [g for g in grants if g['issued_loop'] == payload['state']['game_loop']]
                count -= 1
                if not count:
                    return
    raise AssertionError('incomplete historical fixture')


class LocalPacksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.skills = Path(tmp.name) / 'skills'
        shutil.copytree(FROZEN.parent, cls.skills, ignore=shutil.ignore_patterns('__pycache__'))
        for name, dispatch in [('packs', False), ('reserve', True)]:
            root = cls.skills / name
            shutil.copytree(cls.skills / 'tactical_v1', root)
            apply_variant(root, dispatch)
        for p in (patch.dict(sys.modules), patch.object(sys, 'path', [tmp.name, *sys.path])):
            p.start()
            cls.addClassCleanup(p.stop)
        for name in list(sys.modules):
            if name == 'skills' or name.startswith('skills.'):
                del sys.modules[name]
        cls.ref = importlib.import_module('skills.tactical_v1.skill_manager').SkillManager
        cls.packs = importlib.import_module('skills.packs.skill_manager')
        cls.reserve = importlib.import_module('skills.reserve.skill_manager')

    def test_install_only_frozen_private_copy(self):
        with self.assertRaises(ValueError):
            apply_variant(FROZEN, True)
        with self.assertRaises(ValueError):
            apply_variant(self.skills / 'packs', True)
        original = {p.relative_to(FROZEN): p.read_bytes() for p in FROZEN.rglob('*') if p.suffix in ('.py', '.json')}
        for name in ('packs', 'reserve'):
            root = self.skills / name
            installed = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.suffix in ('.py', '.json')}
            self.assertEqual(set(installed) - set(original), {Path('local_engagement.py')})
            self.assertEqual(installed[Path('local_engagement.py')], (SCRIPTS / 'local_engagement_template.py').read_bytes())
            self.assertEqual({k.as_posix() for k in original if original[k] != installed[k]},
                             {'coordinator.py', 'skill_manager.py'})

    def test_reference_only_matches_history_and_micro(self):
        # Baseline does NOT install apply_variant. Also force a real no-proposal
        # frame in the experiment: its orchestration alone must not change policy.
        m, base = self.packs.SkillManager(), self.ref()
        local = importlib.import_module('skills.packs.local_engagement')
        no_offer = dict(in_scope=False, packs=[], requests=[], releases=[], idle=[], candidate_count=0)
        with patch.object(local, 'propose_local', side_effect=lambda *_: ([], copy.deepcopy(no_offer))):
            for payload, actual, grants in frames(3, 0, 5):
                orig, before = base.step(payload, 10)
                acts, after = m.step(payload, 10)
                self.assertNotIn('local_coord', before)
                self.assertEqual((orig, acts), (actual, actual))
                self.assertEqual(after['units'], before['units'])
                self.assertEqual(after['chosen_plans'], before['chosen_plans'])
                self.assertFalse(after['local_coord']['selected'])
                base.commit_final_actions(orig)
                m.commit_final_actions(acts)
                self.assertEqual(base.last_micro_intents, m.last_micro_intents)
                fields = ('actor', 'target_tag', 'macro_action_id', 'issued_loop')
                signature = lambda rows: sorted(tuple(r[k] for k in fields) for r in rows)
                self.assertEqual(signature(m.last_micro_intents), signature(grants))
                for g in grants:
                    if g['status'] == 'submitted':
                        for policy in (base, m):
                            policy.record_micro_action(g['actor'], int(g['target'][1:])+6)

    def test_severe_exposure_is_not_traded_for_more_fire(self):
        ctx = SimpleNamespace(snapshot=SimpleNamespace(allies={a: SimpleNamespace(
            health_lost=4, shield_lost=0) for a in ('u0', 'u1')}))
        def risk(x, y, unknown=False):
            return {a: {'after': {'reserve_ratio': v, 'unknown': unknown}} for a,v in zip(('u0', 'u1'), (x, y))}
        self.assertIsNone(_survival_margin(ctx, risk(.9, .8), risk(.5, 1.0)))
        self.assertIsNone(_survival_margin(ctx, risk(.9, .4), risk(1.15, .4)))
        self.assertEqual(_survival_margin(ctx, risk(.9, .4), risk(.5, .4)), 1)
        self.assertEqual(_survival_margin(ctx, risk(.9, .4), risk(.5, .4, True)), 0)

    def test_joint_comparison_not_limited_to_valid_singletons(self):
        # Exercise the allocator independently of proposal heuristics. Two
        # alternative open routes in a real observation; a test dependency
        # deliberately rejects each singleton. No scoring/validation mocks.
        payload, _, _ = next(frames(3, 0, 1))
        manager = self.packs.SkillManager()
        model = importlib.import_module('skills.packs.local_engagement')
        common = importlib.import_module('skills.packs.skills.common')
        def propose(ctx, memory, reference, dispatch):
            plans, packs = [], []
            for a, e in (('u4', 'e2'), ('u5', 'e7')):
                self.assertNotEqual(reference[a], 4)
                tag = ctx.snapshot.raw_state['visible_enemies'][e]['tag']
                d = dict(actor_target={a:e}, target_tags={e:tag}, pack_id=a, roles={a:'approach_attack'},
                         protected=['u9'], reference_actions={a:reference[a]})
                plans.append(common.make_plan('local_engagement', [a], e, {a:4}, [], common.ResourceClaims(controls=(a,)),
                                common.Commitment('entry'), {}, ctx.snapshot.snapshot_id, '', goal_kind='enemy', goal_data=d))
                packs.append(dict(pack_id=a, members=[a], enemies=[e], protected=['u9']))
            return plans, dict(in_scope=True, packs=packs, requests=[], idle=[], releases=[])
        def conflicts(ctx, plans, actions):
            return ['requires_peer'] if sum(p.primary_skill == 'local_engagement' for p in plans) != 2 else []
        with patch.object(model, 'propose_local', side_effect=propose), patch.object(model, 'local_conflicts', side_effect=conflicts):
            acts, summary = manager.step(payload, 10)
            trace = summary['local_coord']
            self.assertEqual(trace['max_joint_plans'], 2, trace)
            self.assertEqual(trace['valid_branches'], 1)
            self.assertEqual(trace['rejection_counts'], {'local_conflict:requires_peer': 2})
            self.assertEqual(len(set(s['pack_id'] for s in trace['selected'])), 2 if trace['changes'] else 0)

    def test_real_arbitration_commit_and_scoped_changes(self):
        for module in (self.packs, self.reserve):
            manager = module.SkillManager()
            cnt = dict(frames=0, scored=0, changed=0, combined=0)
            validator = importlib.import_module(module.__package__ + '.validator')
            for payload, recorded, _ in frames(3, 0, 16):
                acts, summary = manager.step(payload, 10)
                trace = summary['local_coord']
                snapshot, plans, _, _, ctx = manager._pending_commit
                self.assertEqual(validator.validate_actions(acts, ctx, 10), [])
                self.assertEqual(validator.validate_plans(plans, ctx, summary['actions'],
                                 manager.evaluator.effects(ctx, summary['actions']), manager.memory), [])
                self.assertEqual({p.proposal_id for p in plans}, {p['id'] for p in summary['chosen_plans']})
                self.assertEqual(trace['changes'].keys(), {a for a,v in trace['reference_actions'].items()
                                                          if v != summary['actions'][a]})
                for a, cmd in summary['actions'].items():
                    forced = ctx.forced_hydra_targets(a)
                    if forced:
                        self.assertIn(ctx.decode(a, cmd).target, forced)
                for row in trace['selected']:
                    a = row['actor']
                    self.assertEqual(row['target_tag'], payload['state']['visible_enemies'][row['target']]['tag'])
                    self.assertEqual(row['after'], acts[snapshot.allies[a].index])
                    self.assertEqual(row['before'], trace['reference_actions'][a])
                    self.assertEqual(row['plan_id'], summary['units'][a]['plan'])
                cnt['frames'] += 1
                cnt['scored'] += trace['valid_branches'] > 1
                cnt['changed'] += len(trace['changes'])
                cnt['combined'] += trace['max_joint_plans'] >= 2
                # A replay can only check decision/commit on the recorded state,
                # not the counterfactual next observation or game outcome.
                submitted = list(acts)
                if trace['changes'] and not cnt.get('mismatch'):
                    a = next(iter(trace['changes']))
                    submitted[snapshot.allies[a].index] = trace['changes'][a]['before']
                    cnt['mismatch'] = 1
                with patch.object(manager.memory, 'apply_delta', wraps=manager.memory.apply_delta) as apply:
                    manager.commit_final_actions(submitted)
                    committed = {c.args[2].proposal_id for c in apply.call_args_list if c.args[2] in plans}
                    self.assertEqual(committed, {p.proposal_id for p in plans if all(submitted[snapshot.allies[a].index]
                                     == cmd for a, cmd in p.current_actions.items())})
                if submitted != acts:
                    self.assertEqual(manager.last_micro_intents, [])
                for grant in manager.last_micro_intents:
                    actor = grant['actor']
                    for row in trace['selected']:
                        if row['actor'] == actor:
                            self.assertEqual((row['changed'], row['role'], row['target_tag'], row['skill']),
                                (True, 'approach_attack', grant['target_tag'], 'local_engagement'))
            self.assertGreater(cnt['scored'], 0, cnt)
            self.assertGreater(cnt['changed'], 0, cnt)
            self.assertGreater(cnt['combined'], 0, cnt)

    def test_named_micro_role_and_rejections(self):
        # Reuse an actual qualifying physical move/range/tag pair. Vary only
        # its ownership, role, target evidence and reference action in isolation.
        payload, _, grants = next(frames(3, 0, 1))
        base = self.ref()
        acts, summary = base.step(payload, 10)
        ctx = base._pending_commit[-1]
        g = grants[0]
        actor, target, tag = g['actor'], g['target'], g['target_tag']
        cmd = acts[ctx.snapshot.allies[actor].index]
        data = dict(pack_id='pack', actor_target={actor: target}, target_tags={target: tag}, roles={actor: 'approach_attack'})
        plan = SimpleNamespace(controlled_actors=(actor,), current_actions={actor: cmd}, proposal_id='offer',
                               primary_skill='local_engagement', goal_data=data)
        row = summary['units'][actor]
        row.update(plan='offer', skill=plan.primary_skill, phase='testing')
        summary['execution_duties']['actors'][actor].update(target=target, reason='local_contact')
        change = dict(before=1, after=cmd, pack_id='pack', target_tag=tag, skill=plan.primary_skill)
        summary['local_coord'] = dict(changes={actor: change})
        def authorized(plans):
            return [(r['target'], r['target_tag']) for r in self.packs._micro_intents(ctx, summary, plans, acts)
                    if r['actor'] == actor]
        self.assertEqual(authorized([plan]), [(target, tag)])
        for container, key, value in ((data['roles'], actor, 'protect'), (data['roles'], actor, 'reserve_approach'),
                                       (data, 'request_id', 'dispatch'), (data['target_tags'], target, '100'),
                                       (change, 'before', cmd), (change, 'pack_id', 'other'),
                                       (row, 'skill', 'reserve_dispatch')):
            with patch.dict(container, {key: value}):
                self.assertFalse(authorized([plan]))
        self.assertFalse(authorized([]))

    def test_valid_menu_does_not_override_mandatory_focus(self):
        manager = self.ref()
        coordinator = importlib.import_module('skills.packs.coordinator')
        for payload, recorded, _ in frames(3, 0, 24):
            acts, summary = manager.step(payload, 10)
            ctx = manager._pending_commit[-1]
            for a in summary['actions']:
                forced = ctx.forced_hydra_targets(a)
                if not forced:
                    continue
                target = next((e for e in ctx.attackable_enemies(a) if e not in forced), None)
                if target is None:
                    continue
                commands = {**summary['actions'], a: ctx.attack_action(a, target)}
                p = SimpleNamespace(controlled_actors=(a,), goal_data={'actor_target': {a: target}, 'target_tags':
                    {target: ctx.snapshot.raw_state['visible_enemies'][target]['tag']}})
                original_formation = dict(ctx.formation_orders)
                self.assertIsNone(coordinator._local_context(ctx, manager.memory,
                        SimpleNamespace(actions=summary['actions']), [p], commands))
                self.assertEqual(original_formation, ctx.formation_orders)
                return
            manager.commit_final_actions(acts)
        self.fail('no historical legal non-mandatory target found')


if __name__ == '__main__':
    unittest.main()
