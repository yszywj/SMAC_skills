"""Create an original Shadow archive and exercise the one-shot arms without SC2."""
import asyncio
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bridge import runner
from bridge.zerg_fork import Fork
from skills.tactical_v1.tests import fixtures as fx
from test_zerg_shadow_runner import OfflineEnv, manifest


class TeamEnv(OfflineEnv):
    n_agents = 3

    def get_avail_agent_actions(self, i):
        return [0, 1, 1, 1, 1, 1, 0, 0]

    def step_micro_window(self, actions, window, macro_loops):
        self.actions = actions
        self._obs.observation.game_loop += macro_loops
        return 0.25, self._obs.observation.game_loop == 44, dict(dead_allies=0, dead_enemies=2, battle_won=True)


def payload(env, _data, _model, episode, *unused):
    return fx.with_questions(fx.scene([((0., 0.), 'zergling'), ((8., 0.), 'hydralisk'), ((16., 0.), 'zergling')],
                {'e0': fx.enemy((30., 0.))}, episode=episode, loop=env._obs.observation.game_loop, map_name='10gen_zerg'))


def feedback(env, loop):
    return dict(from_game_loop=loop, response_mapping='exact', actors={f'u{i}': dict(submission='accepted', requested_action_id=str(a),
                command={'ability_id': 16}, response={'code': 1}) for i, a in enumerate(env.actions)})


def read(path):
    return json.loads(path.read_text())


def fixture_entry(manager, base, _payload):
    """A deterministic test extension: score a new combination on an isolated cache."""
    coord = copy.copy(manager.coordinator)
    coord._cache = {}
    _, _, _, _, ctx = manager._pending_commit
    for old in manager.coordinator._cache.values():
        if len(old.plans) != 1 or len(old.plans[0].controlled_actors) != 1:
            continue
        p = old.plans[0]
        actor, = p.controlled_actors
        for a in (2, 3, 4, 5):
            actions = {**base['actions'], actor: a}
            if any(e.actions == actions for e in manager.coordinator._cache.values()):
                continue
            plan = copy.deepcopy(p)
            plan.current_actions[actor] = a
            e = coord.evaluate_once(ctx, [plan], actions, manager.memory, [1], len(actions))
            if e is not None:
                return e
    raise AssertionError('offline fixture lacks a novel complete plan')


class ForkTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.args = runner.parser().parse_args(['run', '--manifest', 'unused', '--output-dir', 'unused',
                    '--method', 'adaptive-skill', '--skill-version', 'tactical_v1', '--execution-mode', 'micro8-all'])
        self.manifest = manifest()
        self.manifest['episodes'] = self.manifest['episodes'][:1]
        for mock in (patch('bridge.policy.build_request', payload), patch('bridge.environment.ReplayableEnv', TeamEnv),
                     patch('bridge.feedback.action_feedback', feedback), patch.object(runner, 'score_snapshot',
                     return_value={'score_details': {f: {'life': 10} for f in ('total_damage_dealt', 'total_damage_taken')}})):
            mock.start()
            self.addCleanup(mock.stop)
        self.run_arm('source')
        src = self.root / 'source'
        self.assertFalse((src / 'runner/fork_receipt.json').exists())
        path = src / 'manifest.json'
        runner.write_json(path, self.manifest)
        r = json.loads((src / 'shadow/shadow.jsonl').read_text().splitlines()[1])
        base, alt = r['candidates'][:2]
        def arm(c):
            return dict(candidate_id=c['candidate_id'], source_plan_id=c['source_plan_id'], actions=c['proposed_actions'])
        a, b = arm(base), arm(alt)
        self.spec = dict(pair=dict(A=a, B=b, run_id=r['run_id'], episode_group=r['run_id'] + ':1', episode_id='1',
                    manifest=str(path), snapshot_id=r['snapshot_id'], game_loop=r['game_loop'], roster=r['scene_config_id'],
                    seed=r['seed'], changed={u: dict(unit=r['global_features']['unit_types'][u], A=v, B=b['actions'][u])
                        for u, v in a['actions'].items() if v != b['actions'][u]}),
                    source_hash=r['source_hash'], collection_context=r['execution_context_hash'])

    def run_arm(self, name, arm=None):
        root = self.root / name
        with patch.dict(os.environ, dict(ZERG_SHADOW_DIR=str(root / 'shadow'), ZERG_SHADOW_ROSTER='offline-roster',
                                         SMAC_DAMAGE_MAP_DIR='')):
            os.environ.pop('ZERG_FORK_SPEC', None)
            if arm:
                path = self.root / 'spec.json'
                runner.write_json(path, dict(self.spec, arm=arm))
                os.environ['ZERG_FORK_SPEC'] = str(path)
            asyncio.run(runner.evaluate(self.args, self.manifest, 'adaptive-skill', root / 'runner'))
        return root

    def test_one_shot_uses_complete_plan_and_normal_commit(self):
        for arm in ('A', 'B'):
            root = self.run_arm(arm, arm)
            r = read(root / 'runner/fork_receipt.json')
            self.assertNotIn('rule_hash', r)
            self.assertEqual((r['status'], r['anchor_commits'], r['prefix_verified'], r['engine_receipt']),
                             ('complete', 1, True, 'accepted'))
            chosen = self.spec['pair'][arm]
            self.assertEqual((r['candidate_id'], r['final_actions']), (chosen['candidate_id'], chosen['actions']))
            rows = [json.loads(s) for s in (root / 'shadow/shadow.jsonl').read_text().splitlines()]
            self.assertEqual([s['snapshot_id'] for s in rows], [1, 2, 3])
            self.assertEqual([s['selected_candidate'] != s['teacher_choice'] for s in rows], [False, arm == 'B', False])
            self.assertEqual(rows[1]['final_actions'], chosen['actions'])
            if arm == 'B':
                steps = [json.loads(s) for s in (root / 'runner/events.jsonl').read_text().splitlines()]
                at = next(e for e in steps if e['event'] == 'skill_step' and e['step'] == 1)
                self.assertTrue(r['validated_plan_ids'])
                self.assertEqual([p['id'] for p in at['chosen_plans']], r['validated_plan_ids'])
                self.assertEqual({u: row['action'] for u, row in at['units'].items()}, chosen['actions'])

    def test_prefix_and_source_rejections(self):
        def drift(*args):
            p = payload(*args)
            p['state']['objective'] = 'changed'
            return p
        with patch('bridge.policy.build_request', drift), self.assertRaisesRegex(ValueError, 'observed prefix'):
            self.run_arm('prefix', 'B')
        self.spec['collection_context'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'context mismatch'):
            self.run_arm('source-fail', 'A')
        for name in ('prefix', 'source-fail'):
            r = read(self.root / name / 'runner/fork_receipt.json')
            self.assertEqual((r['status'], r['anchor_commits'], r['prefix_verified']), ('failed', 0, False))

    def test_engine_unknown_and_rejected(self):
        with patch('bridge.feedback.action_feedback', return_value={}):
            root = self.run_arm('unmapped', 'B')
        self.assertEqual(read(root / 'runner/fork_receipt.json')['engine_receipt'], 'unknown')
        def reject(*args):
            f = feedback(*args)
            f['actors']['u0'].update(submission='rejected', response={'code': 3})
            return f
        with patch('bridge.feedback.action_feedback', reject), self.assertRaisesRegex(ValueError, 'engine rejected'):
            self.run_arm('reject', 'B')
        r = read(self.root / 'reject/runner/fork_receipt.json')
        self.assertEqual((r['status'], r['anchor_commits'], r['engine_receipt']), ('failed', 1, 'rejected'))

    def test_no_branch_never_falls_back(self):
        replace = Fork.replace
        def no_cache(hook, controller, *args):
            controller.manager.coordinator._cache.clear()
            return replace(hook, controller, *args)
        with patch.object(Fork, 'replace', no_cache), self.assertRaisesRegex(ValueError, 'branch unavailable'):
            self.run_arm('missing', 'B')
        r = read(self.root / 'missing/runner/fork_receipt.json')
        self.assertEqual((r['status'], r['anchor_commits'], r['final_actions']), ('failed', 0, None))

    def test_initial_and_prefix_actions_are_independently_checked(self):
        with patch.object(runner, 'initial_state_digest', return_value='different'), self.assertRaisesRegex(ValueError, 'initial state'):
            self.run_arm('initial', 'B')
        path = self.root / 'source/runner/events.jsonl'
        history = [json.loads(s) for s in path.read_text().splitlines()]
        next(e for e in history if e['event'] == 'step')['actions'][0] = 0
        path.write_text(''.join(json.dumps(e) + '\n' for e in history))
        with self.assertRaisesRegex(ValueError, 'rule actions differ at step 0'):
            self.run_arm('actions', 'B')
        for name in ('initial', 'actions'):
            self.assertEqual(read(self.root / name / 'runner/fork_receipt.json')['anchor_commits'], 0)

    def test_stale_micro_grant_removed_before_commit(self):
        replace, committed = Fork.replace, Fork.committed
        def plant_stale(hook, controller, *args):
            ctx = controller.manager._pending_commit[-1]
            ctx.micro_approaches = [dict(actor='u1', actor_tag='1', phase='stale')]
            controller.manager.last_micro_intents = list(ctx.micro_approaches)
            return replace(hook, controller, *args)
        def inspect(hook, manager, actions):
            self.assertFalse(any(grant['phase'] == 'stale' for grant in manager.last_micro_intents))
            committed(hook, manager, actions)
        with patch.object(Fork, 'replace', plant_stale), patch.object(Fork, 'committed', inspect):
            self.run_arm('stale', 'B')

    def scan_c(self):
        # The extension is not part of the original archive or the old A/B API.
        rule = SimpleNamespace(RULE_ID='test_new_rule', rule_hash=lambda: 'a' * 64, make_entry=fixture_entry)
        mock = patch('decision_tree.zerg.rules_v1', rule, create=True)
        mock.start()
        self.addCleanup(mock.stop)
        self.spec['rule_hash'] = rule.rule_hash()
        step = Fork.step
        def peek(hook, i, loop, ctrl, obs, env):
            actions, base = step(hook, i, loop, ctrl, obs, env)
            if i == hook.anchor:
                self.spec['pair']['C'] = hook.peek_c(ctrl, base, obs, env, rule.rule_hash())
            return actions, base
        with patch.object(Fork, 'step', peek):
            root = self.run_arm('scan', 'A')
        self.assertNotIn('rule_hash', read(root / 'runner/fork_receipt.json'))
        row = json.loads((root / 'shadow/shadow.jsonl').read_text().splitlines()[1])
        self.assertNotIn(rule.RULE_ID, [c['candidate_id'] for c in row['candidates']])
        return rule

    def test_c_peek_then_independent_commit(self):
        rule = self.scan_c()
        card = self.spec['pair']['C']
        root = self.run_arm('execute', 'C')
        r = read(root / 'runner/fork_receipt.json')
        self.assertEqual((r['kind'], r['status'], r['anchor_commits'], r['prefix_verified'], r['engine_receipt']),
                         ('zerg_one_shot_fork_receipt_v1', 'complete', 1, True, 'accepted'))
        self.assertEqual((r['candidate_id'], r['source_plan_id'], r['rule_hash'], r['final_actions']),
                         (rule.RULE_ID, rule.RULE_ID, rule.rule_hash(), card['actions']))
        rows = [json.loads(s) for s in (root / 'shadow/shadow.jsonl').read_text().splitlines()]
        self.assertEqual([row['teacher_choice'] != row['selected_candidate'] for row in rows], [False, True, False])
        candidate, = (c for c in rows[1]['candidates'] if c['candidate_id'] == rule.RULE_ID)
        self.assertEqual((candidate['origin'], candidate['rule_hash'], candidate['proposed_actions']),
                         ('one_shot_fork_C', rule.rule_hash(), card['actions']))
        events = [json.loads(s) for s in (root / 'runner/events.jsonl').read_text().splitlines()]
        at = next(e for e in events if e['event'] == 'skill_step' and e['step'] == 1)
        self.assertTrue(r['validated_plan_ids'])
        self.assertEqual([p['id'] for p in at['chosen_plans']], r['validated_plan_ids'])
        self.assertEqual({u: row['action'] for u, row in at['units'].items()}, card['actions'])

    def test_c_fails_closed(self):
        rule = self.scan_c()
        def bad_entry(kind):
            def proposal(m, b, p):
                if kind == 'duplicate':
                    return next(iter(m.coordinator._cache.values()))
                if kind == 'abstained':
                    return None
                e = fixture_entry(m, b, p)
                if kind == 'partial':
                    del e.actions['u0']
                if kind == 'boolean':
                    e.actions['u0'] = True
                if kind == 'empty':
                    e.plans = []
                if kind == 'leaked':
                    m.coordinator._cache['leaked'] = e
                return e
            return proposal
        for kind, error in [('duplicate', 'already in original'), ('abstained', 'abstained'),
                            ('partial', 'whole team'), ('boolean', 'integer'), ('empty', 'empty'),
                            ('leaked', 'changed original cache')]:
            with self.subTest(kind=kind), patch.object(rule, 'make_entry', bad_entry(kind)), self.assertRaisesRegex(ValueError, error):
                self.run_arm(kind, 'C')
            r = read(self.root / kind / 'runner/fork_receipt.json')
            self.assertEqual((r['status'], r['anchor_commits'], r['final_actions']), ('failed', 0, None))
        with patch.object(rule, 'rule_hash', return_value='b' * 64), self.assertRaisesRegex(ValueError, 'rule_hash mismatch'):
            self.run_arm('hash', 'C')
        self.spec['pair']['C']['source_plan_id'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'frozen declaration'):
            self.run_arm('id', 'C')

    def test_c_hidden_cache_duplicate_and_changed_full_map(self):
        rule = self.scan_c()
        propose = Fork._propose_c
        def with_hidden(hook, ctrl, base, obs, env, h):
            entry = fixture_entry(ctrl.manager, base, obs)
            # This branch exists in the raw search but neither top-k summary.
            self.assertTrue(all(entry.actions != a['actions'] for s in (base, hook.summary) for a in s['top_alternates']))
            ctrl.manager.coordinator._cache['hidden'] = entry
            with patch.object(rule, 'make_entry', return_value=entry):
                return propose(hook, ctrl, base, obs, env, h)
        with patch.object(Fork, '_propose_c', with_hidden), self.assertRaisesRegex(ValueError, 'already in original'):
            self.run_arm('hidden', 'C')
        pair = self.spec['pair']
        old = pair['C']['actions']
        pair['C']['actions'] = next(v for a in (1, 2, 3, 4, 5) if
                                    (v := {**old, 'u2': a}) not in (old, pair['A']['actions'], pair['B']['actions']))
        with self.assertRaisesRegex(ValueError, 'frozen declaration'):
            self.run_arm('drift', 'C')
