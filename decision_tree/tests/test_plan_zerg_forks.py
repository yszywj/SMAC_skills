from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from decision_tree.scripts.linux import plan_zerg_forks as planner, zerg_fork_preflight as archive


class RoundSelectionTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.fit = Path(tmp.name)
        ctx = dict(projection='joint_team_skill_int_action_changes_v1', collection_context='ctx')
        model = dict(source_fingerprint='source', target='imitation', scorer='joint_scorer', model_id='id',
                     execution_context_hash=planner.digest(ctx), training_configuration={'validation_episodes': []})
        self.save('model.json', model)
        self.save('report.json', dict(offline_feature_context=ctx, model_id='id', model_sha256=planner.digest(model)))
        self.split = {}
        for seed in range(10):
            run = f'run{seed}'
            roster = 'r1' if seed < 5 else 'r2'
            self.split[run + ':1'] = dict(partition='train', batch=str(self.fit), worker=run, roster=roster, seed=seed)
            parent = self.fit / run / 'shadow'
            parent.mkdir(parents=True)
            candidates = [dict(candidate_id=k, source_plan_id=k, legal=True, candidate_family='joint_plan',
                               current_actions={}, proposed_actions={'u0': i}) for i, k in enumerate(('A', 'B'))]
            # No outcome fields or runner result files are needed to rank the candidates.
            row = dict(run_id=run, episode_id='1', scene_config_id=roster, seed=seed, source_hash='source',
                       execution_context_hash='ctx', teacher_choice='A', final_actions={'u0': 0}, candidates=candidates,
                       global_features={'team': {}, 'skill': {}, 'unit_types': {'u0': 'baneling'}})
            (parent / 'shadow.jsonl').write_text(''.join(json.dumps(dict(row, snapshot_id=i, game_loop=i*8)) + '\n'
                                                        for i in (1, 2)))
        self.save('split.json', self.split)
        scorer = self.enterContext(patch.object(planner, 'ScorerModel')).return_value
        scorer.estimate.return_value = [{'value': 0.5}]
        scorer.tie_tolerance = 0.001

    def save(self, name, value):
        path = self.fit / name
        path.write_text(json.dumps(value))
        return path

    def test_repeated_cli_exclusions_and_ranked_fallback(self):
        original = planner.build_plan(self.fit)
        blocked = {p['episode_group'] for p in original['pairs']}
        paths = [self.save(f'prior{i}.json', {**original, 'pairs': original['pairs'][i::2]}) for i in (0, 1)]
        calls = []
        def verify(row, plan):
            self.assertNotIn(row['episode_group'], blocked)
            calls.append(row)
            if len(calls) == 1:
                raise ValueError('Zerg fork: alternative lacks real plan objects')
        args = ['--fit', str(self.fit), '--seed', '27', '--exclude-plan', str(paths[0]), '--exclude-plan', str(paths[1])]
        plans = []
        for i in (1, 2):
            calls.clear()
            output = self.fit / f'new{i}.json'
            with patch.object(archive, 'verify_archived_b', side_effect=verify), redirect_stdout(io.StringIO()):
                planner.main(args + ['--out', str(output)])
            plan = json.loads(output.read_text())
            self.assertEqual(plan['pairs'], calls[1:])
            self.assertNotIn(calls[0], plan['pairs'])
            self.assertEqual(len({p['episode_group'] for p in plan['pairs']}), 4)
            self.assertEqual(Counter(p['roster'] for p in plan['pairs']), {'r1': 2, 'r2': 2})
            self.assertEqual(plan['archive_preflight']['excluded_groups'], 4)
            self.assertEqual(plan['archive_preflight']['rejected'], 1)
            plans.append(plan)
        self.assertEqual(*plans)

    def test_quota_failure_no_partial_output(self):
        # Include a roster that has *zero* remaining train episodes, not just a short pool.
        prior = planner.build_plan(self.fit)
        prior['pairs'] = [dict(roster=ep['roster'], seed=ep['seed'], episode_group=g, run_id=g.split(':')[0], episode_id='1')
                          for g, ep in self.split.items() if ep['roster'] == 'r2']
        path = self.save('excluded.json', prior)
        out = self.fit / 'must-not-exist.json'
        with patch.object(archive, 'verify_archived_b'), self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            planner.main(['--fit', str(self.fit), '--exclude-plan', str(path), '--out', str(out)])
        self.assertFalse(out.exists())

    def test_legacy_repair_and_foreign_plan(self):
        with patch.object(archive, 'verify_archived_b', side_effect=AssertionError('unexpected preflight')):
            old = planner.build_plan(self.fit)
            path = self.save('first.json', old)
            repaired = planner.build_plan(self.fit, 7, repair_from=path, replace_pairs=[2])
        self.assertEqual(repaired['pairs'][0], old['pairs'][0])
        self.assertEqual(repaired['pairs'][2:], old['pairs'][2:])
        self.assertNotIn(repaired['pairs'][1]['episode_group'], {p['episode_group'] for p in old['pairs']})
        self.assertNotIn('archive_preflight', repaired)
        path = self.save('bad.json', {**old, 'model_id': 'foreign'})
        with self.assertRaisesRegex(ValueError, 'different fit'):
            planner.build_plan(self.fit, exclude_plans=[path])


class ReplayTest(unittest.TestCase):
    def test_masks_prefix_micro_and_single_native_anchor_commit(self):
        fork, ctrl = Mock(anchor=1), Mock(active=False)
        calls = []
        def step(idx, loop, controller, payload, env):
            calls.append(('step', idx))
            self.assertEqual(env.n_agents, 10)
            self.assertEqual([i for i, ok in enumerate(env.get_avail_agent_actions(0)) if ok], [1, 7])
            self.assertEqual([i for i, ok in enumerate(env.get_avail_agent_actions(1)) if ok], [0])
            return [idx], {}
        fork.step.side_effect = step
        fork.committed.side_effect = lambda manager, actions: calls.append(('confirmed', actions))
        ctrl.commit_final_actions.side_effect = lambda actions: calls.append(('commit', actions))
        ctrl.manager.record_micro_action.side_effect = lambda actor, action: calls.append((actor, action))
        obs = dict(event='skill_observation', episode=1, step=0,
                   payload=dict(state={'game_loop': 2}, questions={'u0': {'criteria': {'1': 'stop', '7': 'e1'}}}))
        fb = dict(event='step', episode=1, step=1, feedback={'micro_submissions': [
            dict(sc2_response=[1], actor='u0', submitted_action_id=7),
            dict(sc2_response=[2], actor='u2', submitted_action_id=5)]})
        archive.replay_prefix(fork, ctrl, [obs, fb, {**obs, 'step': 1}, {}])
        self.assertEqual(calls, [('step', 0), ('commit', [0]), ('u0', 7), ('step', 1), ('commit', [1]), ('confirmed', [1])])
        with self.assertRaisesRegex(ValueError, 'missing prefix step feedback'):
            archive.replay_prefix(fork, ctrl, [obs, {**obs, 'step': 1}])
        with self.assertRaisesRegex(ValueError, 'before B commit'):
            archive.replay_prefix(fork, ctrl, [obs, fb])
