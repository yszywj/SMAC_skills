import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from decision_tree.scripts.linux import run_zerg_abc as abc


class ABCLauncherTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.out = self.root / 'run'
        self.input = self.root / 'input.json'
        self.split_path = self.root / 'split.json'
        self.split = {}
        self.card = dict(rule_id='test_card', rule_hash='a'*64)
        trial = dict(rule_id='test_card', parent_source='src', parent_context='ctx', decision='fixed gate')
        trial_path = self.root / 'trial.json'
        trial_path.write_text(json.dumps(trial))
        self.plan = dict(kind='zerg_fork_review_plan_v1', abc_protocol=abc.ABC, horizon_steps=1,
                         source_hash='src', collection_context='ctx', fit=str(self.root), pairs=[],
                         rule_card=trial, rule_card_hash=abc.digest(trial), **self.card)
        for i in (1, 2):
            group, worker = f'run{i}:1', f'worker{i}'
            self.split[group] = dict(partition='train', seed=i, roster='same', batch=str(self.root), worker=worker)
            self.plan['pairs'].append(dict(run_id=f'run{i}', episode_id='1', episode_group=group, roster='same',
                seed=i, snapshot_id=1, game_loop=12, manifest=str(self.root / worker / 'manifest.json'),
                A=dict(candidate_id='base', source_plan_id='p0', actions={'u0': 3}),
                B=dict(candidate_id='alt', source_plan_id='p1', actions={'u0': 4}),
                C=dict(candidate_id='test_card', source_plan_id='test_card', actions={'u0': 5})))
        self.save_input()
        self.argv = ['--plan', str(self.input), '--out', str(self.out), '--anchor-index', '2']
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(abc, 'rule_identity', return_value=self.card))
        stack.enter_context(patch.object(abc, 'CARD_PATH', trial_path))
        stack.enter_context(patch.object(abc.ab, 'historical_config', return_value=({}, {})))
        self.validate = stack.enter_context(patch.object(abc.ab, 'validate_plan'))
        self.stdout = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))

    def save_input(self):
        self.plan['split_hash'] = abc.digest(self.split)
        self.split_path.write_text(json.dumps(self.split))
        self.input.write_text(json.dumps(self.plan, indent=1))

    def test_freeze_specs_and_separate_modes(self):
        with patch.object(abc.ab, 'run_worker', side_effect=AssertionError('started game')):
            abc.main(self.argv + ['--dry-run'])
        self.validate.assert_called_once_with(self.plan, {}, {})
        self.assertEqual((self.out / 'plan.json').read_bytes(), self.input.read_bytes())
        design = abc.ab.read_json(self.out / 'design.json')
        self.assertEqual((design['workers'], design['pair_indices'], design['arms']), (48, [2], list('ABC')))
        self.assertFalse((self.out / 'pair-001').exists())
        for arm in 'ABC':
            spec = abc.ab.read_json(self.out / 'pair-002' / arm / 'fork_spec.json')
            self.assertEqual(spec['pair'], self.plan['pairs'][1])
            self.assertEqual(spec.get('rule_hash'), self.card['rule_hash'] if arm=='C' else None)
        for argv in (self.argv, ['--compare', str(self.out), '--workers', '1'],
                     ['--compare', str(self.out), '--plan', str(self.input)]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                abc.main(argv)
        result = abc.compare(self.out)
        self.assertEqual(result['accepted_triplets'], 0)
        self.assertEqual(set(result['excluded'][0]['missing_or_invalid']), set('ABC'))

    def test_validation_membership_checked_before_any_historical_preflight(self):
        self.split['run1:1']['partition'] = 'validation'  # even if smoke selects only #2
        self.save_input()
        with patch.object(abc.ab, 'historical_config', side_effect=AssertionError), self.assertRaises(SystemExit):
            abc.main(self.argv + ['--dry-run'])
        self.validate.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_refuse_stale_hash_relocated_manifest_or_incomplete_or_duplicate_c(self):
        p = self.plan['pairs'][0]
        for target, fields in [(self.plan, {'rule_hash': 'old'}), (p, {'manifest': 'validation/manifest.json'}),
                               (p['C'], {'source_plan_id': 'p2'}), (p['C'], {'actions': {'u0': 4}}),
                               (p['C'], {'actions': {'u0': True}}), (p['C'], {'actions': {'u1': 6}})]:
            with self.subTest(fields=fields), patch.dict(target, fields), self.assertRaises(ValueError):
                abc.validate_frozen_plan(self.plan)

    def test_bad_options_and_legacy_cli_do_not_launch(self):
        for extra in (['--anchor-index', '0'], ['--anchor-index', '3'], ['--workers', '0']):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                abc.main(self.argv + extra)
        with self.assertRaises(SystemExit):
            abc.ab.main(['--out', str(self.out), '--plan', str(self.input)])
        self.validate.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_card_and_screening_cannot_be_replaced_even_with_matching_embedded_hash(self):
        card = dict(self.plan['rule_card'], decision='weaker gate')
        for change in [dict(rule_card=card, rule_card_hash=abc.digest(card)), dict(screening={'min_complete': 1})]:
            with self.subTest(change=change), patch.dict(self.plan, change), self.assertRaises(ValueError):
                abc.validate_frozen_plan(self.plan)

    def make_run(self):
        # Only the original-plan preflight and game process are stubbed. All
        # on-disk runner/receipt/shadow validation runs during launch AND compare.
        p = self.plan['pairs'][1]
        original = Path(p['manifest']).parent
        (original / 'runner').mkdir(parents=True)
        identity = dict(scenario_sha256='scenario', initial_state_sha256='initial')
        context = dict(source_fingerprint='src', scene_config_id='same', execution_context_hash='ctx')
        meta = dict(**abc.ab.PROTOCOL, versions={}, env_args={}, map_file=str(abc.ab.MAP_FILE),
            source_hashes={abc.ab.MAP_FILE.relative_to(abc.ab.ROOT).as_posix(): 'map'},
            macro_loops=8, engine_step_loops=1, manifest_sha256='manifest', damage_by_type_enabled=True,
            skill_parameters={}, skill_source_hashes={}, native_teacher_control='off')
        write = abc.ab.write_json
        write(original / 'manifest.json', {'sha256': 'manifest'})
        write(original / 'runner/run.json', dict(**meta, zerg_shadow=dict(run_id=p['run_id'], **context)))
        write(original / 'runner/episodes.json', [identity])

        def fake_worker(work, pair, spec_path):
            arm = work.name
            spec = abc.ab.read_json(spec_path)
            self.assertEqual(spec['arm'], arm)
            self.assertEqual(spec['pair'], pair)
            runner = work / 'runner'
            runner.mkdir()
            (work / 'shadow').mkdir()
            replay_id = f'new-{arm}'
            sm = dict(**context, run_id=replay_id)
            write(runner / 'run.json', dict(**meta, zerg_shadow=sm))
            outcome, reward = {'A': ('win', 6), 'B': ('loss', -2), 'C': ('win', 9)}[arm]
            write(runner / 'episodes.json', [dict(**identity, outcome=outcome, reward=reward, steps=1, zerg_shadow_decisions=1)])
            write(runner / 'summary.json', dict(**abc.ab.PROTOCOL, status='complete', completed_episodes=1,
                  expected_episodes=1, manifest_sha256='manifest', zerg_shadow=dict(**sm,
                      stats=dict(decisions_seen=1, decisions_recorded=1, model_errors=0))))
            row = dict(run_id=replay_id, episode_id='1', snapshot_id=1, game_loop=12, source_hash='src',
                seed=2, execution_context_hash='ctx', scene_config_id='same', teacher_choice='base',
                selected_candidate=p[arm]['candidate_id'], final_actions=p[arm]['actions'], outcome=outcome,
                **{'return': reward})
            receipt = dict(kind='zerg_one_shot_fork_receipt_v1', arm=arm, status='complete', error=None,
                prefix_verified=True, anchor_commits=1, engine_receipt='accepted', episode_group='run2:1',
                roster='same', seed=2, snapshot_id=1, game_loop=12, source_hash='src', collection_context='ctx',
                candidate_id=p[arm]['candidate_id'], source_plan_id=p[arm]['source_plan_id'],
                final_actions=p[arm]['actions'], **identity)
            if arm == 'C':
                receipt.update(validated_plan_ids=['native'], rule_hash=self.card['rule_hash'])
                row['candidates'] = [dict(candidate_id='test_card', candidate_family='joint_plan', legal=True,
                    source_plan_id='test_card', proposed_actions=p['C']['actions'], origin='one_shot_fork_C',
                    rule_hash=self.card['rule_hash'])]
            (work / 'shadow/shadow.jsonl').write_text(json.dumps(row) + '\n')
            write(runner / 'fork_receipt.json', receipt)

        with patch.object(abc.ab, 'run_worker', side_effect=fake_worker) as spawn:
            abc.main(self.argv + ['--workers', '2'])
        self.assertEqual(spawn.call_count, 3)

    def test_descriptive_complete_case_contrasts_and_win_flips(self):
        self.make_run()
        self.stdout.truncate(0)
        self.stdout.seek(0)
        with patch.object(abc.ab, 'run_worker', side_effect=AssertionError), patch.object(abc.ab, 'validate_plan', side_effect=AssertionError):
            abc.main(['--compare', str(self.out)])
        report = json.loads(self.stdout.getvalue())
        self.assertEqual((report['planned_triplets'], report['accepted_triplets'], report['missing_triplets']), (1, 1, 0))
        for key, delta, flip in [('B-A', -8, -1), ('C-A', 3, 0), ('C-B', 11, 1)]:
            self.assertEqual(report['triplets'][0]['deltas'][key], dict(win_delta=flip, return_delta=delta))
            self.assertEqual(report['contrasts'][key], dict(mean_return_delta=delta, more_wins=int(flip>0), fewer_wins=int(flip<0)))

    def test_no_partial_deltas_for_missing_or_corrupt_evidence(self):
        self.make_run()
        triplet = self.out / 'pair-002'
        for arm, name, edit in [
                ('A', 'runner/fork_receipt.json', {'engine_receipt':'unknown'}),
                ('B', 'runner/fork_receipt.json', {'engine_receipt':'rejected'}),
                ('B', 'runner/fork_receipt.json', {'prefix_verified':1}),
                ('C', 'fork_spec.json', {'rule_hash':'other'}),
                ('C', 'runner/fork_receipt.json', {'initial_state_sha256':'other'}),
                ('C', 'runner/fork_receipt.json', {'rule_hash':'other'}),
                ('C', 'runner/fork_receipt.json', {'source_hash':'other'}),
                ('C', 'runner/fork_receipt.json', {'snapshot_id':2}),
                ('C', 'runner/fork_receipt.json', {'validated_plan_ids':[]}),
                ('C', 'runner/fork_receipt.json', {'final_actions':{'u0':4}}),
                ('C', 'shadow/shadow.jsonl', {'selected_candidate':'base'}),
                ('C', 'shadow/shadow.jsonl', {'candidates':[]}),
                ('C', 'runner/summary.json', None)]:
            with self.subTest(arm=arm, name=name, edit=edit):
                path = triplet / arm / name
                saved = path.read_text()
                if edit is None:
                    path.unlink()
                else:
                    path.write_text(json.dumps({**json.loads(saved), **edit}))
                report = abc.compare(self.out)
                self.assertEqual(report['missing_triplets'], 1)
                self.assertEqual(report['triplets'], [])
                self.assertEqual(set(report['excluded'][0]['missing_or_invalid']), {arm})
                self.assertIsNone(report['contrasts']['B-A']['mean_return_delta'])
                path.write_text(saved)

    def test_comparison_rejects_changed_plan_or_index(self):
        abc.main(self.argv + ['--dry-run'])
        design_path = self.out / 'design.json'
        design = abc.ab.read_json(design_path)
        for change in [dict(plan_hash='old'), dict(pair_indices=[0]), dict(pair_indices=[2,2])]:
            design_path.write_text(json.dumps({**design, **change}))
            with self.subTest(change=change), self.assertRaises(ValueError):
                abc.compare(self.out)

    def test_reused_replay_identity_excludes_entire_triplet(self):
        self.make_run()
        work = self.out / 'pair-002' / 'C'
        for filename in ('runner/run.json', 'runner/summary.json', 'shadow/shadow.jsonl'):
            path = work / filename
            data = abc.ab.read_json(path)
            data.get('zerg_shadow', data)['run_id'] = 'new-A'
            path.write_text(json.dumps(data))
        with self.assertRaises(SystemExit) as failure:
            abc.main(['--compare', str(self.out)])
        self.assertEqual(failure.exception.code, 1)
        self.assertEqual(abc.compare(self.out)['excluded'][0]['missing_or_invalid'], {'triplet': 'reused replay run id'})


if __name__ == '__main__':
    unittest.main()
