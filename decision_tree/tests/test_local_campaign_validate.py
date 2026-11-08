"""Synthetic cohort and policy promotion cases; no simulator or experiment records."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from decision_tree.local.authored import build
from decision_tree.local.runtime import digest
from decision_tree.scripts import local_campaign_validate as validation
from decision_tree.scripts.local_campaign_validate import summarize, validate_cohort
from decision_tree.tests.test_local_evaluate import Fixture, save
from sc2_skill_adapter.bridge.config import load_config
from skills.local_v1 import FEATURE_NAMES


def cohort():
    env = load_config(10, 'protoss')
    env['unit_balance'] = 'native'
    scenes = []
    for index in range(40):
        config = {'synthetic_scene': index}
        scenes.append({'index': index, 'sc2_seed': 7000 + index,
                       'episode_config': config, 'sha256': digest(config)})
    manifest = {'version': 1, 'seed': 7000, 'env_args': env, 'episodes': scenes}
    manifest['sha256'] = digest(manifest)
    validation = [scene['sha256'] for scene in scenes[::4]]
    split = {'schema_version': 1, 'kind': 'local_scene_split',
             'training_group_ids': [s['sha256'] for s in scenes if s['sha256'] not in validation],
             'validation_group_ids': validation}
    return manifest, split


def observations():
    records = []
    for i in range(10):
        base = {'return': 10.0, 'outcome': 'draw', 'steps': 3,
                'proposals': 0, 'vetoes': 0, 'interventions': 0}
        records.append({'group_id': digest(i), 'sc2_seed': 7000 + i, 'base_episode': i + 1,
                        'candidate': deepcopy(base), 'incumbent': base, 'gain': 0.0})
    return records


def set_gain(record, delta):
    record['candidate']['return'] = record['incumbent']['return'] + delta
    record['gain'] = record['candidate']['return'] - record['incumbent']['return']


class AdoptionTests(unittest.TestCase):
    def test_tiny_floating_residue_retains_incumbent(self):
        records = observations()
        for record in records:
            set_gain(record, 1e-14)
        self.assertEqual(summarize(records)['decision'], 'retain_incumbent')

    def test_more_losses_blocks_positive_mean_gain(self):
        records = observations()
        set_gain(records[2], 4)
        records[2]['candidate']['outcome'] = 'loss'
        self.assertEqual(summarize(records)['decision'], 'retain_incumbent')

    def test_fewer_wins_blocks_positive_mean_gain(self):
        records = observations()
        set_gain(records[2], 4)
        records[2]['incumbent']['outcome'] = 'win'
        self.assertEqual(summarize(records)['decision'], 'retain_incumbent')

    def test_positive_mean_with_equal_outcomes_adopts_despite_one_harmed_scene(self):
        records = observations()
        set_gain(records[0], 3)
        set_gain(records[1], -1)
        self.assertEqual(summarize(records)['decision'], 'adopt_candidate')

    def test_more_wins_with_lower_return_retains_incumbent(self):
        records = observations()
        records[0]['candidate']['outcome'] = 'win'
        set_gain(records[0], -1)
        self.assertEqual(summarize(records)['decision'], 'retain_incumbent')

    def test_partial_validation_cannot_emit_a_decision(self):
        with self.assertRaises(ValueError):
            summarize(observations()[:-1])


class CohortTests(unittest.TestCase):
    def test_pairs_use_original_seed_and_one_based_base_ordinal(self):
        manifest, split = cohort()
        selected = validate_cohort(manifest, split)
        self.assertEqual([(number, scene['sc2_seed']) for number, scene in selected],
                         [(i + 1, 7000 + i) for i in range(0, 40, 4)])

    def test_incomplete_manifest_rejected_even_if_its_hash_is_current(self):
        manifest, split = cohort()
        manifest['episodes'].pop()
        del manifest['sha256']
        manifest['sha256'] = digest(manifest)
        with self.assertRaises(ValueError):
            validate_cohort(manifest, split)


class AttemptTests(unittest.TestCase):
    def test_resume_reuses_successful_attempt_and_preserves_failed_predecessor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root/'manifest.json'
            manifest_path.write_text(json.dumps(cohort()[0]), encoding='utf-8')
            policy_path = root/'policy.json'
            policy_path.write_text(json.dumps({'schema_version': 1, 'kind': 'local_patch_policy'}), encoding='utf-8')
            first = root/'candidate.attempt-01'
            second = root/'candidate.attempt-02'
            first.mkdir()
            second.mkdir()
            failure = {'status': 'failed', 'error': 'ConnectError: no websocket'}
            (first/'summary.json').write_text(json.dumps(failure), encoding='utf-8')
            success = {'path': str(second)}
            with patch.object(validation.recovery, 'successful', side_effect=[ValueError('failed'), success]), \
                    patch.object(validation.recovery, 'run_attempt') as launched:
                got = validation.audited_attempts({}, manifest_path, policy_path, root/'candidate')
            launched.assert_not_called()
            self.assertEqual(got, success)
            self.assertEqual(json.loads((first/'summary.json').read_text(encoding='utf-8')), failure)

class SplitTests(unittest.TestCase):
    def test_holdout_cannot_replace_missing_member_with_a_duplicate(self):
        manifest, split = cohort()
        split['validation_group_ids'][-1] = split['validation_group_ids'][0]
        with self.assertRaises(ValueError):
            validate_cohort(manifest, split)

    def test_split_rejects_train_validation_overlap(self):
        manifest, split = cohort()
        split['training_group_ids'][-1] = split['validation_group_ids'][0]
        with self.assertRaises(ValueError):
            validate_cohort(manifest, split)


class DryRunTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.f = Fixture(Path(tmp.name))
        self.manifest, self.split = cohort()
        self.split_path = self.f.folder/'split.json'
        save(self.split_path, self.split)
        self.candidate = self.f.folder/'candidate'
        self.candidate.mkdir()
        self.spec = {'schema_version': 1, 'kind': 'local_authored_structure',
                     'families': {'retarget': {'leaf': True}}}
        rows = [{'group_id': s['sha256'], 'anchor_id': 'a', 'candidate_id': 'retarget:u0:7',
                 'candidate_family': 'retarget', 'features': {n: 0.0 for n in FEATURE_NAMES},
                 'base_return': 4.0, 'candidate_return': 5.0, 'gain': 1.0,
                 'available_candidate_ids': ['retarget:u0:7'], 'anchor_complete': True}
                for s in self.manifest['episodes']]
        data = {'schema_version': 1, 'kind': 'local_patch_dataset', 'audited': True,
                'context_hash': 'context', 'game_version': 'test-sc2', 'rows': rows,
                'feature_names': list(FEATURE_NAMES), 'policy': self.f.base}
        _, self.policy, report = build(data, self.spec, self.split)
        for name, value in [('policy.json', self.policy), ('report.json', report), ('structure.json', self.spec),
                            ('split.json', self.split), ('model.json', self.policy['model'])]:
            save(self.candidate/name, value)
        self.f.old = self.manifest
        self.f.protocol['manifest_sha256'] = digest(self.manifest)
        self.f.reference['manifest.json'] = self.manifest
        self.f.write()

    def dry_run(self):
        with patch.object(validation.study, 'source_hashes', return_value=self.f.sources), \
                patch.object(validation.recovery, 'successful', return_value=self.f.reference), \
                patch.object(validation.recovery, 'run_attempt', side_effect=AssertionError('game launch')):
            return validation.validate(self.f.study, self.split_path, self.candidate/'policy.json',
                                       self.f.output, dry_run=True)

    def test_dry_run_binds_authored_policy_and_selects_only_ten_holdout_seeds(self):
        result = self.dry_run()
        self.assertEqual(result['started_games'], 0)
        self.assertEqual(result['planned_model_games'], 10)
        protocol = result['protocol']
        self.assertEqual(protocol['bindings']['candidate']['structure_sha256'], digest(self.spec))
        self.assertEqual([r['sc2_seed'] for r in protocol['heldout_episodes']], list(range(7000, 7040, 4)))
        self.assertFalse(self.f.output.exists())

    def test_changed_structure_after_freeze_fails_before_simulation(self):
        save(self.candidate/'structure.json', {**self.spec, 'rationale': 'changed'})
        with self.assertRaisesRegex(ValueError, 'structure'):
            self.dry_run()

    def test_rejects_mismatched_game_build_before_simulation(self):
        self.policy['model']['game_version'] = 'different-build'
        save(self.candidate/'policy.json', self.policy)
        with self.assertRaisesRegex(ValueError, 'game version'):
            self.dry_run()

    def test_rejects_policy_that_fit_one_holdout_group(self):
        model = self.policy['model']
        leaked = model['validation_group_ids'].pop()
        model['training_group_ids'][0] = leaked
        save(self.candidate/'policy.json', self.policy)
        with self.assertRaisesRegex(ValueError, 'model training'):
            self.dry_run()

    def test_validation_partition_cannot_be_swapped_after_registration(self):
        changed = deepcopy(self.split)
        train, holdout = changed['training_group_ids'], changed['validation_group_ids']
        train[0], holdout[0] = holdout[0], train[0]
        path = self.f.folder/'different.json'
        save(path, changed)
        self.split_path = path
        with self.assertRaisesRegex(ValueError, 'predeclared'):
            self.dry_run()

    def test_full_comparison_reuses_base_and_confirmed_games_on_resume(self):
        f = self.f
        f.new = self.manifest
        base = f.fake_run(f.base, [10.0]*40, [False]*40, ['draw']*40)
        base['path'] = str(f.study/'base')
        runs = {Path(base['path']): base}

        def fake_game(protocol, manifest_path, policy_path, destination, timeout_seconds):
            single = json.loads(manifest_path.read_text(encoding='utf-8'))
            compiled = json.loads(policy_path.read_text(encoding='utf-8'))
            self.assertEqual(compiled, self.policy)
            ordinal = single['episodes'][0]['sc2_seed'] - 7000 + 1
            self.assertEqual(timeout_seconds, 120)
            f.new = single
            synthetic = f.fake_run(compiled, [11.0], [True], ['draw'])
            synthetic['path'] = str(destination)
            for event in synthetic['episodes.json'] + synthetic['events.jsonl']:
                if 'initial_state_sha256' in event:
                    event['initial_state_sha256'] = f'initial-{ordinal}'
            destination.mkdir()
            runs[destination] = synthetic
            return {'status': 'exited', 'returncode': 0}

        def fake_audit(path, manifest, policy):
            run = runs[path]
            self.assertEqual(run['manifest.json'], manifest)
            self.assertEqual(run['run.json']['local_policy']['policy'], policy)
            return run

        with patch.object(validation.study, 'source_hashes', return_value=f.sources), \
                patch.object(validation.recovery, 'successful', side_effect=fake_audit), \
                patch.object(validation.recovery, 'run_attempt', side_effect=fake_game) as launch:
            def validate():
                return validation.validate(f.study, self.split_path, self.candidate/'policy.json', f.output)
            first = validate()
            self.assertEqual(first['decision'], 'adopt_candidate')
            self.assertEqual(first['positive'], 10)
            self.assertEqual([row['base']['ordinal'] for row in first['rows']], list(range(1, 41, 4)))
            self.assertTrue(all(row['candidate']['interventions'] == 1 for row in first['rows']))
            self.assertEqual(first, validate())
            self.assertEqual(launch.call_count, 10)


if __name__ == '__main__':
    unittest.main()
