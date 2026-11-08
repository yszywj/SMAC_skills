"""Protocol/orchestration tests; audited-run boundary is mocked, no SC2 process."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from decision_tree.local import evaluate as deployment
from decision_tree.local.models import fit_gain, model_hash
from decision_tree.local.runtime import digest, load_policy
from decision_tree.local.study import DEFAULT_FLAGS
from skills.local_v1 import FEATURE_NAMES


def save(path, data):
    path.write_text(json.dumps(data), encoding='utf-8')


def manifest(groups, env):
    body = {'version': 1, 'seed': 1000, 'env_args': deepcopy(env), 'episodes': [
        {'index': i, 'sc2_seed': 1000+i, 'episode_config': {'test_scene': group},
         'sha256': digest({'test_scene': group})} for i, group in enumerate(groups)]}
    return dict(body, sha256=digest(body))


def model(context='context', gain=2):
    rows = [{'group_id': digest({'test_scene': f'train{i}'}), 'candidate_family': 'retarget',
             'features': {name: 0.0 for name in FEATURE_NAMES}, 'gain': gain} for i in range(4)]
    fitted = fit_gain(rows, list(FEATURE_NAMES), context)
    fitted.update(game_version='test-sc2', validation_group_ids=[digest({'test_scene': 'validation'})])
    return fitted


class Fixture:
    def __init__(self, folder):
        from sc2_skill_adapter.bridge.config import load_config
        from sc2_skill_adapter.bridge.runner import parser
        from sc2_skill_adapter.bridge.skill_runtime import manager_configuration
        self.folder = folder
        self.study = folder/'study'; self.study.mkdir()
        self.output = folder/'evaluation'
        self.source = folder/'source.py'; self.source.write_text('# frozen source\n', encoding='utf-8')
        self.sources = {str(self.source.resolve()): hashlib.sha256(self.source.read_bytes()).hexdigest()}
        self.base = load_policy({'schema_version': 1, 'kind': 'local_patch_policy'})
        self.policy = load_policy({**deepcopy(self.base), 'selection': {'mode': 'model'}, 'model': model()})
        self.policy_path = folder/'policy.json'
        self.old = manifest(['development'], load_config(5))
        self.new = manifest(['new1', 'new2'], self.old['env_args'])
        self.manifest_path = folder/'new_manifest.json'
        args = parser().parse_args(['run', *DEFAULT_FLAGS, '--manifest', str(self.manifest_path),
                                    '--output-dir', str(self.output)])
        config = manager_configuration(args)
        params = {'posture': args.adaptive_posture, 'max_steps': args.skill_max_steps,
                  **config['manager_kwargs']['combat_params'], 'spatial_params': {}, **config['value_metadata']}
        self.metadata = {'source_hashes': self.sources, 'skill_parameters': params,
                         'versions': {'numpy': importlib.metadata.version('numpy')},
                         'local_policy': {'context_hash': 'context', 'policy': self.base}}
        self.reference = {'manifest.json': self.old, 'run.json': deepcopy(self.metadata),
                          'game_version': 'test-sc2', 'hashes': {'run.json': 'original-hash'},
                          'path': str(self.study/'base')}
        self.protocol = {'schema_version': 1, 'kind': 'local_patch_collection',
                         'flags': list(DEFAULT_FLAGS), 'python': sys.executable, 'sc2_path': None,
                         'sources': self.sources, 'manifest_sha256': digest(self.old),
                         'policy_sha256': digest(self.base)}
        self.write()

    def write(self):
        save(self.study/'protocol.json', self.protocol)
        save(self.study/'manifest.json', self.old)
        save(self.study/'base_policy.json', self.base)
        save(self.policy_path, self.policy)
        save(self.manifest_path, self.new)

    def mocks(self, runs=None):
        stack = ExitStack()
        stack.enter_context(patch.object(deployment.study, 'source_hashes', return_value=self.sources))
        lookup = {str((self.study/'base').resolve()): self.reference, **(runs or {})}
        stack.enter_context(patch.object(deployment, 'read_run', side_effect=lambda p: lookup[str(Path(p).resolve())]))
        return stack

    def preflight(self, **kwargs):
        return deployment.preflight(self.study, self.policy_path, self.manifest_path, self.output, **kwargs)

    def fake_run(self, policy, rewards, interventions, outcomes=None):
        """Already-audited run-shaped input for testing this module's boundary."""
        is_model = policy['selection']['mode'] == 'model'
        run = {'path': str(self.output/('model' if is_model else 'base')), 'hashes': {'run.json': 'test'},
               'manifest.json': deepcopy(self.new), 'run.json': deepcopy(self.metadata),
               'game_version': 'test-sc2', 'episodes.json': [], 'events.jsonl': []}
        run['run.json']['local_policy']['policy'] = deepcopy(policy)
        for number, scene in enumerate(self.new['episodes'], 1):
            candidate = {'id': 'retarget:u0:7', 'family': 'retarget', 'duration_steps': 1,
                         'actors': ['u0'], 'actions': {'u0': 7}, 'features': {n: 0.0 for n in FEATURE_NAMES}}
            acted = interventions[number-1]
            # No candidate opportunity in the second scene, so both arms have
            # the same initial observation but no learned action there.
            candidates = [candidate] if number == 1 else []
            record = {'episode': number, 'initial_state_sha256': f'initial-{number}',
                      'scenario_sha256': scene['sha256'], 'reward': rewards[number-1],
                      'steps': 1, 'outcome': (outcomes or ['loss', 'draw'])[number-1]}
            event = {'event': 'local_decision', 'episode': number, 'step': 0,
                     'baseline_actions': [6], 'final_actions': [7 if acted else 6], 'candidates': candidates,
                     'selected_id': candidate['id'] if acted else 'BASE',
                     'proposed_id': candidate['id'] if acted else 'BASE', 'vetoed': False,
                     'proposals_used': int(acted), 'interventions': int(acted)}
            run['episodes.json'].append(record)
            run['events.jsonl'].extend([
                {'event': 'episode_started', 'episode': number, 'game_version': 'test-sc2',
                 'initial_state_sha256': record['initial_state_sha256'], 'scenario_sha256': scene['sha256']},
                {'event': 'skill_observation', 'episode': number, 'step': 0, 'payload': {'state': {'episode': number}}},
                event,
                {'event': 'step', 'episode': number, 'step': 1, 'reward': record['reward'], 'terminated': True},
                {'event': 'episode_completed', **record}])
        return run


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = Fixture(Path(self.tmp.name))

    def test_dry_run_preserves_exact_flags_starts_no_games_and_creates_nothing(self):
        before = sorted(str(p) for p in self.f.folder.rglob('*'))
        with self.f.mocks(), patch.object(deployment.study, 'run_arm') as run:
            result = deployment.evaluate(self.f.study, self.f.policy_path, self.f.manifest_path,
                                         self.f.output, dry_run=True)
        self.assertEqual(result['started_games'], 0)
        self.assertEqual(result['planned_games'], 4)
        self.assertEqual(result['commands']['MODEL'][4:4+len(DEFAULT_FLAGS)], list(DEFAULT_FLAGS))
        run.assert_not_called()
        self.assertFalse(self.f.output.exists())
        self.assertEqual(before, sorted(str(p) for p in self.f.folder.rglob('*')))

    def test_rejects_training_validation_and_unused_development_scenes(self):
        for group in ('train0', 'validation', 'development'):
            with self.subTest(group=group):
                self.f.new = manifest([group], self.f.old['env_args']); self.f.write()
                with self.f.mocks(), self.assertRaisesRegex(ValueError, 'overlap'):
                    self.f.preflight()

    def test_veto_training_and_validation_excluded_too(self):
        veto = model(gain=1)
        veto.update(kind='local_veto_model', old_model_hash=model_hash(self.f.policy['model']), old_min_gain=0,
                    training_group_ids=[digest({'test_scene': 'veto_train'+str(i)}) for i in range(4)],
                    validation_group_ids=[digest({'test_scene': 'veto_validation'})])
        self.f.policy['veto_model'] = veto
        for group in ('veto_train0', 'veto_validation'):
            with self.subTest(group=group):
                self.f.new = manifest([group], self.f.old['env_args']); self.f.write()
                with self.f.mocks(), self.assertRaisesRegex(ValueError, 'overlap'):
                    self.f.preflight()

    def test_legacy_validation_requires_bound_report(self):
        gain = self.f.policy['model']; del gain['validation_group_ids']; self.f.write()
        with self.f.mocks(), self.assertRaisesRegex(ValueError, 'no recorded validation'):
            self.f.preflight()
        report = {'schema_version': 1, 'kind': 'gain', 'context_hash': 'context', 'model_hash': model_hash(gain),
                  'split': {'train_groups': gain['training_group_ids'],
                            'validation_groups': [digest({'test_scene': 'validation'})]}}
        save(self.f.folder/'report.json', report)
        with self.f.mocks():
            self.assertEqual(self.f.preflight()['protocol']['excluded_model_groups']['gain']['validation_groups'],
                             report['split']['validation_groups'])
        report['model_hash'] = '0'*64; save(self.f.folder/'report.json', report)
        with self.f.mocks(), self.assertRaisesRegex(ValueError, 'does not bind'):
            self.f.preflight()

    def test_rejects_changed_sources_context_version_and_candidates(self):
        for field in ('context_hash', 'game_version', 'candidates', 'sources'):
            with self.subTest(field=field):
                policy = deepcopy(self.f.policy)
                sources = deepcopy(self.f.protocol['sources'])
                if field == 'candidates':
                    self.f.policy['candidates']['max_candidates'] += 1
                elif field == 'sources':
                    self.f.protocol['sources'] = {str(self.f.source): '0'*64}
                else:
                    self.f.policy['model'][field] = 'changed'
                self.f.write()
                with self.f.mocks(), self.assertRaises(ValueError):
                    self.f.preflight()
                self.f.policy, self.f.protocol['sources'] = policy, sources

    def test_rejects_frozen_flags_that_no_longer_match_baseline(self):
        self.f.protocol['flags'].remove('--adaptive-critical-escape'); self.f.write()
        with self.f.mocks(), self.assertRaisesRegex(ValueError, 'parameters'):
            self.f.preflight()

    def test_complete_episode_comparison_includes_no_intervention_and_draw(self):
        with self.f.mocks():
            prepared = self.f.preflight()
        base = self.f.fake_run(self.f.base, [1, 2], [False, False])
        learned = self.f.fake_run(self.f.policy, [3, 2], [True, False], ['win', 'draw'])
        result = deployment.compare_runs(base, learned, self.f.new, self.f.base, self.f.policy, prepared['protocol'])
        self.assertEqual(result['episodes'], 2)
        self.assertEqual(result['mean_gain'], 1)
        self.assertEqual(result['model']['win_rate'], .5)
        self.assertEqual(result['episodes_without_intervention'], 1)
        self.assertEqual(result['model']['outcomes']['draw'], 1)
        self.assertEqual(result['evidence_kind'], 'full_policy_matched_scenario_evaluation')

    def test_rejects_wrong_deployed_selection_and_initial_state(self):
        with self.f.mocks():
            prepared = self.f.preflight()
        base = self.f.fake_run(self.f.base, [1, 2], [False, False])
        learned = self.f.fake_run(self.f.policy, [3, 2], [False, False])
        with self.assertRaisesRegex(ValueError, 'frozen selector'):
            deployment.compare_runs(base, learned, self.f.new, self.f.base, self.f.policy, prepared['protocol'])
        learned = self.f.fake_run(self.f.policy, [3, 2], [True, False])
        learned['events.jsonl'][0]['game_version'] = 'other'
        with self.assertRaisesRegex(ValueError, 'initial state'):
            deployment.compare_runs(base, learned, self.f.new, self.f.base, self.f.policy, prepared['protocol'])

    def test_orchestrates_two_full_arms_and_writes_only_audited_report(self):
        base = self.f.fake_run(self.f.base, [1, 2], [False, False])
        learned = self.f.fake_run(self.f.policy, [3, 2], [True, False], ['win', 'draw'])
        with self.f.mocks({base['path']: base, learned['path']: learned}), patch.object(deployment.study, 'run_arm') as run:
            result = deployment.evaluate(self.f.study, self.f.policy_path, self.f.manifest_path, self.f.output)
        self.assertEqual(run.call_count, 2)
        self.assertEqual([c.args[3].name for c in run.call_args_list], ['base', 'model'])
        self.assertEqual([c.args[1] for c in run.call_args_list], [self.f.output/'manifest.json']*2)
        self.assertEqual(json.loads((self.f.output/'result.json').read_text(encoding='utf-8')), result)
        self.assertEqual(result['completed_games'], 4)

    def test_failed_arm_never_produces_a_performance_result(self):
        with self.f.mocks(), patch.object(deployment.study, 'run_arm', side_effect=ValueError('incomplete_run')):
            with self.assertRaisesRegex(ValueError, 'incomplete_run'):
                deployment.evaluate(self.f.study, self.f.policy_path, self.f.manifest_path, self.f.output)
        self.assertFalse((self.f.output/'result.json').exists())

    def test_wrong_base_game_version_stops_before_model_arm(self):
        base = self.f.fake_run(self.f.base, [1, 2], [False, False])
        base['game_version'] = 'new-sc2-version'
        with self.f.mocks({base['path']: base}), patch.object(deployment.study, 'run_arm') as run:
            with self.assertRaisesRegex(ValueError, 'game version'):
                deployment.evaluate(self.f.study, self.f.policy_path, self.f.manifest_path, self.f.output)
        self.assertEqual(run.call_count, 1)
        self.assertFalse((self.f.output/'result.json').exists())

    def test_veto_consumes_the_old_policy_opportunity_and_cannot_reschedule(self):
        policy = deepcopy(self.f.policy)
        veto = model(gain=1)
        veto.update(kind='local_veto_model', old_model_hash=model_hash(policy['model']), old_min_gain=0)
        policy['veto_model'] = veto
        learned = self.f.fake_run(policy, [1, 2], [True, False])
        event = next(e for e in learned['events.jsonl'] if e['event'] == 'local_decision')
        event.update(selected_id='BASE', vetoed=True, interventions=0, final_actions=[6])
        later = deepcopy(event)
        later.update(step=1, proposed_id='BASE', vetoed=False)
        counts = deployment._audit_selection({'events': [event, later]}, policy)
        self.assertEqual(counts['interventions'], 0)
        self.assertEqual(counts['proposals'], 1)
        self.assertEqual(counts['vetoes'], 1)
        later.update(proposed_id='retarget:u0:7', vetoed=True, proposals_used=2)
        with self.assertRaisesRegex(ValueError, 'frozen selector'):
            deployment._audit_selection({'events': [event, later]}, policy)


if __name__ == '__main__':
    unittest.main()
