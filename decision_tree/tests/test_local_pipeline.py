"""Synthetic offline contract test, never SC2 or evidence of policy performance.

Canonical candidates and native-receipt-shaped fixtures exercise the real audit,
training CLI, exported policy loader, and manager commit boundary. Artificial
scene identities and returns exist solely to test schema/CLI compatibility.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

from decision_tree.local.dataset import build_dataset, read_run
from decision_tree.local.models import model_hash
from decision_tree.local.runtime import LocalController, digest, load_policy
from decision_tree.local.training import main as training_main
from decision_tree.tests.test_local_dataset import fixture, P1, replace_events, write
from skills.local_v1 import FEATURE_NAMES


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def bind_scene(path, scene_id, seed):
    """Give one synthetic episode a unique scene/initial-state identity."""
    manifest = read(path/'manifest.json')
    scene = manifest['episodes'][0]
    scene['episode_config']['synthetic_scene'] = scene_id
    scene['sha256'] = digest(scene['episode_config'])
    scene['sc2_seed'] = seed
    manifest['sha256'] = digest({k: v for k, v in manifest.items() if k != 'sha256'})
    initial = digest({'synthetic_initial_scene': scene_id, 'seed': seed})
    records = read(path/'episodes.json')
    events = [json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
    for event in [*records, *events]:
        if 'scenario_sha256' in event:
            event['scenario_sha256'] = scene['sha256']
        if 'initial_state_sha256' in event:
            event['initial_state_sha256'] = initial
    for name in ('run.json', 'summary.json'):
        value = read(path/name)
        value['manifest_sha256'] = manifest['sha256']
        write(path/name, value)
    write(path/'manifest.json', manifest)
    write(path/'episodes.json', records)
    replace_events(path, events)
    return manifest, records, events


def paired_batch(directory, namespace, gain, count=8):
    """One multi-episode BASE and one fresh forced arm per synthetic scene."""
    directory.mkdir()
    scenes, records, events, entries = [], [], [], []
    for index in range(count):
        base_input, arm = directory/f'base-input-{index}', directory/f'arm-{index}'
        fixture(base_input, reward=3.0, candidates=1)
        fixture(arm, P1, reward=3.0 + gain, candidates=1)
        identity = f'{namespace}:{index}'
        manifest, episode_records, episode_events = bind_scene(base_input, identity, 1000 + index)
        bind_scene(arm, identity, 1000 + index)
        scene = deepcopy(manifest['episodes'][0])
        scene['index'] = index
        scenes.append(scene)
        for event in [*episode_records, *episode_events]:
            if 'episode' in event:
                event['episode'] = index + 1
            if event.get('event') == 'skill_observation':
                event['payload']['state']['episode'] = index + 1
        records.extend(episode_records)
        events.extend(episode_events)
        entries.append({'path': str(arm), 'base_episode': index + 1, 'arm_episode': 1,
                        'anchor_step': 1, 'candidate_id': P1})
    base = directory/'BASE'
    base.mkdir()
    manifest['episodes'] = scenes
    manifest['sha256'] = digest({k: v for k, v in manifest.items() if k != 'sha256'})
    metadata, summary = read(base_input/'run.json'), read(base_input/'summary.json')
    metadata.update(manifest_sha256=manifest['sha256'], scenario_count=count)
    summary.update(manifest_sha256=manifest['sha256'], completed_episodes=count, expected_episodes=count)
    for name, value in (('run.json', metadata), ('manifest.json', manifest),
                        ('summary.json', summary), ('episodes.json', records)):
        write(base/name, value)
    replace_events(base, events)
    data = build_dataset(base, entries)
    paired = directory/'paired.json'
    write(paired, data)
    return paired, data, base


class LocalPipelineTests(unittest.TestCase):
    def test_audited_multiepisode_logs_to_gain_and_frozen_veto_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            gain_path, gain_data, base = paired_batch(directory/'gain-data', 'gain', 2.0)
            self.assertEqual(len(read_run(base)['episodes.json']), 8)
            self.assertEqual(len(gain_data['rows']), 8)
            self.assertFalse(gain_data['rejected'])
            self.assertTrue(all(row['anchor_complete'] and row['gain'] == 2.0 for row in gain_data['rows']))
            self.assertEqual(len({row['group_id'] for row in gain_data['rows']}), 8)
            self.assertEqual(gain_data['feature_names'], list(FEATURE_NAMES))

            gain_output = directory/'gain-model'
            with redirect_stdout(io.StringIO()):
                training_main(['gain', '--dataset', str(gain_path), '--output', str(gain_output)])
            gain_policy = load_policy(gain_output/'policy.json')
            self.assertEqual(gain_policy['selection']['mode'], 'model')
            self.assertEqual(gain_policy['model']['feature_names'], list(FEATURE_NAMES))
            self.assertEqual(gain_policy['model']['game_version'], 'test')
            self.assertEqual(len(gain_policy['model']['training_group_ids']), 6)
            self.assertEqual(len(gain_policy['model']['validation_group_ids']), 2)

            correction_path, correction_data, _ = paired_batch(directory/'correction-data', 'correction', -2.0)
            self.assertFalse({r['group_id'] for r in gain_data['rows']} &
                             {r['group_id'] for r in correction_data['rows']})
            self.assertEqual(gain_data['context_hash'], correction_data['context_hash'])
            veto_output = directory/'veto-model'
            with redirect_stdout(io.StringIO()):
                training_main(['veto', '--dataset', str(correction_path), '--old-model',
                               str(gain_output/'model.json'), '--output', str(veto_output)])
            veto_policy = load_policy(veto_output/'policy.json')
            self.assertEqual(veto_policy['model'], gain_policy['model'])
            self.assertEqual(veto_policy['veto_model']['old_model_hash'], model_hash(gain_policy['model']))
            self.assertEqual(veto_policy['veto_model']['game_version'], 'test')
            self.assertFalse(read(veto_output/'report.json')['correction_rejected'])

            # This executes the real allocator/manager without any environment.
            # Supplying the synthetic context here tests wiring, not deployment
            # eligibility: a real runner must also bind sources and SC2 version.
            from decision_tree.tests.test_integration import observed
            from decision_tree.tests.test_invocation import combat_scene
            from sc2_skill_adapter.bridge.skill_runtime import load_manager
            payload = observed(combat_scene())
            context = gain_data['context_hash']
            gain_controller = LocalController(load_manager()(), gain_policy, context_hash=context)
            actions, _, decisions = gain_controller.step(deepcopy(payload), 1)
            self.assertEqual(decisions[0]['selected_id'], P1)
            self.assertEqual(actions, [7])
            veto_controller = LocalController(load_manager()(), veto_policy, context_hash=context)
            actions, _, decisions = veto_controller.step(deepcopy(payload), 1)
            self.assertEqual(decisions[0]['proposed_id'], P1)
            self.assertTrue(decisions[0]['vetoed'])
            self.assertEqual(decisions[0]['selected_id'], 'BASE')
            self.assertEqual(actions, decisions[0]['baseline_actions'])
            self.assertEqual(veto_controller.proposals_used, 1)
            self.assertEqual(veto_controller.interventions, 0)


if __name__ == '__main__':
    unittest.main()
