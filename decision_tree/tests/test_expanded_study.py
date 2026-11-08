"""Expanded study planning contracts; these tests never launch StarCraft."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from decision_tree.experiment import digest, policy_contract
from decision_tree.schema import load_policy
from decision_tree.scripts import expanded_study as study


def gate(number, phase='engage', health=1., count=10, candidates=6, opportunity=True):
    return {'call_number': number, 'step': (number-1)*4, 'phase': phase,
            'features': {'ally_effective_health_ratio': health, 'ally_count': count},
            'candidate_calls': [{'id': f'c{i}', 'call': {'skill': 'adaptive_focus_fire', 'max_steps': i+1}}
                                for i in range(candidates)],
            'opportunity': opportunity}


class ExpandedStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_first_anchor_prefers_contact_and_second_ordinal_is_predeclared(self):
        gates = [gate(1, 'search'), gate(2, 'approach'), gate(3), gate(4), gate(5)]
        chosen = study.select_anchors(gates)
        self.assertEqual([g['call_number'] for g, _ in chosen], [2, 5])
        self.assertEqual(chosen[1][1], 'damage_or_later_boundary')

    def test_second_anchor_can_be_visible_loss_before_ordinal(self):
        for changed in (gate(2, health=.85), gate(2, count=9)):
            chosen = study.select_anchors([gate(1), changed, gate(4)])
            self.assertEqual([g['call_number'] for g, _ in chosen], [1, 2])

    def test_search_and_short_episode_fallback_no_reward_dependency(self):
        original = [gate(1, 'search'), gate(2, 'search')]
        selected = study.select_anchors(original)
        self.assertEqual([g['call_number'] for g, _ in selected], [1, 2])
        self.assertEqual(selected[1][1], 'second_boundary_fallback')
        modified = deepcopy(original)
        modified[0]['reward'] = -999
        modified[1]['final_return'] = 999
        self.assertEqual([g['call_number'] for g, _ in study.select_anchors(modified)], [1, 2])

    def test_missing_anchors_not_duplicated(self):
        self.assertEqual(study.select_anchors([]), [])
        self.assertEqual(study.select_anchors([gate(1, opportunity=False)]), [])
        self.assertEqual(len(study.select_anchors([gate(1)])), 1)
        self.assertEqual(len(study.select_anchors([gate(1, candidates=1)])), 1)

    def fake_run(self, count=40):
        policy = load_policy(study.ROOT/'decision_tree/configs/expanded_v2.json')
        episodes = [{'index': i, 'sc2_seed': 35000+i, 'episode_config': {'scene': i},
                     'sha256': digest({'scene': i})} for i in range(count)]
        manifest = {'version': 1, 'seed': 35000, 'env_args': {'map_name': 'fixture'},
                    'episodes': episodes, 'split': 'development'}
        manifest['sha256'] = digest(manifest)
        base = {'run.json': {'skill_tree': {'policy': policy}}, 'manifest.json': manifest,
                'hashes': {'events.jsonl': 'frozen-fixture'}}
        gates = [gate(1), gate(2), gate(4)]
        for g in gates:
            g['candidate_calls'] = [{'id': c['id'], 'call': {k:v for k,v in c.items() if k != 'id'}}
                                    for c in policy['candidates']]
        return base, gates, {'policy_contract': policy_contract(policy)}

    def make_plan(self, output, base, gates, protocol):
        output.mkdir()
        with patch('decision_tree.learning.dataset.read_run', return_value=base), \
             patch('decision_tree.learning.dataset.episode', return_value={'gates': gates}):
            return study.make_plan(self.root/'base', output, protocol)

    def test_40_scenes_make_160_independent_single_call_arms(self):
        base, gates, protocol = self.fake_run()
        plan = self.make_plan(self.root/'out', base, gates, protocol)
        self.assertEqual(plan['scene_count'], 40)
        self.assertEqual(plan['planned_games'], 200)
        self.assertEqual(len(plan['arms']), 160)
        self.assertEqual(len({r['group_id'] for r in plan['scenes']}), 40)
        keys = {(a['base_episode'], a['anchor_id'], a['candidate_id']) for a in plan['arms']}
        self.assertEqual(len(keys), 160)
        self.assertEqual({a['anchor_call_number'] for a in plan['arms']}, {1, 4})
        self.assertEqual({a['sampling_probability'] for a in plan['arms']}, {2/6})
        for arm in plan['arms']:
            manifest = study.read(arm['manifest'])
            original = base['manifest.json']['episodes'][arm['base_episode']-1]
            self.assertEqual(len(manifest['episodes']), 1)
            self.assertEqual(manifest['episodes'][0]['index'], 0)
            for k in ('sc2_seed', 'sha256', 'episode_config'):
                self.assertEqual(manifest['episodes'][0][k], original[k])
            self.assertEqual(manifest['sha256'], digest({k:v for k,v in manifest.items() if k != 'sha256'}))
            selection = study.read(arm['policy'])['selection']
            self.assertEqual(selection['candidate_id'], arm['candidate_id'])
            self.assertEqual(selection['anchor_call_number'], arm['anchor_call_number'])
            self.assertEqual(selection['max_interventions_per_episode'], 1)

    def test_sparse_plan_retains_missing_slots_and_repeat_selection(self):
        base, gates, protocol = self.fake_run(count=3)
        first = self.make_plan(self.root/'one', base, gates[:1], protocol)
        second = self.make_plan(self.root/'two', base, gates[:1], protocol)
        self.assertEqual(first['planned_games'], 9)
        self.assertEqual(sum(s['missing_anchors'] for s in first['scenes']), 3)
        identity = lambda p: [(a['base_episode'], a['anchor_id'], a['candidate_id'], a['anchor_call_number']) for a in p['arms']]
        self.assertEqual(identity(first), identity(second))

    def test_changed_frozen_sources_fail_before_launch(self):
        source = self.root/'source.py'
        source.write_text('one', encoding='utf-8')
        protocol = {'sources': {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}}
        study.check_sources(protocol)
        source.write_text('two', encoding='utf-8')
        with patch.object(study.subprocess, 'run') as launch, self.assertRaisesRegex(ValueError, 'Frozen source'):
            study.run_arm(protocol, self.root/'manifest', self.root/'policy', self.root/'out')
        launch.assert_not_called()

    def test_resume_different_arm_never_launches(self):
        base, _, _ = self.fake_run(count=1)
        manifest, policy = self.root/'manifest.json', self.root/'policy.json'
        study.save(manifest, base['manifest.json'])
        study.save(policy, base['run.json']['skill_tree']['policy'])
        output = self.root/'out'
        output.mkdir()
        protocol = {'sources': {}, 'policy_contract': policy_contract(base['run.json']['skill_tree']['policy'])}
        with patch('decision_tree.learning.dataset.read_run', return_value=base), patch.object(study.subprocess, 'run') as launch:
            study.run_arm(protocol, manifest, policy, output)
            launch.assert_not_called()
        base['manifest.json']['seed'] += 1
        with patch('decision_tree.learning.dataset.read_run', return_value=base), patch.object(study.subprocess, 'run') as launch:
            with self.assertRaisesRegex(ValueError, 'different arm'):
                study.run_arm(protocol, manifest, policy, output)
            launch.assert_not_called()

    def test_collect_resume_wrong_protocol_stops_before_lock_or_launch(self):
        prepared, output = self.root/'study', self.root/'out'
        prepared.mkdir()
        output.mkdir()
        manifest = study.read(study.ROOT/'sc2_skill_adapter/configs/protoss_5_vs_5_smoke.json')
        study.save(prepared/'manifest.json', manifest)
        policy = load_policy(study.ROOT/'decision_tree/configs/expanded_v2.json')
        study.save(prepared/'base.json', policy)
        protocol = {'sources': {}, 'manifest_sha256': manifest['sha256'], 'split': 'development',
                    'policy_contract': policy_contract(policy)}
        study.save(prepared/'expanded_protocol.json', protocol)
        study.save(output/'workflow.json', {'protocol_sha256': 'different'})
        with patch.object(study, 'run_arm') as launch, self.assertRaisesRegex(ValueError, 'Resume protocol mismatch'):
            study.collect(prepared, output)
        launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
