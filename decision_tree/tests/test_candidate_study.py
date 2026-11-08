"""Complete same-anchor candidate comparisons, with no real game launches."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from decision_tree.experiment import digest, policy_contract
from decision_tree.schema import load_policy
from decision_tree.scripts import candidate_study as study


def gate(number, ids, phase='engage', health=1., count=10):
    return {'call_number': number, 'step': (number-1)*4, 'phase': phase,
            'features': {'ally_effective_health_ratio': health, 'ally_count': count},
            'candidate_calls': [{'id': cid, 'call': {'skill': cid if cid in ('regroup', 'cohesive_search', 'cover_retreat')
                                                    else 'adaptive_focus_fire'}} for cid in ids],
            'opportunity': bool(ids)}


class CandidateStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, n=2, anchors=3, cap=None):
        policy = load_policy(study.ROOT/'decision_tree/configs/candidate_v3.json')
        episodes = [{'index': i, 'sc2_seed': 81000+i, 'episode_config': {'scene': i},
                     'sha256': digest({'scene': i})} for i in range(n)]
        manifest = {'version': 1, 'seed': 81000, 'env_args': {'map_name': 'fixture'},
                    'episodes': episodes, 'split': 'development'}
        manifest['sha256'] = digest(manifest)
        base = {'run.json': {'skill_tree': {'policy': policy}}, 'manifest.json': manifest,
                'hashes': {'events.jsonl': 'fixture_hash'}}
        gates = [gate(1, ['regroup', 'cohesive_search'], 'search'),
                 gate(2, ['focus_weakest', 'press', 'focus_budget', 'focus_threat', 'focus_easy']),
                 gate(5, ['focus_weakest', 'press', 'focus_budget', 'focus_threat', 'focus_easy', 'cover_retreat'], health=.8)]
        protocol = {'policy_contract': policy_contract(policy), 'anchors_per_scene': anchors,
                    'candidate_ids': [c['id'] for c in policy['candidates']], 'max_candidate_games': cap}
        return base, gates, protocol

    def plan(self, name='out', n=2, anchors=3, cap=None, modify=None):
        base, gates, protocol = self.fixture(n, anchors, cap)
        if modify:
            modify(gates)
        output = self.root/name
        output.mkdir()
        with patch('decision_tree.learning.dataset.read_run', return_value=base), \
             patch('decision_tree.learning.dataset.episode', return_value={'gates': gates}):
            return study.make_plan(self.root/'base', output, protocol), base

    def test_default_flags_enable_executors_without_automatic_dispatch_or_positioning(self):
        self.assertIn('--enable-regroup', study.CANDIDATE_FLAGS)
        self.assertIn('--enable-cohesive-search', study.CANDIDATE_FLAGS)
        self.assertNotIn('--auto-spatial-dispatch', study.CANDIDATE_FLAGS)
        self.assertNotIn('--enable-combat-positioning', study.CANDIDATE_FLAGS)
        self.assertEqual(study.CANDIDATE_FLAGS[:len(study.DEFAULT_FLAGS)], study.DEFAULT_FLAGS)

    def test_config_has_distinct_skills_and_no_duration_only_candidates(self):
        policy = load_policy(study.ROOT/'decision_tree/configs/candidate_v3.json')
        self.assertEqual(policy['exploration']['feature_schema'], 3)
        skills = {c['skill'] for c in policy['candidates']}
        self.assertEqual(skills, {'adaptive_focus_fire', 'focus_fire_budget', 'cover_retreat', 'regroup', 'cohesive_search'})
        self.assertFalse({'short_call', 'long_call'} & {c['id'] for c in policy['candidates']})

    def test_third_anchor_includes_early_search_and_is_sorted_chronologically(self):
        _, gates, _ = self.fixture()
        selected = study.select_anchors(gates)
        self.assertEqual([g['call_number'] for g,_ in selected], [1, 2, 5])
        self.assertEqual([r for _,r in selected], ['spatial_phase', 'first_contact_or_available', 'loss_or_later'])

    def test_two_anchors_keep_contact_then_loss_and_no_future_reward_dependency(self):
        _, gates, _ = self.fixture()
        before = study.select_anchors(gates, 2)
        self.assertEqual([g['call_number'] for g,_ in before], [2,5])
        changed = deepcopy(gates)
        for i,g in enumerate(changed):
            g['reward'] = 999-i
            g['outcome'] = 'win'
        after = study.select_anchors(changed, 2)
        self.assertEqual([g['call_number'] for g,_ in before], [g['call_number'] for g,_ in after])

    def test_later_ordinal_and_short_episode_fallback_never_duplicate_anchors(self):
        gates = [gate(i, ['press']) for i in (1,2,4,5,7)]
        self.assertEqual([g['call_number'] for g,_ in study.select_anchors(gates)], [1,4,7])
        self.assertEqual([g['call_number'] for g,_ in study.select_anchors(gates[:2])], [1,2])
        self.assertEqual(study.select_anchors([]), [])
        self.assertEqual(study.select_anchors([gate(1, [])]), [])
        for value in (0,4,True,1.5):
            with self.assertRaises(ValueError):
                study.select_anchors(gates, value)

    def test_every_available_candidate_is_tested_at_every_anchor(self):
        plan, _ = self.plan(n=3)
        self.assertEqual(plan['scene_count'], 3)
        self.assertEqual(plan['candidate_games'], 3*(2+5+6))
        self.assertEqual(plan['planned_games'], 42)
        self.assertFalse(plan['budget_exceeded'])
        self.assertEqual({a['sampling_probability'] for a in plan['arms']}, {1.})
        self.assertEqual(len(plan['anchors']), 9)
        for anchor in plan['anchors']:
            tested = [a['candidate_id'] for a in plan['arms']
                      if a['base_episode']==anchor['base_episode'] and a['anchor_id']==anchor['anchor_id']]
            self.assertEqual(tested, anchor['available_candidate_ids'])
            self.assertEqual(tested, anchor['tested_candidate_ids'])
        self.assertEqual(len({(a['base_episode'],a['anchor_id'],a['candidate_id']) for a in plan['arms']}),39)

    def test_single_episode_arms_preserve_seed_and_scene_and_change_only_selected_call(self):
        plan, base = self.plan(n=1)
        for arm in plan['arms']:
            single = study.read(arm['manifest'])
            original = base['manifest.json']['episodes'][0]
            self.assertEqual(single['episodes'], [original])
            self.assertEqual(single['sha256'], arm['manifest_sha256'])
            policy = load_policy(arm['policy'])
            self.assertEqual(digest(policy), arm['policy_sha256'])
            self.assertEqual(policy_contract(policy), policy_contract(base['run.json']['skill_tree']['policy']))
            self.assertEqual(policy['selection']['anchor_call_number'], arm['anchor_call_number'])
            self.assertEqual(policy['selection']['candidate_id'], arm['candidate_id'])
            self.assertEqual(policy['selection']['max_interventions_per_episode'],1)

    def test_budget_never_silently_truncates_candidates(self):
        plan, _ = self.plan(cap=3)
        self.assertTrue(plan['budget_exceeded'])
        self.assertEqual(plan['candidate_games'],26)
        self.assertEqual(len(plan['arms']),26)

    def test_duplicate_and_unknown_candidates_rejected(self):
        def duplicate(gates):
            gates[0]['candidate_calls'].append(deepcopy(gates[0]['candidate_calls'][0]))
        def unknown(gates):
            gates[0]['candidate_calls'][0]['id']='unknown'
        for name, edit in (('duplicate',duplicate), ('unknown',unknown)):
            with self.assertRaisesRegex(ValueError,'Unknown or duplicate'):
                self.plan(name=name, modify=edit)

    def prepared(self):
        manifest_path=study.ROOT/'sc2_skill_adapter/configs/protoss_5_vs_5_smoke.json'
        policy_path=study.ROOT/'decision_tree/configs/candidate_v3.json'
        output=self.root/'study'
        result=study.prepare_candidates(manifest_path,policy_path,output,anchors=2,
                                        candidate_ids=['press','focus_budget'],max_candidate_games=0)
        return output,result

    def test_prepare_declares_bounds_freezes_sources_and_filters_before_runtime(self):
        prepared,result=self.prepared()
        protocol=study.read(prepared/'candidate_protocol.json')
        self.assertEqual(result['started_games'],0)
        self.assertEqual(protocol['candidate_ids'],['press','focus_budget'])
        self.assertEqual(protocol['max_candidate_games'],0)
        self.assertEqual(protocol['anchors_per_scene'],2)
        self.assertEqual(protocol['candidates_per_anchor'],'all_available_declared')
        self.assertIn(str(Path(study.__file__).resolve()),protocol['sources'])
        study.check_sources(protocol)

    def test_prepare_rejects_automatic_dispatch_before_creating_output(self):
        with self.assertRaisesRegex(ValueError,'auto-spatial-dispatch'):
            study.prepare_candidates('unused',study.ROOT/'decision_tree/configs/candidate_v3.json',self.root/'bad',
                                     flags=[*study.CANDIDATE_FLAGS,'--auto-spatial-dispatch'])
        self.assertFalse((self.root/'bad').exists())

    def test_budget_stop_runs_only_base_and_writes_reviewable_plan(self):
        prepared,_=self.prepared()
        output=self.root/'run'
        # Lock stays in the test workspace. No real process is started.
        (self.root/'sc2_skill_adapter').mkdir()
        protocol=study.read(prepared/'candidate_protocol.json')
        plan={'scene_count':2,'planned_games':7,'arms':[{}]*5,'scenes':[{'missing_anchors':0}]*2,
              'budget_exceeded':True,'protocol_sha256':digest(protocol)}
        with patch.object(study,'ROOT',self.root), patch.object(study,'run_arm') as launch, \
             patch.object(study,'make_plan',return_value=plan):
            result=study.collect_candidates(prepared,output)
        self.assertEqual(launch.call_count,1)
        self.assertEqual(result['status'],'budget_exceeded')
        self.assertEqual(result['started_candidate_games'],0)
        self.assertEqual(study.read(output/'anchor_plan.json'),plan)
        self.assertFalse((self.root/'sc2_skill_adapter/.medivac-return-study.lock').exists())


if __name__ == '__main__':
    unittest.main()
