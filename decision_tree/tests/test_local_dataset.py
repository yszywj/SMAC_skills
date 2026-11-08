"""Behavioral auditing tests; synthetic logs are not performance evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from decision_tree.local.dataset import read_run, audit_pair, build_dataset, episode
from decision_tree.local.runtime import digest, execution_context, load_policy
from decision_tree.local.study import select_anchors, make_plan, save_once
from skills.current.tests.test_group_skills import scene as observed_scene, enemy_scene
from skills.local_v1 import FEATURE_NAMES, generate_candidates

P1 = 'retarget:u0:7'
P2 = 'retarget:u0:8'


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def receipt(payload, actions):
    """Synthetic native receipts follow the real adapter's accepted/dead schema."""
    actors, results = {}, []
    for slot, action in enumerate(actions):
        actor = f'u{slot}'
        alive = payload['state']['allies'][actor]['alive']
        record = {'requested_action_id': str(action), 'observation_errors': [],
                  'submission': 'accepted' if alive else 'dead_noop',
                  'command': None, 'response': None}
        if alive:
            record.update(command_index=len(results), response={'code': 1, 'name': 'Success'},
                          command={'ability_id': 23 if action >= 6 else 16, 'kind': 'attack' if action >= 6 else 'move',
                                   'target_unit': f'e{action-6}' if action >= 6 else None,
                                   'target_tag': None, 'target_position': None})
            results.append(1)
        actors[actor] = record
    loop = payload['state']['game_loop']
    return {'response_mapping': 'exact', 'actors': actors, 'raw_response_results': results,
            'unmatched_observation_errors': [], 'from_game_loop': loop,
            'to_game_loop': loop + payload['state']['step_mul']}


def fixture(path, selected=None, reward=1.0, candidates=2, dead_actor=False):
    path.mkdir()
    policy = load_policy({'schema_version': 1, 'kind': 'local_patch_policy',
                          'selection': {'mode': 'base' if selected is None else 'forced',
                                        **({'anchor_step': 1, 'candidate_id': selected} if selected else {})}})
    scene = {'episode_config': {'roster': [1, 2]}, 'sc2_seed': 100, 'index': 0}
    scene['sha256'] = digest(scene['episode_config'])
    manifest = {'env_args': {'map_name': '10gen_protoss'}, 'episodes': [scene], 'split': 'development'}
    manifest['sha256'] = digest(manifest)
    meta = {'method': 'adaptive-skill', 'skill_version': 'current', 'manifest_sha256': manifest['sha256'],
            'env_args': manifest['env_args'], 'source_hashes': {'code.py': 'frozen'},
            'versions': {'SC2': 'test'}, 'skill_parameters': {'max_steps': 4},
            'observation_protocol': 'test', 'native_teacher_control': 'off', 'medivac_policy': 'legacy'}
    meta['local_policy'] = {'policy': policy, 'feature_names': list(FEATURE_NAMES),
                            'context_hash': execution_context(meta, policy)}
    summary = {'status': 'complete', 'error': None, 'completed_episodes': 1,
               'expected_episodes': 1, 'manifest_sha256': manifest['sha256'], 'pair_mode': 'off'}
    start = {'event': 'episode_started', 'episode': 1, 'scenario_sha256': scene['sha256'],
             'initial_state_sha256': 'initial', 'game_version': 'test'}
    record = {'episode': 1, 'scenario_sha256': scene['sha256'], 'initial_state_sha256': 'initial',
              'steps': 3, 'reward': reward, 'outcome': 'win'}
    state = observed_scene([(0., 0.)] + ([(1., 0.)] if dead_actor else []),
                  {f'e{i}': enemy_scene((2. + i, 0.)) for i in range(candidates + 1)}, loop=0)
    if dead_actor:
        state['allies']['u1'].update(alive=False, health=0.0)
    for enemy in state['visible_enemies'].values():
        enemy.update(max_health=35., max_shield=0.)
    for relations in state['relations'].values():
        for relation in relations.values():
            relation.update(smac_attack_available=True, in_weapon_range=True, distance=2.)
    baseline = {'actions': [6] + ([0] if dead_actor else []),
                'roles': {'u0': 'attack'}, 'predicted_damage': {'e0': 10.0}}
    events = [start]
    for step in range(3):
        state['game_loop'] = step*8
        payload = {'state': deepcopy(state), 'questions': {
            actor: {'criteria': {**{str(i): 'move/stop' for i in range(1, 6)},
                                 **{str(6 + i): 'attack' for i in range(candidates + 1)}}}
            for actor in state['allies']}}
        patches = generate_candidates(payload, baseline, policy['candidates'])
        assert len(patches) == candidates
        actions = list(baseline['actions'])
        cid = selected if step == 1 and selected else 'BASE'
        if cid != 'BASE':
            actions[0] = next(p for p in patches if p['id'] == cid)['actions']['u0']
        events += [
            {'event': 'skill_observation', 'episode': 1, 'step': step, 'payload': payload},
            {'event': 'local_decision', 'episode': 1, 'step': step, 'game_loop': step*8,
             'baseline_actions': deepcopy(baseline['actions']), 'baseline_roles': deepcopy(baseline['roles']),
             'baseline_predicted_damage': deepcopy(baseline['predicted_damage']),
             'final_actions': actions, 'selected_id': cid,
             'proposed_id': cid, 'vetoed': False, 'candidates': deepcopy(patches)},
            {'event': 'step', 'episode': 1, 'step': step+1, 'game_loop': step*8,
             'actions': actions, 'reward': reward if step == 2 else 0.0, 'terminated': step == 2,
             'feedback': receipt(payload, actions)},
        ]
    events.append({'event': 'episode_completed', **record})
    for name, value in (('run.json', meta), ('manifest.json', manifest), ('summary.json', summary),
                        ('episodes.json', [record])):
        write(path/name, value)
    (path/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events), encoding='utf-8')
    return events


def replace_events(path, events):
    (path/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events), encoding='utf-8')


class LocalDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base, self.arm = self.root/'base', self.root/'arm'
        fixture(self.base)
        self.events = fixture(self.arm, P1, 3.0)

    def tearDown(self):
        self.temp.cleanup()

    def entry(self, path=None, cid=P1):
        return {'path': str(path or self.arm), 'base_episode': 1, 'anchor_step': 1, 'candidate_id': cid}

    def test_actual_delta_and_missing_arms_remain_incomplete(self):
        data = build_dataset(self.base, [self.entry()])
        self.assertTrue(data['audited'])
        self.assertEqual(len(data['rows']), 1)
        self.assertEqual(data['rows'][0]['gain'], 2.0)
        self.assertFalse(data['rows'][0]['anchor_complete'])
        self.assertEqual(data['rows'][0]['available_candidate_ids'], [P1, P2])

    def test_full_candidate_set_sets_complete_and_preserves_negative(self):
        second = self.root/'second'
        fixture(second, P2, -2.0)
        data = build_dataset(self.base, [self.entry(), self.entry(second, P2)])
        self.assertEqual([r['gain'] for r in data['rows']], [2.0, -3.0])
        self.assertTrue(all(r['anchor_complete'] for r in data['rows']))

    def test_missing_directory_never_becomes_zero_reward(self):
        data = build_dataset(self.base, [self.entry(self.root/'missing')])
        self.assertEqual(data['rows'], [])
        self.assertEqual(len(data['rejected']), 1)

    def test_prefix_change_rejected_but_suffix_observation_may_differ(self):
        for e in self.events:
            if e['event'] == 'skill_observation' and e['step'] == 2:
                e['payload']['state']['new_observed_value'] = 99
        replace_events(self.arm, self.events)
        self.assertEqual(len(build_dataset(self.base, [self.entry()])['rows']), 1)
        for e in self.events:
            if e['event'] == 'skill_observation' and e['step'] == 1:
                e['payload']['state']['new_observed_value'] = 99
        replace_events(self.arm, self.events)
        self.assertIn('observed_prefix_mismatch', build_dataset(self.base, [self.entry()])['rejected'][0]['reason'])

    def test_rejected_command_anywhere_rejects_evidence(self):
        self.events[-2]['feedback']['actors']['u0']['submission'] = 'rejected'
        replace_events(self.arm, self.events)
        with self.assertRaisesRegex(ValueError, 'command_rejected'):
            read_run(self.arm)

    def test_second_intervention_rejected(self):
        for e in self.events:
            if e['event'] == 'local_decision' and e['step'] == 2:
                e['selected_id'] = e['proposed_id'] = P1
                e['final_actions'] = [7]
            if e['event'] == 'step' and e['step'] == 3:
                e['actions'] = [7]
                e['feedback']['actors']['u0']['requested_action_id'] = '7'
        replace_events(self.arm, self.events)
        self.assertIn('not_one_isolated', build_dataset(self.base, [self.entry()])['rejected'][0]['reason'])

    def test_unsubmitted_proposal_and_illegal_action_rejected(self):
        for e in self.events:
            if e['event'] == 'local_decision' and e['step'] == 1:
                e['final_actions'] = [3]
        replace_events(self.arm, self.events)
        with self.assertRaisesRegex(ValueError, 'final_actions_not_submitted'):
            read_run(self.arm)

    def test_duplicate_pair_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate_comparison'):
            build_dataset(self.base, [self.entry(), self.entry()])

    def test_context_hash_must_be_recomputed(self):
        path = self.arm/'run.json'
        meta = json.loads(path.read_text())
        meta['skill_parameters']['max_steps'] = 5
        write(path, meta)
        self.assertIn('context_hash_mismatch', build_dataset(self.base, [self.entry()])['rejected'][0]['reason'])

    def test_candidate_features_recomputed_even_when_both_logs_agree(self):
        for path in (self.base, self.arm):
            events = [json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
            for event in events:
                if event['event'] == 'local_decision':
                    event['candidates'][0]['features']['actor_health_ratio'] = 999999.0
            replace_events(path, events)
        with self.assertRaisesRegex(ValueError, 'candidate_snapshot_mismatch'):
            build_dataset(self.base, [self.entry()])

    def test_candidate_effect_and_binding_recomputed(self):
        for key, value in (('effect_key', 'forged'), ('aliases', ['unobserved_alias']),
                           ('roles', {'u0': 'retreat'})):
            with self.subTest(key=key):
                events = deepcopy(self.events)
                next(e for e in events if e['event'] == 'local_decision')['candidates'][0][key] = value
                replace_events(self.arm, events)
                with self.assertRaisesRegex(ValueError, 'candidate_snapshot_mismatch'):
                    read_run(self.arm)

    def test_candidate_limits_use_the_recorded_policy(self):
        path = self.arm/'run.json'
        meta = json.loads(path.read_text())
        meta['local_policy']['policy']['candidates']['max_candidates'] = 1
        meta['local_policy']['context_hash'] = execution_context(meta, meta['local_policy']['policy'])
        write(path, meta)
        with self.assertRaisesRegex(ValueError, 'candidate_snapshot_mismatch'):
            read_run(self.arm)

    def test_feature_schema_must_match_public_runtime(self):
        path = self.arm/'run.json'
        meta = json.loads(path.read_text())
        meta['local_policy']['feature_names'] = ['health']
        write(path, meta)
        with self.assertRaisesRegex(ValueError, 'invalid_feature_schema'):
            read_run(self.arm)

    def test_empty_feedback_or_action_mismatch_rejected(self):
        for change, reason in ((lambda f: f.update(actors={}), 'feedback_actor_coverage'),
                               (lambda f: f['actors']['u0'].update(requested_action_id='1'), 'feedback_action_mismatch')):
            events = deepcopy(self.events)
            change(next(e for e in events if e['event'] == 'step')['feedback'])
            replace_events(self.arm, events)
            with self.assertRaisesRegex(ValueError, reason):
                read_run(self.arm)

    def test_feedback_clock_matches_observation_and_step_size(self):
        for field in ('from_game_loop', 'to_game_loop'):
            events = deepcopy(self.events)
            next(e for e in events if e['event'] == 'step')['feedback'][field] += 1
            replace_events(self.arm, events)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'feedback_clock_mismatch'):
                read_run(self.arm)
        events = deepcopy(self.events)
        next(e for e in events if e['event'] == 'skill_observation')['payload']['state']['game_loop'] += 1
        replace_events(self.arm, events)
        with self.assertRaisesRegex(ValueError, 'decision_clock_mismatch'):
            read_run(self.arm)

    def test_live_actor_requires_native_success_receipt(self):
        for changes, reason in (({'submission': 'dead_noop'}, 'command_rejected'),
                                ({'response': None}, 'invalid_command_receipt'),
                                ({'command_index': 1}, 'invalid_command_receipt')):
            events = deepcopy(self.events)
            next(e for e in events if e['event'] == 'step')['feedback']['actors']['u0'].update(changes)
            replace_events(self.arm, events)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, reason):
                read_run(self.arm)

    def test_dead_actor_only_accepts_exact_dead_noop(self):
        path = self.root/'dead'
        events = fixture(path, dead_actor=True)
        read_run(path)
        next(e for e in events if e['event'] == 'step')['feedback']['actors']['u1']['submission'] = 'accepted'
        replace_events(path, events)
        with self.assertRaisesRegex(ValueError, 'invalid_dead_noop_receipt'):
            read_run(path)

    def test_batch_ordinal_is_only_observation_field_ignored(self):
        for e in self.events:
            if e['event'] == 'skill_observation':
                e['payload']['state']['episode'] = 42
        replace_events(self.arm, self.events)
        self.assertEqual(len(build_dataset(self.base, [self.entry()])['rows']), 1)

    def test_anchor_selection_outcome_blind_spaced(self):
        events = [{'event': 'local_decision', 'step': t, 'candidates': [{}], 'reward': 100-t}
                  for t in range(15)]
        self.assertEqual([r['step'] for r in select_anchors(events, 3, 4)], [0, 4, 8])
        for e in events:
            e['reward'] = -e['reward']
        self.assertEqual([r['step'] for r in select_anchors(events, 3, 4)], [0, 4, 8])

    def test_plan_all_canonical_candidates_with_single_episode_inputs(self):
        output = self.root/'collection'
        output.mkdir()
        policy = read_run(self.base)['run.json']['local_policy']['policy']
        plan = make_plan(self.base, output, policy, anchors=2, min_gap=2)
        self.assertEqual(plan['candidate_games'], 4)
        self.assertEqual([a['anchor_step'] for a in plan['arms']], [0, 0, 2, 2])
        for arm in plan['arms']:
            self.assertEqual(len(json.loads(Path(arm['manifest']).read_text())['episodes']), 1)
            self.assertEqual(load_policy(arm['policy'])['selection']['anchor_step'], arm['anchor_step'])
        self.assertEqual(plan, make_plan(self.base, output, policy, anchors=2, min_gap=2))
        with self.assertRaisesRegex(ValueError, 'Refusing'):
            save_once(output/'inputs'/'changed.json', {'x': 1})
            save_once(output/'inputs'/'changed.json', {'x': 2})


if __name__ == '__main__':
    unittest.main()
