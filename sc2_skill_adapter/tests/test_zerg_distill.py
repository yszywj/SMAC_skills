"""Replay actual pre-decision payloads against the native manager, never an SC2 env."""
import asyncio
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bridge import runner, zerg_shadow
from bridge.skill_runtime import manager_configuration
from bridge.zerg_distill import Distill, check_protocol
from test_zerg_shadow_runner import OfflineEnv


WORKER = (Path(__file__).resolve().parents[2] / 'runs/zerg_shadow_ptpy1yzu/workers'
          / 'zerg_1zergling_0baneling_9hydralisk/episode-006-seed-20262168')


class Historical(unittest.TestCase):
    @unittest.skipUnless(WORKER.exists(), 'historical archive not installed')
    def test_runner_stats_restart_at_opening(self):
        run = json.loads((WORKER/'runner/run.json').read_text())
        manifest = json.loads((WORKER/'manifest.json').read_text())
        manifest['episodes'] *= 2  # same opening twice: verify reset, not rollout outcomes
        observations = []
        with (WORKER/'runner/events.jsonl').open() as f:
            for line in f:
                e = json.loads(line)
                if e['event'] == 'skill_observation':
                    observations.append(e['payload'])
                    if len(observations) == 4:
                        break

        class ArchivedEnv(OfflineEnv):
            n_agents = n_enemies = 10
            n_actions = 16
            _episode_steps = episode_limit = 4

            def __init__(self, config):
                super().__init__(config)
                self.index = 0
                self._obs.observation.game_loop = 2
                self._controller.data = lambda: SimpleNamespace(units={107: 'hydralisk'})
                self._controller.ping = lambda: SimpleNamespace(game_version='4.10.0.75689')

            def get_avail_agent_actions(self, i):
                q = observations[self.index]['questions'].get(f'u{i}')
                return [str(j) in q['criteria'] if q else j == 0 for j in range(16)]

            def step_micro_window(self, actions, window, macro_loops):
                self.index += 1
                self._obs.observation.game_loop += macro_loops
                return 0, self.index == len(observations), dict(dead_allies=0, dead_enemies=0, battle_won=False)

        def payload(env, data, model, episode, health, previous, *args):
            pre = copy.deepcopy(observations[env.index])
            pre['state'].update(episode=episode, previous_action_ids=previous)
            return pre

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            args = runner.parser().parse_args(['run', '--manifest', 'unused', '--output-dir', 'unused', '--method',
                'adaptive-skill', '--skill-version', 'tactical_v1', '--execution-mode', 'micro8-all'])
            model = Path(__file__).resolve().parents[2]/'runs/zerg_policy_distill_v1/model.json'
            for mock in (patch.dict(os.environ, dict(ZERG_DISTILL_MODEL=str(model), ZERG_SHADOW_DIR=str(root/'shadow'),
                         ZERG_SHADOW_ROSTER=run['zerg_shadow']['scene_config_id'], SMAC_DAMAGE_MAP_DIR=str(Path(run['map_file']).parent))),
                         patch('bridge.environment.ReplayableEnv', ArchivedEnv), patch('bridge.policy.build_request', payload),
                         patch('bridge.feedback.action_feedback', side_effect=lambda *args: {}),
                         patch('bridge.environment.audit_unit_balance', return_value=dict(passed=True)),
                         patch('scripts.make_damage_map.begin_resource_damage'), patch('scripts.make_damage_map.read_resource_damage',
                         return_value=[0]*8), patch.object(runner, 'score_snapshot', side_effect=lambda env: dict(
                         game_loop=env._obs.observation.game_loop, score_details={k: {'life': 0} for k in
                         ('total_damage_dealt', 'total_damage_taken')}))):
                stack.enter_context(mock)
            from decision_tree.zerg import policy_repair_v2
            with patch.object(policy_repair_v2, 'v2_choose', wraps=policy_repair_v2.v2_choose) as v2:
                for version in ('v1', 'v2'):
                    with patch.dict(os.environ, dict(ZERG_DISTILL_VERSION=version, ZERG_SHADOW_DIR=str(root/version))):
                        out = root/f'runner-{version}'
                        summary = asyncio.run(runner.evaluate(args, manifest, 'adaptive-skill', out))
                    rows = json.loads((out/'episodes.json').read_text())
                    self.assertEqual(rows[0]['zerg_distill'], rows[1]['zerg_distill'])
                    self.assertEqual(summary['zerg_distill']['stats'], {k: v*2 for k, v in rows[0]['zerg_distill'].items()})
                    self.assertEqual(summary['zerg_distill']['stats']['steps'], 8)
                    self.assertEqual(v2.call_count, 8 if version == 'v2' else 0)
                    for d in json.loads((out/'distill/episode-0001.json').read_text())['decisions']:
                        self.assertEqual(d['policy_version'], version)
                        self.assertTrue(d['selection_reason'])

    @unittest.skipUnless(WORKER.exists(), 'historical archive not installed')
    def test_base_and_switch_from_opening(self):
        manifest = json.loads((WORKER/'manifest.json').read_text())
        run = json.loads((WORKER/'runner/run.json').read_text())
        check_protocol(run)
        with self.assertRaisesRegex(ValueError, 'execution_mode'):
            check_protocol(dict(run, execution_mode='legacy8'))
        archived = run['zerg_shadow']
        args = runner.parser().parse_args(['run', '--manifest', 'unused', '--output-dir', 'unused', '--method',
            'adaptive-skill', '--skill-version', 'tactical_v1', '--execution-mode', 'micro8-all'])
        rows, receipts = [], []
        with (WORKER/'runner/events.jsonl').open() as f:
            for line in f:
                event = json.loads(line)
                if event['event'] == 'skill_observation':
                    rows.append(event['payload'])
                elif event['event'] == 'step':
                    receipts.append(event)
        # Deliberate selector probes exercise both native paths independently of fit quality.
        for index, expected in [(0, 'model_base'), (-1, 'model_switch'), (None, 'real_model')]:
            def choose(model, pre, menu):
                self.assertIs(pre, rows[step])
                pick = menu[index]
                return dict(actions=pick, scores=[float(v == pick) for v in menu], score=1., unique_count=len(menu), top_ties=1)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                runtime, ctrl = zerg_shadow.start_shadow(str(root/'shadow'), archived['scene_config_id'], manifest,
                            'adaptive-skill', args, manager_configuration(args)['manager_kwargs'])
                model = dict(source_hash=archived['source_fingerprint'], ctx_hash=archived['execution_context_hash'],
                             schema_hash='projection', split_hash='split')
                context = dict(archived, execution_context_hash=runtime.execution_context_hash,
                               source_fingerprint=runtime.bundle_provenance['source_fingerprint'])
                if index is None:
                    path = Path(__file__).resolve().parents[2] / 'runs/zerg_policy_distill_v1/model.json'
                    with self.assertRaisesRegex(ValueError, 'context mismatch'):
                        Distill(root, path, dict(context, execution_context_hash='0'*64))
                    hook = Distill(root, path, context)
                else:
                    with patch('decision_tree.zerg.policy_distill', SimpleNamespace(load_model=lambda p: model, choose=choose), create=True):
                        hook = Distill(root, WORKER/'manifest.json', context)
                scene = manifest['episodes'][0]
                hook.start(1, scene, 'offline', '4.10.0.75689')
                ctrl.reset()
                runtime.episode_seed = scene['sc2_seed']
                ctrl.manager.micro_all_attackers = True
                for step, payload in enumerate(rows):
                    def mask(i):
                        q = payload['questions'].get(f'u{i}')
                        ids = set(map(int, q['criteria'])) if q else {0}
                        return [j in ids for j in range(16)]
                    env = SimpleNamespace(n_agents=10, get_avail_agent_actions=mask)
                    if not ctrl.active:
                        ctrl.select('adaptive_focus_fire', posture='balanced')
                    a, s = hook.step(step, payload['state']['game_loop'], ctrl, payload, env)
                    hook.before_commit(ctrl.manager, a)
                    ctrl.commit_final_actions(a)
                    hook.committed(ctrl.manager, a)
                    if index == 0:
                        self.assertEqual(a, receipts[step]['actions'])
                        hook.engine(receipts[step]['feedback'])
                    else:
                        hook.engine({})  # no counterfactual engine acknowledgement
                    self.assertEqual(hook.decision['final_actions'], s['actions'])
                    if index is None and {'model_base', 'model_switch', 'fallback'} <= {d['mode'] for d in hook.receipt['decisions']}:
                        break
                    if index == 0 and hook.decision['micro_receipt'] == 'accepted' and any(
                            d['mode'] == expected for d in hook.receipt['decisions']):
                        break
                    if index == -1 and hook.decision['mode'] == expected and step >= 1:
                        break
                hook.finish(step + 1)
                if index is None:
                    self.assertTrue(all(hook.totals[k] for k in ('model_base', 'model_switch', 'fallback')), hook.totals)
                else:
                    self.assertGreater(hook.totals[expected], 0)
                if index == 0:
                    self.assertTrue(any(d['micro_receipt'] == 'accepted' for d in hook.receipt['decisions']))
                else:
                    self.assertEqual(hook.totals['engine_unconfirmed'], step + 1)
                counts = hook.totals
                self.assertEqual(counts['steps'], counts['fallback'] + counts['model_base'] + counts['model_switch'])


if __name__ == '__main__':
    unittest.main()
