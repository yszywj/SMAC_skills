"""Exercise the frozen router on real pre-action observations, with no game process."""
import asyncio
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bridge import runner
from bridge.zerg_swarm_route import FROZEN_GATE, OnlineRoute
from test_zerg_shadow_runner import OfflineEnv


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT/'runs/zerg_v2_vs_swarm_random900_20261124_223951'
GATE = ROOT/'runs/zerg_swarm_explorer_900_v1/gate.json'


def read(path):
    return json.loads(path.read_bytes())


@unittest.skipUnless(GATE.exists() and PANEL.exists(), 'frozen archive not installed')
class RouterTests(unittest.TestCase):
    def setUp(self):
        self.policy = read(PANEL/'design.json')['policies']['v2']
        self.env = dict(ZERG_SWARM_GATE=str(GATE), ZERG_SWARM_GATE_SHA256=FROZEN_GATE,
                        ZERG_SWARM_SHA256=read(GATE)['policy_hashes']['radical_swarm'])

    def test_prefix_micro_is_irreversible(self):
        route = OnlineRoute(self.env, self.policy, self.env['ZERG_SWARM_SHA256'])
        route.reset()
        with patch.object(route.swarm, 'choose', return_value={'actions':[2]}) as choose:
            self.assertIsNone(route.decide({}, [], [2])['choice'])
            self.assertFalse(route.after_engine([]))
            self.assertTrue(route.after_engine([{'status':'submitted'}]))
            self.assertEqual(route.decide({}, [], [4])['actions'], [4])
            self.assertEqual(route.choice, 'v2')
            self.assertEqual(choose.call_count, 1)
        with self.assertRaises(ValueError):
            OnlineRoute(self.env, dict(self.policy, opening_sha256='wrong'), self.env['ZERG_SWARM_SHA256'])

    def test_real_parent_and_swarm_on_archived_first_beat(self):
        choices = set()
        for pid in ('pair-036', 'pair-347', 'pair-810', 'pair-001'):
            pair = PANEL/pid
            manifest = read(pair/'manifest.json')
            meta = read(pair/'v2/runner/run.json')
            def first(arm, event):
                with (pair/arm/'runner/events.jsonl').open() as stream:
                    return next(json.loads(line) for line in stream if line.startswith('{"event": "'+event+'",'))
            obs = first('v2', 'skill_observation')['payload']
            left = first('v2', 'step')['actions']
            other = first('swarm', 'swarm_decision')
            clock = obs['state']['game_loop']

            class Env(OfflineEnv):
                n_agents = n_enemies = 10
                n_actions = 16
                _episode_steps = episode_limit = 1

                def __init__(self, config):
                    super().__init__(config)
                    self._obs.observation.game_loop = clock
                    self._controller.data = lambda: SimpleNamespace(units={107:'hydralisk'})
                    self._controller.ping = lambda: SimpleNamespace(game_version='4.10.0.75689')

                def get_avail_agent_actions(self, i):
                    return other['avail_actions'][i]

                def step_micro_window(self, actions, window, macro_loops):
                    self._obs.observation.game_loop += macro_loops
                    return 0, True, dict(dead_allies=0, dead_enemies=0)

            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                env = dict(self.env, ZERG_DISTILL_VERSION='v2', ZERG_DISTILL_MODEL=str(PANEL/'model.json'),
                    ZERG_DISTILL_MODEL_SHA256=self.policy['model_sha256'], ZERG_DISTILL_TRIAL_ID=self.policy['trial_id'],
                    ZERG_DISTILL_SELECTOR_SHA256=self.policy['selector_sha256'], ZERG_DISTILL_OPENING_SHA256=self.policy['opening_sha256'],
                    ZERG_SHADOW_ROSTER=meta['zerg_shadow']['scene_config_id'], ZERG_SHADOW_DIR=tmp+'/shadow',
                    SMAC_DAMAGE_MAP_DIR=str(Path(meta['map_file']).parent))
                for mock in (patch.dict(os.environ, env), patch('bridge.environment.ReplayableEnv', Env),
                        patch('bridge.policy.build_request', return_value=obs), patch('bridge.feedback.action_feedback',return_value={}),
                        patch('bridge.environment.audit_unit_balance', return_value={}), patch('scripts.make_damage_map.begin_resource_damage'),
                        patch('scripts.make_damage_map.read_resource_damage',return_value=[0]*8), patch.object(runner,'score_snapshot',
                        side_effect=lambda e: dict(game_loop=e._obs.observation.game_loop, score_details={k:{'life':0} for k in
                        ('total_damage_taken','total_damage_dealt')}))):
                    stack.enter_context(mock)
                out = Path(tmp)/'runner'
                args = runner.parser().parse_args(['run','--method','adaptive-skill','--skill-version','tactical_v1',
                    '--execution-mode','micro8-all','--manifest','unused','--output-dir',str(out)])
                asyncio.run(runner.evaluate(args, manifest, 'adaptive-skill', out))
                events = [json.loads(s) for s in (out/'events.jsonl').read_text().splitlines()]
                d, = [e for e in events if e['event']=='swarm_route_decision']
                self.assertEqual((d['v2_actions'], d['swarm_decision']), (left, other['decision']))
                self.assertEqual(d['reason'], 'first_macro_difference')
                choices.add(d['choice'])
                executed, = [e for e in events if e['event']=='step']
                self.assertEqual(executed['actions'], other['decision']['actions'] if d['choice']=='swarm' else left)
                prefix = read(out/'distill/episode-0001.json')
                self.assertEqual(prefix['steps'], int(d['choice']=='v2'))
                if d['choice']=='swarm':
                    self.assertFalse(d['uncommitted_v2']['committed'])
                    self.assertFalse(read(out/'micro_triggers/episode-0001.json')['authorizations'])
        self.assertEqual(choices, {'swarm','v2'})
