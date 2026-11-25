"""Real R6/R8 controllers on archived pair 001, with a scripted offline engine."""
import asyncio
from contextlib import ExitStack
import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bridge import runner
from bridge.zerg_relay import GUARD, SOURCE, check_options
from bridge.zerg_squad import ADAPTIVE_SHA256
from test_zerg_swarm_route import ROOT, read
from test_zerg_shadow_runner import OfflineEnv

ARCHIVE = ROOT/'runs/zerg_R1_R6_R7_random1000_20261125_155323'


class RelayTests(unittest.TestCase):
    def test_invalid_opt_in(self):
        with self.assertRaises(ValueError):
            check_options({'ZERG_RELAY_MODE':'full'}, 'adaptive-skill')

    @unittest.skipUnless(ARCHIVE.is_dir(), 'requires R6 fixtures')
    def test_real_menu_and_custom(self):
        for mode in ('menu','full'):
            with self.subTest(mode=mode):
                self.replay(mode)

    def replay(self, mode):
        pair = ARCHIVE/'pair-001'
        meta = read(pair/'R6/runner/run.json')
        policy = meta['zerg_distill']
        with (pair/'R6/runner/events.jsonl').open() as stream:
            obs = list(itertools.islice((json.loads(s)['payload'] for s in stream
                if s.startswith('{"event": "skill_observation"')), 16))
        executed = []

        class Env(OfflineEnv):
            n_agents = n_enemies = 10
            n_actions = 16
            _episode_steps = episode_limit = len(obs)

            def __init__(self, config):
                super().__init__(config)
                self._obs.observation.game_loop = obs[0]['state']['game_loop']
                self._controller.data = lambda: SimpleNamespace(units={107:'hydralisk'})
                self._controller.ping = lambda: SimpleNamespace(game_version='4.10.0.75689')

            def get_avail_agent_actions(self, i):
                criteria = obs[len(executed)]['questions'].get(f'u{i}', {}).get('criteria', {'0':None})
                return [str(a) in criteria for a in range(16)]

            def step_micro_window(self, actions, window, macro_loops):
                executed.append(list(actions))
                self._obs.observation.game_loop += macro_loops
                return 0, len(executed) == len(obs), dict(dead_allies=0, dead_enemies=0)

        def feedback(env, loop):
            return dict(response_mapping='exact', from_game_loop=loop, raw_response_results=[1 for a in executed[-1] if a],
                        actors={f'u{i}':dict(requested_action_id=str(a), command={'kind':'test'} if a else None,
                                response={'code':1} if a else None, submission='accepted' if a else 'dead_noop')
                                for i,a in enumerate(executed[-1])})

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            env = dict(ZERG_SQUAD_MODE='guard', ZERG_SQUAD_SHA256=ADAPTIVE_SHA256, ZERG_SQUAD_CROSS_SHA256=GUARD,
                ZERG_RELAY_MODE=mode, ZERG_RELAY_SHA256=hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest(),
                ZERG_DISTILL_VERSION='v2', ZERG_DISTILL_MODEL=str(ARCHIVE/'model.json'),
                ZERG_DISTILL_MODEL_SHA256=policy['model_sha256'], ZERG_DISTILL_TRIAL_ID=policy['trial_id'],
                ZERG_DISTILL_SELECTOR_SHA256=policy['selector_sha256'], ZERG_DISTILL_OPENING_SHA256=policy['opening_sha256'],
                ZERG_SHADOW_ROSTER=meta['zerg_shadow']['scene_config_id'], ZERG_SHADOW_DIR=tmp+'/shadow',
                SMAC_DAMAGE_MAP_DIR=str(Path(meta['map_file']).parent))
            out = Path(tmp)/'runner'
            for mock in (patch.dict(os.environ, env), patch('bridge.environment.ReplayableEnv', Env),
                    patch('bridge.policy.build_request', side_effect=obs), patch('bridge.feedback.action_feedback',feedback),
                    patch('bridge.environment.audit_unit_balance', return_value={}), patch('scripts.make_damage_map.begin_resource_damage'),
                    patch('scripts.make_damage_map.read_resource_damage',return_value=[0]*8), patch.object(runner,'score_snapshot',
                    side_effect=lambda e: dict(game_loop=e._obs.observation.game_loop, score_details={k:{'life':0} for k in
                    ('total_damage_taken','total_damage_dealt')}))):
                stack.enter_context(mock)
            args = runner.parser().parse_args(['run','--method','adaptive-skill','--skill-version','tactical_v1',
                '--execution-mode','micro8-all','--manifest','unused','--output-dir',str(out)])
            asyncio.run(runner.evaluate(args, read(pair/'manifest.json'), 'adaptive-skill', out))
            events = [json.loads(s) for s in (out/'events.jsonl').read_text().splitlines()]
            rows = [e for e in events if e['event']=='combat_relay_decision']
            receipts = [e for e in events if e['event']=='combat_relay_receipt']
            self.assertEqual([r['actions'] for r in receipts], executed)
            self.assertEqual([r['actions'] for r in rows], executed)
            self.assertIn('menu' if mode=='menu' else 'custom', [r['owner'] for r in rows])
            self.assertTrue(any(r['handoff_to_r6'] for r in receipts))
            for r,ack in zip(rows,receipts):
                if r['owner']=='custom':
                    self.assertEqual(ack['plan_ids'], [])
                    self.assertFalse(set(r['changed_actors']) & {g['actor'] for g in ack['micro_grants']})
                if r['owner']=='menu':
                    self.assertIn(r['actions'], r['candidates'].values())
            native = [r for r in rows if r['owner'] in ('menu','r6')]
            self.assertEqual([d['step'] for d in read(out/'distill/episode-0001.json')['decisions']],
                             [r['step'] for r in native])
