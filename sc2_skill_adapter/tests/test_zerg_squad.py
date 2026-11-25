"""Real frozen v2, scripted ownership and recorded observations; no SC2 process."""
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
from bridge.zerg_squad import CROSS_SOURCE, SOURCE, SquadRuntime, action_map, check_options
from decision_tree.zerg import squad_cross_support, squad_rendezvous
from test_zerg_shadow_runner import OfflineEnv
from test_zerg_swarm_route import PANEL, ROOT, read


class SquadTests(unittest.TestCase):
    def test_incomplete_opt_in_fails_before_launch(self):
        for env in ({'ZERG_SQUAD_SHA256':'x'}, {'ZERG_SQUAD_MODE':'fixed'}, {'ZERG_SQUAD_MODE':'typo'}):
            with self.assertRaises(ValueError):
                check_options(env, 'adaptive-skill')

    @unittest.skipUnless(PANEL.exists(), 'requires historical fixtures')
    def test_macro_memory_and_native_suffix(self):
        for mode in ('fixed', 'guard', 'front'):
            with self.subTest(mode=mode):
                self.check_scripted_episode(mode)

    def check_scripted_episode(self, mode):
        pair = PANEL/'pair-036'
        manifest = read(pair/'manifest.json')
        meta = read(pair/'v2/runner/run.json')
        policy = meta['zerg_distill']
        with (pair/'v2/runner/events.jsonl').open() as stream:
            obs = list(itertools.islice((json.loads(s)['payload'] for s in stream
                        if s.startswith('{"event": "skill_observation"')), 4))
        clock = obs[0]['state']['game_loop']
        executed, windows, memory = [], [], []

        class Env(OfflineEnv):
            n_agents = n_enemies = 10
            n_actions = 16
            _episode_steps = episode_limit = 4

            def __init__(self, config):
                super().__init__(config)
                self._obs.observation.game_loop = clock
                self._controller.data = lambda: SimpleNamespace(units={107:'hydralisk'})
                self._controller.ping = lambda: SimpleNamespace(game_version='4.10.0.75689')

            def get_avail_agent_actions(self, i):
                return [str(a) in obs[len(executed)]['questions'][f'u{i}']['criteria'] for a in range(16)]

            def step_micro_window(self, actions, window, macro_loops):
                executed.append(list(actions))
                windows.append((window.execute, window.all_attackers))
                self._obs.observation.game_loop += macro_loops
                return 0, len(executed) == 4, dict(dead_allies=0, dead_enemies=0)

        def propose(_, payload, masks, base):
            n = len(executed)
            # Beat 0 differs; beat 1 deliberately coincides but is still Squad.
            a = [next((j for j,v in enumerate(m) if v and j != b), b) for m,b in zip(masks,base)] if not n else base
            return dict(actions=a, phase='v2' if n >= 2 else 'rendezvous', handoff=n >= 2, reason='test',
                        veto=[{'actor': 0}], front={'changes': []})

        def feedback(env, loop):
            return dict(response_mapping='exact', from_game_loop=loop, raw_response_results=[1]*10,
                        actors={u:dict(requested_action_id=str(a), command={'kind':'test'}, response={'code':1},
                                       submission='accepted') for u,a in action_map(executed[-1]).items()})

        acknowledge = SquadRuntime.acknowledge_squad
        def checked(rt, manager, fb):
            self.assertIsNone(manager._pending_commit)
            self.assertFalse(manager.last_micro_intents)
            acknowledge(rt, manager, fb)
            self.assertFalse(manager.memory.instances)
            memory.append(manager.memory.last_committed_actions.copy())

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            env = dict(ZERG_SQUAD_MODE=mode, ZERG_SQUAD_SHA256=hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest(),
                ZERG_DISTILL_VERSION='v2', ZERG_DISTILL_MODEL=str(PANEL/'model.json'),
                ZERG_DISTILL_MODEL_SHA256=policy['model_sha256'], ZERG_DISTILL_TRIAL_ID=policy['trial_id'],
                ZERG_DISTILL_SELECTOR_SHA256=policy['selector_sha256'], ZERG_DISTILL_OPENING_SHA256=policy['opening_sha256'],
                ZERG_SHADOW_ROSTER=meta['zerg_shadow']['scene_config_id'], ZERG_SHADOW_DIR=tmp+'/shadow',
                SMAC_DAMAGE_MAP_DIR=str(Path(meta['map_file']).parent))
            out = Path(tmp)/'runner'
            factory = squad_rendezvous.SquadRendezvous
            if mode != 'fixed':
                env['ZERG_SQUAD_CROSS_SHA256'] = hashlib.sha256((ROOT/CROSS_SOURCE).read_bytes()).hexdigest()
                factory = squad_cross_support.CrossSupportRendezvous
            for mock in (patch.dict(os.environ, env), patch('bridge.environment.ReplayableEnv', Env),
                    patch.object(factory, 'propose', propose),
                    patch.object(SquadRuntime, 'acknowledge_squad', checked),
                    patch('bridge.policy.build_request', side_effect=obs), patch('bridge.feedback.action_feedback',feedback),
                    patch('bridge.environment.audit_unit_balance', return_value={}), patch('scripts.make_damage_map.begin_resource_damage'),
                    patch('scripts.make_damage_map.read_resource_damage',return_value=[0]*8), patch.object(runner,'score_snapshot',
                    side_effect=lambda e: dict(game_loop=e._obs.observation.game_loop, score_details={k:{'life':0} for k in
                    ('total_damage_taken','total_damage_dealt')}))):
                stack.enter_context(mock)
            args = runner.parser().parse_args(['run','--method','adaptive-skill','--skill-version','tactical_v1',
                '--execution-mode','micro8-all','--manifest','unused','--output-dir',str(out)])
            asyncio.run(runner.evaluate(args, manifest, 'adaptive-skill', out))
            events = [json.loads(s) for s in (out/'events.jsonl').read_text().splitlines()]
            rows = [e for e in events if e['event']=='squad_rendezvous_decision']
            if mode != 'fixed':
                for row in rows:
                    self.assertEqual(row['skill_audit'], dict(actions=row['actions'], phase=row['phase'],
                        handoff=row['handoff'], reason='test', veto=[{'actor': 0}], front={'changes': []}))
            self.assertNotEqual(rows[0]['v2_actions'], executed[0])
            self.assertEqual(rows[1]['v2_actions'], executed[1])
            self.assertEqual(memory, list(map(action_map, executed[:2])))
            self.assertEqual(windows, [(False,False)]*2+[(True,True)]*2)
            self.assertEqual([e['step'] for e in events if e['event']=='skill_step'], [2,3])
            self.assertEqual(read(out/'episodes.json')[0]['squad_rendezvous'], dict(squad_steps=2,v2_steps=2))
            receipt = read(out/'distill/episode-0001.json')
            self.assertEqual([(d['step'],d['final_actions'],d['macro_receipt']) for d in receipt['decisions']],
                             [(i,action_map(executed[i]),'accepted') for i in (2,3)])
            rows = [json.loads(s) for s in (Path(tmp)/'shadow/shadow.jsonl').read_text().splitlines()]
            self.assertEqual([r['game_loop'] for r in rows], [clock+16,clock+24])
