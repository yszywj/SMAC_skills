"""Offline report tests. No SC2 process, adapter runtime or generated game data required."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import contact_muster_report as report


def frame(i, *, visible=True, mask=True, unit_types=('Zergling', 'Hydralisk', 'Hydralisk'), label='baseline'):
    loop = 26 + 8*i
    # Hydra u1 covers the first dog's cluster. Others have no legal shot there.
    allies = {f'u{j}': dict(type=u, tag=str(j+1), alive=True, position=[1, 2],
                            weapon_cooldown_loops=0) for j, u in enumerate(unit_types)}
    enemies = {'e0': dict(type='Zergling', alive=True, position=[1, 2])} if visible else {}
    relations, questions = {}, {}
    for actor in allies:
        legal = visible and mask and actor in ('u0', 'u1')
        relations[actor] = {e: dict(smac_attack_available=legal, in_weapon_range=True,
                                    in_enemy_weapon_range=False) for e in enemies}
        questions[actor] = dict(criteria=dict.fromkeys(['1', '4'] + (['6'] if legal else [])))
    state = dict(step_mul=8, game_loop=loop, visible_enemies=enemies, allies=allies, relations=relations)
    obs = dict(step=i, payload=dict(state=state, questions=questions))
    decision = dict(step=i, game_loop=loop, actions=dict.fromkeys(allies, 4))
    if label == 'muster':
        decision['opening_muster'] = dict(changes={'u0': dict(before=1, after=4)} if i == 0 else {},
                                         held=list(allies), released={})
    receipt = dict(requested_action_id='4', response={'code': 1}, submission='accepted', command={'kind': 'move'})
    step = dict(step=i+1, game_loop=loop, actions=[4]*len(allies), feedback=dict(from_game_loop=loop,
                actors={a: deepcopy(receipt) for a in allies}))
    return obs, decision, step


class BehaviorTests(unittest.TestCase):
    def game(self, frames, label='baseline'):
        g = dict(ally_hydras=2)
        report._behavior(frames, g, label, 'test')
        return g

    def test_window_and_support(self):
        frames = [frame(i, visible=i >= 1, mask=i >= 2) for i in range(9)]
        # Restarts at first visibility and after +48 must not count, at +48 counts.
        for i, actor in [(1, 'u1'), (7, 'u2'), (8, 'u1')]:
            frames[i][0]['payload']['state']['allies'][actor]['weapon_cooldown_loops'] = 9.5
        g = self.game(frames)
        self.assertEqual((g['first_dog_mask_since_start'], g['first_dog_mask_since_visible']), (16, 8))
        self.assertEqual(g['ally_support_fraction'], .5)
        self.assertEqual(g['first_ally_cd_since_visible'], 0)
        self.assertIsNone(g['first_dog_cd_since_start'])
        self.assertEqual(g['hydra_cd_participant_tags_48'], ['3'])
        self.assertEqual(g['hydra_cd_fraction_48'], .5)
        for frames, status, fraction in [([frame(0, visible=False)], 'no_visibility', None),
                ([frame(i, mask=False) for i in range(6)], 'truncated', None),
                ([frame(i, mask=False) for i in range(7)], 'complete', 0)]:
            g = self.game(frames)
            self.assertIsNone(g['ally_support_fraction'])
            self.assertIsNone(g['first_dog_mask_since_visible'])
            self.assertEqual((g['hydra_cd_window_status'], g['hydra_cd_fraction_48']), (status, fraction))

    def test_receipt_not_candidate_or_shot(self):
        f = frame(0, label='muster')
        changes = self.game([f], 'muster')['opening_muster_changes']
        self.assertEqual([c['actor'] for c in changes], ['u0'])
        f[2]['feedback']['actors']['u0'].update(submission='rejected', response={'code': 3})
        self.assertEqual(self.game([f], 'muster')['opening_muster_changes'][0]['receipt']['submission'], 'rejected')
        for component, path, value in [(2, ('actions', 0), 1), (2, ('feedback','actors','u0','requested_action_id'), '1'),
                (1, ('game_loop',), 99), (1, ('opening_muster','changes','u0','before'), 4),
                (2, ('feedback','actors','u0','response','code'), 1)]:
            bad = deepcopy(f)
            obj = bad[component]
            for k in path[:-1]:
                obj = obj[k]
            obj[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.game([bad], 'muster')


class BatchTests(unittest.TestCase):
    """Emit real-format five-game event streams and metadata; mock only the replay decoders."""
    @classmethod
    def setUpClass(cls):
        tmp = TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        b = cls.batch = Path(tmp.name)
        (b / 'manifests').mkdir()
        write = report.write_json
        design = dict(labels=list(report.LABELS), execution_mode=report.MODE, seed=100, shards={}, policy_files={})
        for label in report.LABELS:
            files = design['policy_files'][label] = {}
            for k in ('skills/tactical_v1/manager.py', 'sc2_skill_adapter/bridge/micro_trigger.py',
                      'sc2_skill_adapter/damage_maps/10gen_zerg.SC2Map'):
                p = b / label / k
                p.parent.mkdir(parents=True)
                data = (label if k.startswith('skills') else k).encode()
                p.write_bytes(data)
                files[k] = hashlib.sha256(data).hexdigest()
        env = dict(map_name='10gen_zerg', unit_balance='native', step_mul=8, use_unit_ranges=True, min_attack_range=2,
                   capability_config=dict(n_units=10, n_enemies=10, team_gen=dict(unit_types=['zergling','hydralisk']),
                                          start_positions={}))
        timing = dict(mode=report.MODE, macro_loops=8, engine_step_loops=1)
        for group,(name, roster) in enumerate(report.COMPOSITIONS.items()):
            units = list(Counter(roster).elements())
            team = [s.lower() for s in units]
            for offset in range(0,75,5):
                stem = f'{name}_{offset:02d}'
                episodes = []
                for i in range(5):
                    x = 16 if i == 0 else i
                    cfg = dict(team_gen=dict(ally_team=team,enemy_team=team), ally_start_positions=dict(item=[[x,16]]*10),
                               enemy_start_positions=dict(item=[[32-x,16]]*10))
                    episodes.append(dict(index=offset+i, sc2_seed=100+75*group+offset+i, episode_config=cfg,
                                         sha256=report.digest(cfg)))
                m = dict(version=1, seed=100+75*group, env_args=env, episodes=episodes)
                m['sha256'] = report.digest(m)
                design['shards'][stem] = dict(composition=name, manifest_sha256=m['sha256'])
                write(b/'manifests'/(stem+'.json'), m)
                for label in report.LABELS:
                    w = b / label / stem
                    for sub in ('damage_by_type','score_details','micro_triggers'):
                        (w/sub).mkdir(parents=True)
                    files = design['policy_files'][label]
                    meta = dict(method='adaptive-skill', skill_version='tactical_v1', execution_mode=report.MODE,
                                manifest_sha256=m['sha256'])
                    write(w/'manifest.json',m)
                    write(w/'run.json',dict(meta, macro_loops=8, engine_step_loops=1, env_args=env, versions={},
                          skill_parameters={}, source_hashes=files, damage_by_type_enabled=True,
                          skill_source_hashes={k:v for k,v in files.items() if k.startswith('skills/')}))
                    write(w/'summary.json',dict(meta,status='complete',completed_episodes=5))
                    events,records = [],[]
                    for ep,e in enumerate(episodes,1):
                        filename = f'episode-{ep:04d}.json'
                        win = ep == (2 if label == 'baseline' else 1)
                        g = dict(episode=ep,scenario_sha256=e['sha256'],steps=1,initial_state_sha256=report.digest(e),
                                 outcome='win' if win else 'timeout', info=dict(battle_won=win,dead_allies=0,dead_enemies=0),
                                 micro_trigger=dict(timing,windows=1,authorizations=0,states={}))
                        records.append(g)
                        events.append(dict(event='episode_started',episode=ep,game_loop=26,scenario_sha256=e['sha256'],
                                           initial_state_sha256=g['initial_state_sha256']))
                        events.extend(dict(f,event=kind,episode=ep) for kind,f in zip(('skill_observation','skill_step','step'),
                                      frame(0,label=label,unit_types=units)))
                        events.append(dict(g,event='episode_completed'))
                        players = {}
                        for side in (1,2):
                            values = dict.fromkeys(report.TYPES, 0)
                            values['Zergling'] = side + int(label == 'muster')
                            players[str(side)] = dict(initial_roster_by_api_tag={str((side-1)*10+j):u for j,u in enumerate(units,1)},
                                ticks_by_source_type={k:v*4096 for k,v in values.items()},damage_dealt_by_source_type=values)
                        write(w/'damage_by_type'/filename,dict(meta, episode=ep,scenario_sha256=e['sha256'],players=players,
                              ticks_per_damage=4096,begin_request_loop=26,terminal_loop=34,freeze_loop_range=[34,36]))
                        write(w/'score_details'/filename,dict(meta, episode=ep, scenario_sha256=e['sha256'],
                              sc2_seed=e['sc2_seed'],steps=1,outcome=g['outcome'],initial={'game_loop':26},final={'game_loop':34}))
                        write(w/'micro_triggers'/filename,dict(timing,episode=ep,authorizations=[]))
                    write(w/'episodes.json',records)
                    (w/'events.jsonl').write_text('\n'.join(map(json.dumps, events))+'\n')
        write(b/'design.json',design)

    def setUp(self):
        def replay(path):
            bonus = int(path.parents[2].name == 'muster')
            return [dict(freeze_loop_range=[34,36], ticks=[(1+bonus)*4096,0,0,0,(2+bonus)*4096,0,0,0])]
        for stub in [patch('contact_early_report.replay_resource_damage', replay), patch('contact_early_report._deaths',return_value={})]:
            stub.start()
            self.addCleanup(stub.stop)

    def test_150_pairs(self):
        r = report.summarize(self.batch)
        self.assertEqual(r, report._read(self.batch/'comparison.json'))
        self.assertEqual(len(r['episodes']),150)
        for scope,pairs,gains,drops in [(r['total'],150,30,30),(r['by_layout']['center'],30,30,0),
                (r['by_layout']['scattered'],120,0,30), (next(iter(r['by_composition'].values())),75,15,15)]:
            self.assertEqual(scope['pairs'],pairs)
            self.assertEqual(scope['paired_vs_baseline']['muster'],dict(gained_wins=gains,dropped_wins=drops))
            stats = scope['versions']['muster']
            self.assertEqual(stats['metrics']['hydra_cd_fraction_48'],dict(n=0,missing=pairs,mean=None))
            self.assertEqual(stats['damage_by_type']['enemy']['Zergling'], 3*pairs)
            self.assertEqual(stats['opening_muster']['receipts'],{'accepted':pairs})

    def test_invalid_design_and_execution_preserve_prior_report(self):
        for path, key, wrong in [(self.batch/'design.json', 'labels', ['baseline','fake']),
                (self.batch/'baseline'/(next(iter(report.COMPOSITIONS))+'_00')/'run.json', 'execution_mode','legacy8')]:
            original = path.read_text()
            previous = self.batch/'comparison.json'
            previous.write_text('previous')
            report.write_json(path, dict(report._read(path), **{key:wrong}))
            try:
                with self.assertRaises(ValueError):
                    report.summarize(self.batch)
                self.assertEqual(previous.read_text(),'previous')
            finally:
                path.write_text(original)

    def test_paired_initial_state_checked_independently_of_artifact_consistency(self):
        folder = self.batch / 'muster' / (next(iter(report.COMPOSITIONS)) + '_00')
        old = report._read(folder/'episodes.json')[0]['initial_state_sha256']
        for path in (folder/'episodes.json', folder/'events.jsonl'):
            original = path.read_text()
            self.addCleanup(path.write_text, original)
            path.write_text(original.replace(old, '0'*64))
        with self.assertRaisesRegex(ValueError, 'initial state mismatch'):
            report.summarize(self.batch)


if __name__ == '__main__':
    unittest.main()
