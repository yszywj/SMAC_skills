"""Offline tests for round-2 reporting; no game or policy imports."""
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

if __package__:
    from . import hydra_supported_report as r
else:
    import hydra_supported_report as r


def frame(index=0, lane=True, cd=0, order='attack'):
    return dict(step=index, state_hash=str(index), live={'u0': '7', 'u1': '8'}, dead=set(),
                cooldown={'u0': cd, 'u1': cd}, order_kind={'u0': order}, visible={'9', '10'},
                attack_target={'u0': '10' if order == 'attack' else None},
                lanes={'9': {'u0': True, 'u1': lane}, '10': {'u1': True}})


def step(action):
    cmd = dict(ability_id={3: 16, 7: 23, 1: 4}[action], target_tag='10' if action == 7 else None)
    return [action], dict(actors=dict(u0=dict(command=cmd, requested_action_id=str(action),
                         submission='accepted', response=dict(code=1))), micro_submissions=[], response_mapping='exact')


B = dict(switch='B2', actor='u0', actor_tag='7', game_loop=10, target_tag='10', current_target='e1',
         support_target='e0', support_target_tag='9', supporting_for=['u1'], original_action=3, final_action=7)
A = dict(switch='A2', actor='u0', actor_tag='7', game_loop=10, support_class='if_pursued',
         support_mate='u1', support_mate_tag='8', support_threat='e0', support_threat_tag='9')


class ReportTests(unittest.TestCase):
    def test_named_mate_original_threat_not_other_enemy(self):
        frames = {10: frame(), 18: frame(1, lane=False, cd=7)}
        follow = dict(actor='u0', game_loop=18, alive=True, threat_state={'e1': 'visible_in_designated_mate_range'},
                      arrived_at_predicted=False, final=True, designated_mate_fire_proxy=True)
        c = Counter()
        r._followup(A, [follow], 34, c, frames)
        self.assertEqual(c['route_estimate:if_pursued'], 1)
        self.assertEqual(c['designated_lane_8:outside'], 1)
        self.assertEqual(c['threat_in_designated_mate_range'], 0)
        self.assertEqual(c['arrival:terminal_censored'], 1)
        frames[18]['lanes']['9']['u1'] = True
        c.clear()
        r._followup(A, [], 80, c, frames)
        self.assertEqual(c['designated_mate_fire_proxy_while_covered:restart'], 1)
        frames[18]['live']['u1'] = 'replacement'
        c.clear()
        r._followup(A, [], 80, c, frames)
        self.assertEqual(c['designated_lane_8:missing'], 1)

    def test_missing_dead_terminal_not_failed_escape(self):
        for terminal, deaths, expected in [(80, None, 'missing_observation'), (26, {}, 'terminal_censored'),
                                            (80, {'7': 14}, 'arrival_unknown')]:
            c = Counter()
            r._followup(A, [dict(actor='u0', game_loop=18, alive=None, final=True)], terminal,c,{},deaths)
            self.assertEqual(c['arrival:'+expected], 1)

    def test_mate_line_cost_and_actor_shot_do_not_share_target(self):
        f = {10: frame(), 18: frame(1, lane=False, cd=3)}
        base = {10: frame(), 18: frame(1, lane=True, order='stop')}
        pair = r._paired_b2(B, base, f, (34,34))
        self.assertEqual(pair['baseline'], dict(own_next_8='no_restart_observed', own_lane_on_support_target_next_8='covered', original_mate_lanes_next_8={'u1':'covered'}))
        self.assertEqual(pair['variant'], dict(own_next_8='restart_matching_end_order', own_lane_on_support_target_next_8='covered', original_mate_lanes_next_8={'u1':'outside'}))
        self.assertEqual(r._receipt({(2,10):step(7)},2,B), 'accepted')
        self.assertNotEqual(r._receipt({(2,10):step(1)},2,dict(B,final_action=1)), 'accepted')
        self.assertNotEqual(r._receipt({(2,10):step(7)},2,dict(B,target_tag=None)), 'accepted')
        self.assertEqual(r._paired_b2(B, {}, {}, (18,None))['baseline']['original_mate_lanes_next_8']['u1'], 'terminal_censored')

    def test_only_same_observed_prefix_can_match(self):
        f = {2:frame(),10:frame(1)}
        g = {2:step(1),10:step(7)}
        b = {2:step(1),10:step(3)}
        def compare(frames):
            return r._divergence(g,b,frames,f,[B],None)
        self.assertTrue(compare(f)['matches_switch_window'])
        self.assertFalse(compare({2:f[2],10:dict(f[10],state_hash='different enemy hp')})['matches_switch_window'])
        self.assertEqual(compare({10:f[10]})['kind'], 'incomplete_prefix')
        b[2][1]['micro_submissions'].append(dict(submitted_loop=6, actor_tag='8', target_tag='9', sc2_response=[1]))
        self.assertEqual(compare(f)['first_divergence'], 6)
        self.assertFalse(compare(f)['matches_switch_window'])

    def test_episode_keys_and_missing_artifacts_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'m').mkdir()
            teams = ['zergling']*3 + ['hydralisk']*7
            cfg = dict(team_gen=dict(ally_team=teams,enemy_team=teams),
                       ally_start_positions=dict(item=[[16,16]]*10), enemy_start_positions=dict(item=[[20,16]]*10))
            sha = r._digest(cfg)
            manifest = dict(episodes=[dict(sha256=sha,episode_config=cfg,sc2_seed=3)])
            manifest['sha256'] = r._digest(manifest)
            (root/'m/s.json').write_text(json.dumps(manifest))
            design = dict(manifests='m',shards={'s':manifest['sha256']},labels=r.LABELS)
            results = {}
            for label in r.LABELS:
                p = root/label/'s'
                p.mkdir(parents=True)
                rec = dict(episode=1, steps=1, outcome='win', scenario_sha256=sha, initial_state_sha256=r._digest(2))
                results[label,'s'] = [rec]
                state = dict(allies={'u0':dict(type='Hydralisk',tag='7',alive=True,weapon_cooldown_loops=0,
                                               position=[0,0])},visible_enemies={},relations={'u0':{}},game_loop=10)
                action = 7 if label == 'guard_support_move' else 3
                actions, fb = step(action)
                events = [dict(event='step',episode=2,game_loop=2,actions=actions,feedback=fb),
                          dict(event='skill_observation',episode=1,step=0,payload=dict(state=state)),
                          dict(event='step',episode=1,game_loop=10,actions=actions,feedback=fb)]
                (p/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
                if action == 7:
                    (p/'hydra_supported').mkdir()
                    rows = [dict(B,changed=True,result='applied',candidate_action=7)]
                    (p/'hydra_supported/episode-0001.json').write_text(json.dumps(dict(episode=1,decisions=rows)))
                    rec['hydra_supported'] = dict(states={'B2:applied':1},decisions=1)
            report = r.build_report(root, design, results,{})
            group = report['groups']['guard_support_move']
            self.assertEqual(group['first_divergence'], {'matched_switch_window':1})
            self.assertEqual(group['damage_paired_games'], 0)
            self.assertIsNone(group['ally_damage_by_source_type'])
            results['both','s'][0]['initial_state_sha256'] = r._digest('different')
            self.assertEqual(r.build_report(root,design,results,{})['paired_scenarios'],0)


if __name__ == '__main__':
    unittest.main()
