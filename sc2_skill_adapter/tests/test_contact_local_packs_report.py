"""Offline traces and five-game shards; no SC2 binary or policy execution."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import contact_local_packs_report as r


def frame(i, label='packs_reserve', types=('Zergling', 'Hydralisk', 'Hydralisk'), visible=True):
    loop = 2 + 8*i
    units = {f'u{j}': dict(tag=str(j+1), type=t, alive=True, position=[1,0], orders=[], weapon_cooldown_loops=0.)
             for j,t in enumerate(types)}
    enemies = {'e0':dict(tag='11', type='Zergling', alive=True, position=[0,0])} if visible else {}
    relations, questions, actions, receipts = {}, {}, {}, {}
    for a,u in units.items():
        legal = visible and (a=='u0' or i>0)
        relations[a] = {e:dict(smac_attack_available=legal, in_weapon_range=legal, in_enemy_weapon_range=legal)
                        for e in enemies}
        questions[a] = dict(criteria=dict.fromkeys(['1','4'] + (['6'] if legal else [])))
        active = (i==0 and a=='u0') or (i>0 and a=='u1')
        if active and visible:
            u['orders'] = [dict(ability_id=3674, kind='other', target_tag='11')]
        act = 6 if legal and active else 4
        tag = '11' if act == 6 else None
        actions[a] = act
        receipts[a] = dict(requested_action_id=str(act), submission='accepted', response={'code':1},
                           command=dict(kind='attack' if tag else 'move', ability_id=23 if tag else 16, target_tag=tag))
    state = dict(game_loop=loop, step_mul=8, allies=units, visible_enemies=enemies, relations=relations)
    obs, decision = dict(step=i, payload=dict(state=state, questions=questions)), dict(step=i,game_loop=loop,actions=actions)
    if label!='baseline':
        trace = decision['local_coord'] = dict(packs=[], requests=[], selected=[], changes={},
            dispatch=label=='packs_reserve', reference_actions=actions.copy())
        decision['units'] = {a:dict(skill='baseline', plan=None, action=act) for a,act in actions.items()}
        if visible:
            p = dict(pack_id='P', members=['u0','u2'], enemies=['e0'], anchor_enemy_tag='11', anchor_ally_tag='1',
                     protected=['u0'], fire_rays={'u0':['e0']}, one_step_rays={})
            trace['packs'].append(p)
            if label=='packs_reserve':
                trace['requests'].append(dict(request_id='R', pack_id='P', target='e0', target_tag='11',
                    role='ranged_fire', protected=['u0'], slots=1, expires_loop=34, created_loop=2,
                    claimed_by=['u1'] if i else []))
            if not i:
                a = 'u1' if label=='packs_reserve' else 'u2'
                s = dict(actor=a, before=1, after=4, target='e0', target_tag='11', pack_id='P', changed=True,
                    skill='reserve_dispatch' if a=='u1' else 'local_engagement', role='reserve_approach' if a=='u1'
                    else 'approach_attack', request_id='R' if a=='u1' else None, plan_id='plan')
                trace['selected'].append(s)
                trace['reference_actions'][a] = s['before']
                decision['units'][a] = dict(skill=s['skill'],plan=s['plan_id'],action=s['after'])
                trace['changes'][a] = {k:s[k] for k in ('before','after','target_tag','pack_id','skill')}
    step = dict(step=i+1, game_loop=loop, actions=list(actions.values()), feedback=dict(actors=receipts,
                from_game_loop=loop,to_game_loop=loop+8))
    return obs,decision,step


def analyze(frames, deaths=None, label='packs_reserve', roster=None):
    g = dict(ally_hydras=2)
    r._behavior(frames,g,label,roster or {'11':'Zergling'},deaths or {},'test')
    return g['local_coord']


class BehaviorTests(unittest.TestCase):
    def test_named_followup_not_just_cd(self):
        fs = [frame(i) for i in range(3)]
        fs[1][0]['payload']['state']['allies']['u1'].update(weapon_cooldown_loops=6, orders=[])
        fs[2][0]['payload']['state']['allies']['u1']['weapon_cooldown_loops'] = 9
        audit = analyze(fs)
        e,member = audit['events']
        self.assertEqual((e['kind'],member['actor']), ('reserve','u0'))
        w8,w16 = (e['windows'][str(h)] for h in r.HORIZONS)
        self.assertTrue(w8['endpoint']['cd_restart'])
        self.assertIs(w8['new_named_cd_by_h'], False)
        self.assertIs(w8['new_ray_by_h'], True)
        self.assertIs(w16['new_named_cd_by_h'], True)
        self.assertEqual(member['windows']['8']['continuity'],'ray_or_order_lost')
        c = audit['counts']
        self.assertEqual([c[k] for k in ('pack_frames','pack_starts','request_frames','request_starts',
            'prior_claimant_frames','reserve_claim_starts','reserve_changed_actions')], [3,1,3,1,2,1,1])

    def test_censoring_and_unknown_range(self):
        for fs,deaths,status in [([frame(0)],{},'truncated'),([frame(0), frame(1,visible=False)],{},'target_unseen'),
                                  ([frame(0),frame(1,visible=False)],{'11':9},'target_dead')]:
            with self.subTest(status=status):
                e,m = analyze(fs,deaths)['events']
                w = e['windows']['8']
                self.assertEqual(w['endpoint'],dict(status=status))
                self.assertIsNone(w['new_ray_by_h'])
                self.assertIsNone(w['new_named_cd_by_h'])
                self.assertEqual(m['windows']['8']['continuity'],status)
        fs = [frame(0),frame(1),frame(2,visible=False)]
        fs[1][0]['payload']['state']['relations']['u1']['e0']['in_weapon_range'] = None
        self.assertIsNone(analyze(fs)['events'][0]['windows']['16']['new_ray_by_h'])
        fs[1][0]['payload']['state']['relations']['u1']['e0']['in_weapon_range'] = True
        self.assertTrue(analyze(fs)['events'][0]['windows']['16']['new_ray_by_h'])

    def test_continuation_not_new_dispatch_and_rejected_not_accepted(self):
        f0,f1 = frame(0),frame(1)
        s = dict(f0[1]['local_coord']['selected'][0],before=6,after=6,changed=False,continued=True)
        f1[1]['local_coord']['selected'] = [s]
        f1[1]['units']['u1'] = dict(skill=s['skill'],plan=s['plan_id'], action=6)
        f0[2]['feedback']['actors']['u1'].update(submission='rejected',response={'code':5})
        c = analyze([f0,f1])['counts']
        self.assertEqual([c[k] for k in ('reserve_claim_starts','reserve_selected_actor_frames','reserve_changed_actions',
                         'reserve_receipt_accepted','reserve_receipt_rejected')],[1,2,1,1,1])
        self.assertNotIn('reserve_new_ray_by_h_8_true',c)

    def test_inconsistent_ownership_and_receipt_fail_closed(self):
        for index,path,bad in [(1,('local_coord',), None), (1,('local_coord','changes'), {}),
                (1,('local_coord','selected',0,'target_tag'),'12'),(1,('local_coord','selected',0,'request_id'),'bad'),
                (1,('local_coord','changes','u1','before'),4),(1,('local_coord','reference_actions','u2'),1),
                (1,('units','u1','plan'),'other-plan'), (2,('feedback','actors','u0','command','target_tag'),'12'),
                (2,('feedback','actors','u1','requested_action_id'),'1'),(2,('feedback','actors','u1','response','code'),3),
                (2,('actions',1),1)]:
            f = frame(0)
            obj = f[index]
            for key in path[:-1]:
                obj = obj[key]
            obj[path[-1]] = bad
            with self.subTest(path=path), self.assertRaises(ValueError):
                analyze([f])
        for label in ('baseline','packs'):
            with self.subTest(label=label), self.assertRaises(ValueError):
                analyze([frame(0)],label=label)

    def test_multi_front_is_not_pincer(self):
        # No layout label is available to the classifier. Two disjoint fights,
        # then the second enemy closes from the opposite side of u0.
        f = dict(units={'u0':dict(position=[1,0],rays={'10':True,'20':False}),
                        'u1':dict(position=[11,0],rays={'10':False,'20':True})},
                 enemies={'10':[0,0],'20':[10,0]},reverse={'u0':{'10':True,'20':False},'u1':{'10':False,'20':True}})
        e = r._fronts(f)
        self.assertTrue(e['multiple_fronts'])
        self.assertIsNone(e['incoming_pincer'])
        f['enemies']['20'] = [2,0]
        f['reverse']['u0']['20'] = True
        e = r._fronts(f)
        self.assertIsNone(e['multiple_fronts'])
        self.assertEqual(e['incoming_pincer']['actor'],'u0')

    def test_continuing_shooter_between_restarts_is_not_new(self):
        fs = [frame(i) for i in range(3)]
        f = fs[0]
        f[0]['payload']['state']['relations']['u1']['e0'].update(smac_attack_available=True,in_weapon_range=True)
        f[0]['payload']['questions']['u1']['criteria']['6'] = 'attack'
        f[0]['payload']['state']['allies']['u1']['orders'] = [dict(ability_id=23,target_tag='11')]
        fs[1][0]['payload']['state']['allies']['u1']['weapon_cooldown_loops'] = 9
        audit = analyze(fs)
        w = audit['events'][0]['windows']['8']
        self.assertTrue(w['named_cd_by_h'])
        self.assertIs(w['new_named_cd_by_h'],False)


def make_batch(b):
    write = r.write_json
    (b/'manifests').mkdir()
    design = dict(labels=list(r.LABELS), policy_files={}, seed=100, shards={}, execution_mode=r.MODE)
    for label in r.LABELS:
        files = design['policy_files'][label] = {}
        for name in ('skills/tactical_v1/manager.py','sc2_skill_adapter/bridge/micro_trigger.py',
                     'sc2_skill_adapter/damage_maps/10gen_zerg.SC2Map'):
            p = b/label/name
            p.parent.mkdir(parents=True)
            data = (label if name.startswith('skills') else name).encode()
            p.write_bytes(data)
            files[name] = hashlib.sha256(data).hexdigest()
    env = dict(map_name='10gen_zerg',unit_balance='native',use_unit_ranges=True,min_attack_range=2,step_mul=8,
               capability_config=dict(n_units=10,n_enemies=10,team_gen=dict(unit_types=['zergling','hydralisk']),start_positions={}))
    timing = dict(mode=r.MODE,macro_loops=8,engine_step_loops=1)
    for group,(name,roster) in enumerate(r.COMPOSITIONS.items()):
        types = list(Counter(roster).elements())
        team = [u.lower() for u in types]
        seed = design['seed']+75*group
        for offset in range(0,75,5):
            stem = f'{name}_{offset:02d}'
            episodes = []
            for i in range(5):
                x = 16 if i==0 else i
                cfg = dict(team_gen=dict(ally_team=team,enemy_team=team),ally_start_positions=dict(item=[[x,16]]*10),
                           enemy_start_positions=dict(item=[[32-x,16]]*10))
                episodes.append(dict(episode_config=cfg,sha256=r.digest(cfg),sc2_seed=seed+offset+i,index=offset+i))
            m = dict(version=1,seed=seed,episodes=episodes,env_args=env)
            m['sha256'] = r.digest(m)
            write(b/'manifests'/(stem+'.json'),m)
            design['shards'][stem] = dict(composition=name,manifest_sha256=m['sha256'])
            for label in r.LABELS:
                w = b/label/stem
                files = design['policy_files'][label]
                for sub in ('damage_by_type','micro_triggers','score_details'):
                    (w/sub).mkdir(parents=True)
                meta = dict(manifest_sha256=m['sha256'],method='adaptive-skill',skill_version='tactical_v1',execution_mode=r.MODE)
                write(w/'manifest.json',m)
                write(w/'run.json',dict(meta,env_args=env,versions={},skill_parameters={},damage_by_type_enabled=True,
                        macro_loops=8,engine_step_loops=1,source_hashes=files,
                        skill_source_hashes={k:v for k,v in files.items() if k.startswith('skills/')}))
                write(w/'summary.json',dict(meta,status='complete',completed_episodes=5))
                events,records = [],[]
                for ep,e in enumerate(episodes,1):
                    filename = f'episode-{ep:04d}.json'
                    win = ep in {'baseline':{2},'packs':{1},'packs_reserve':{1,3}}[label]
                    g = dict(episode=ep,steps=1,scenario_sha256=e['sha256'],initial_state_sha256=r.digest(e),
                             outcome='win' if win else 'timeout', info=dict(battle_won=win,dead_allies=0,dead_enemies=0),
                             micro_trigger=dict(timing,windows=1,authorizations=0,states={}))
                    records.append(g)
                    events.append(dict(event='episode_started',episode=ep,game_loop=2,scenario_sha256=e['sha256'],
                                       initial_state_sha256=g['initial_state_sha256']))
                    events.extend(dict(f,event=k,episode=ep) for k,f in zip(('skill_observation','skill_step','step'),frame(0,label,types)))
                    events.append(dict(g,event='episode_completed'))
                    players = {}
                    for player,side in enumerate(r.SIDES,1):
                        damage = dict.fromkeys(r.TYPES,0)
                        damage['Hydralisk'] = player + r.LABELS.index(label)
                        players[str(player)] = dict(initial_roster_by_api_tag={str(10*(player-1)+j):u for j,u in enumerate(types,1)},
                            damage_dealt_by_source_type=damage,ticks_by_source_type={k:v*4096 for k,v in damage.items()})
                    write(w/'damage_by_type'/filename,dict(meta,episode=ep,scenario_sha256=e['sha256'],players=players,
                            ticks_per_damage=4096,begin_request_loop=2,terminal_loop=10,freeze_loop_range=[10,12]))
                    write(w/'micro_triggers'/filename,dict(timing,episode=ep,authorizations=[]))
                    write(w/'score_details'/filename,dict(meta,episode=ep,steps=1,scenario_sha256=e['sha256'],sc2_seed=e['sc2_seed'],
                            outcome=g['outcome'],initial={'game_loop':2},final={'game_loop':10}))
                write(w/'episodes.json',records)
                (w/'events.jsonl').write_text('\n'.join(map(json.dumps,events))+'\n')
    write(b/'design.json',design)


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.b = Path(tmp.name)
        make_batch(cls.b)

    def setUp(self):
        def replay(path):
            k = r.LABELS.index(path.parents[2].name)
            return [dict(freeze_loop_range=[10,12],ticks=[0,0,4096*(1+k),0,0,0,4096*(2+k),0])]
        for stub in (patch('contact_early_report.replay_resource_damage',replay),patch('contact_early_report._deaths',return_value={}),
                     patch.object(r,'_deaths',return_value={})):
            stub.start()
            self.addCleanup(stub.stop)

    def test_150_triples(self):
        report = r.summarize(self.b)
        self.assertEqual(r._read(self.b/'comparison.json'),report)
        self.assertEqual(len(report['episodes']),150)
        stats = report['total']
        self.assertEqual(stats['pairs'],150)
        self.assertEqual(set(report['by_baseline_contact_geometry']),{'no_positive_evidence'})
        self.assertEqual(report['by_layout']['center']['pairs'],30)
        self.assertEqual([v['pairs'] for v in report['by_composition'].values()],[75,75])
        a,b = (stats['paired_vs_baseline'][k] for k in r.LABELS[1:])
        self.assertEqual((a['gained_wins'],a['dropped_wins'],b['gained_wins'],b['dropped_wins']), (30,30,60,30))
        self.assertEqual(stats['reserve_vs_packs']['gained_wins'],30)
        self.assertEqual(report['reserve_selection_descriptive']['claimed']['pairs'],150)
        self.assertEqual(stats['versions']['packs_reserve']['damage_by_type']['enemy']['Hydralisk'],600)
        self.assertEqual(stats['versions']['packs']['metrics']['hydra_cd_fraction_48'],dict(n=0,missing=150,mean=None))

    def test_invalid_artifacts_do_not_replace_report(self):
        w = self.b/'packs_reserve'/(next(iter(r.COMPOSITIONS))+'_00')
        previous = self.b/'comparison.json'
        for path,key,value in [(w/'run.json','execution_mode','micro8-off'),(w/'manifest.json','seed',0),
                               (w/'run.json','versions',{'sc2':'different'}),(self.b/'design.json','shards',{})]:
            original = path.read_text()
            previous.write_text('old')
            try:
                r.write_json(path,dict(json.loads(original),**{key:value}))
                with self.subTest(key=key),self.assertRaises(ValueError):
                    r.summarize(self.b)
                self.assertEqual(previous.read_text(),'old')
            finally:
                path.write_text(original)
        p = self.b/'packs'/'skills/tactical_v1/unlisted.py'
        p.touch()
        self.addCleanup(p.unlink)
        with self.assertRaisesRegex(ValueError,'snapshot coverage'):
            r.summarize(self.b)

    def test_third_policy_initial_state_independent(self):
        w = self.b/'packs_reserve'/(next(iter(r.COMPOSITIONS))+'_00')
        old = r._read(w/'episodes.json')[0]['initial_state_sha256']
        for path in (w/'episodes.json',w/'events.jsonl'):
            original = path.read_text()
            self.addCleanup(path.write_text,original)
            path.write_text(original.replace(old,'f'*64))
        with self.assertRaisesRegex(ValueError,'initial state mismatch'):
            r.summarize(self.b)


if __name__ == '__main__':
    unittest.main()
