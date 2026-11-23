"""Offline replay-decoder doubles; exercise real shard/event/roster and report validation."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import contact_all_micro_report as r
from contact_early_report import COMPOSITIONS
from test_contact_local_packs_report import frame


def write(path, obj):
    path.write_text(json.dumps(obj))


def sample(label, ep, types, i):
    trace_label = 'packs_reserve' if label in r.RESERVE else 'packs'
    obs, decision, step = frame(i, trace_label, types)
    if i:
        obs['payload']['state']['allies']['u1']['weapon_cooldown_loops'] = 8
    rows = []
    if i == 0:
        slots = [types.index('Hydralisk')] + ([] if label == 'packs_hydra' else [1])
        for slot in slots:
            actor = f'u{slot}'
            response = [1] if slot != 1 or ep == 1 else [] if ep == 2 else [5]
            rows.append(dict(actor=actor, actor_tag=str(slot+1), target='e0', target_tag='11',
                             actor_type=types[slot].lower(), submitted_action_id=6, kind='approach_then_attack',
                             episode=ep, macro_step=0, issued_loop=2, expires_loop=9, macro_action_id=4,
                             status='submitted', checked_loops=3, first_legal_loop=5, submitted_loop=5, sc2_response=response))
    loop = 2 + i*8
    step['feedback']['micro_submissions'] = list(rows)
    window = dict(macro_step=i, issued_loop=loop, closed_loop=loop+8, records=rows, interior_checks=7,
                  authorizations=len(rows), states=dict(Counter(a['status'] for a in rows)))
    return [dict(e, event=k, episode=ep) for e, k in zip((obs, decision, window, step),
            ('skill_observation', 'skill_step', 'micro_window_closed', 'step'))], rows


def shard(work, label, manifest, files, types):
    mode = r.MODE_MAP[label]
    for sub in ('micro_triggers', 'score_details', 'damage_by_type'):
        (work / sub).mkdir(parents=True)
    timing = dict(mode=mode, macro_loops=8, engine_step_loops=1)
    meta = dict(method='adaptive-skill', skill_version='tactical_v1', execution_mode=mode, manifest_sha256=manifest['sha256'])
    write(work/'manifest.json', manifest)
    write(work/'run.json', dict(meta, macro_loops=8, engine_step_loops=1, env_args=manifest['env_args'],
          versions={}, skill_parameters={}, damage_by_type_enabled=True, source_hashes=files,
          skill_source_hashes={k:v for k,v in files.items() if k.startswith('skills/')}))
    records, events = [], []
    k = r.LABELS.index(label)
    for ep, e in enumerate(manifest['episodes'], 1):
        fn = f'episode-{ep:04d}.json'
        ids = dict(episode=ep, scenario_sha256=e['sha256'], initial_state_sha256=r.digest(e))
        win = ep == k+1
        events.append(dict(ids, event='episode_started', game_loop=2))
        frames, rows = sample(label, ep, types, 0)
        events.extend(frames + sample(label, ep, types, 1)[0])
        game = dict(ids, outcome='win' if win else 'draw', steps=2, info=dict(battle_won=win, dead_allies=0, dead_enemies=0),
                    micro_trigger=dict(timing, windows=2, authorizations=len(rows), states={'submitted': len(rows)}))
        records.append(game)
        events.append(dict(game, event='episode_completed'))
        write(work/'micro_triggers'/fn, dict(timing, episode=ep, authorizations=rows))
        write(work/'score_details'/fn, dict(meta, **game, sc2_seed=e['sc2_seed'], initial={'game_loop':2}, final={'game_loop':18}))
        players = {}
        for p in (1, 2):
            damage = dict.fromkeys(r.packs.TYPES, 0)
            damage['Hydralisk'] = p+k
            players[str(p)] = dict(initial_roster_by_api_tag={str(10*(p-1)+i):u for i,u in enumerate(types,1)},
                                  ticks_by_source_type={u:v*4096 for u,v in damage.items()}, damage_dealt_by_source_type=damage)
        write(work/'damage_by_type'/fn, dict(ids, manifest_sha256=manifest['sha256'], players=players,
              ticks_per_damage=4096, begin_request_loop=2, terminal_loop=18, freeze_loop_range=[18,20]))
    (work/'events.jsonl').write_text('\n'.join(map(json.dumps,events))+'\n')
    write(work/'episodes.json', records)
    n = sum(g['micro_trigger']['authorizations'] for g in records)
    write(work/'summary.json', dict(meta, status='complete', completed_episodes=5,
          micro_trigger_totals=dict(timing, windows=10, authorizations=n, states={'submitted':n})))


def batch(root):
    (root/'manifests').mkdir()
    design = dict(labels=r.LABELS, mode_map=r.MODE_MAP, policy_files={}, seed=10, shards={})
    for label in r.LABELS:
        files = design['policy_files'][label] = {}
        for filename in ('skills/tactical_v1/test_policy.py', 'sc2_skill_adapter/bridge/micro_trigger.py',
                         'sc2_skill_adapter/damage_maps/10gen_zerg.SC2Map'):
            path = root/label/filename
            path.parent.mkdir(parents=True)
            data = (label.split('_')[0] if filename.startswith('skills/') else filename).encode()
            path.write_bytes(data)
            files[filename] = hashlib.sha256(data).hexdigest()
    env = dict(map_name='10gen_zerg', unit_balance='native', use_unit_ranges=True, min_attack_range=2, step_mul=8,
               capability_config=dict(n_units=10,n_enemies=10,team_gen=dict(unit_types=['zergling','hydralisk']),start_positions={}))
    for group, (name, roster) in enumerate(COMPOSITIONS.items()):
        types = list(Counter(roster).elements())
        team = [u.lower() for u in types]
        seed = design['seed'] + 75*group
        for offset in range(0,75,5):
            stem = f'{name}_{offset:02d}'
            episodes = []
            for i in range(offset,offset+5):
                x = 16 if i%5 == 0 else 1
                cfg = dict(team_gen=dict(ally_team=team,enemy_team=team),ally_start_positions={'item':[[x,16]]*10},
                           enemy_start_positions={'item':[[32-x,16]]*10})
                episodes.append(dict(index=i,sc2_seed=seed+i,episode_config=cfg,sha256=r.digest(cfg)))
            m = dict(version=1,env_args=env,seed=seed,episodes=episodes)
            m['sha256'] = r.digest(m)
            write(root/'manifests'/(stem+'.json'),m)
            design['shards'][stem] = dict(manifest_sha256=m['sha256'],composition=name)
            for label in r.LABELS:
                shard(root/label/stem,label,m,design['policy_files'][label],types)
    write(root/'design.json',design)


class FiveWayReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.root = Path(tmp.name)
        batch(cls.root)

    def setUp(self):
        def replay(path):
            k = r.LABELS.index(path.parents[2].name)
            return [dict(freeze_loop_range=[18,20], ticks=[0,0,(1+k)*4096,0,0,0,(2+k)*4096,0])]
        for stub in (patch('contact_early_report.replay_resource_damage',replay),
                     patch('contact_early_report._deaths',return_value={}), patch.object(r.packs,'_deaths',return_value={})):
            stub.start()
            self.addCleanup(stub.stop)

    def test_five_way_roundtrip_and_incremental_evidence(self):
        report = r.summarize(self.root)
        self.assertEqual(report, r._read(self.root/'comparison.json'))
        self.assertEqual(len(report['episodes']),150)
        self.assertEqual(report['mode_map'], r.MODE_MAP)
        self.assertEqual([g['pairs'] for g in report['by_composition'].values()], [75,75])
        self.assertEqual(report['by_layout']['center']['pairs'],30)
        group = report['by_composition'][next(iter(COMPOSITIONS))]['by_layout']['scattered']
        self.assertEqual(group['pairs'],60)
        total = report['total']
        self.assertEqual(total, r._summary(report['episodes']))
        for label in r.LABELS:
            v = total['versions'][label]
            dog = v['micro_trigger']['by_type']['Zergling']
            self.assertEqual([dog[k] for k in r.COUNTS], [0]*5 if label=='packs_hydra' else [150,150,30,90,30])
            self.assertEqual(v['micro_trigger']['by_type']['Hydralisk']['accepted'],150)
            self.assertEqual(v['damage_by_type']['ally']['Zergling'],0)  # Accepted attacks need not hit.
            c = v['local_coord']['counts']
            self.assertEqual(c.get('reserve_new_named_cd_by_h_8_true',0),150 if label in r.RESERVE else 0)
        self.assertEqual(len(total['incremental']),5)
        for left,right in r.CONTRASTS:
            p = total['incremental'][right+'_minus_'+left]
            self.assertEqual((p['left'],p['right'],p['pairs'],p['gained_wins'],p['dropped_wins']), (left,right,150,30,30))
            self.assertEqual(p['paired_outcomes']['draw']['win'],30)
            self.assertEqual(p['damage_delta']['enemy']['Hydralisk'],150*(r.LABELS.index(right)-r.LABELS.index(left)))
        self.assertEqual(total['incremental']['packs_all_minus_packs_hydra']['micro_delta_by_type']['Zergling']['accepted'],30)

    def test_invalid_shard_does_not_replace_report(self):
        d = self.root/'design.json'
        w = self.root/'marginal_near'/(next(iter(COMPOSITIONS))+'_00')
        previous = self.root/'comparison.json'
        previous.write_text('keep')
        def corrupt(path, change, lines=False):
            original = path.read_text()
            data = list(map(json.loads, original.splitlines())) if lines else json.loads(original)
            change(data)
            try:
                if lines:
                    path.write_text('\n'.join(map(json.dumps, data))+'\n')
                else:
                    write(path, data)
                with self.assertRaises(ValueError):
                    r.summarize(self.root)
                self.assertEqual(previous.read_text(),'keep')
            finally:
                path.write_text(original)
        for path, change in ((d,lambda z:z['mode_map'].update(packs_hydra='micro8-all')),
                    (w/'run.json',lambda z:z.update(execution_mode='micro8-hydra')),
                    (w/'summary.json',lambda z:z['micro_trigger_totals'].update(windows=5)),
                    (w/'manifest.json',lambda z:z.update(seed=0)),
                    (w/'micro_triggers/episode-0001.json',lambda z:z['authorizations'][1].update(actor_tag='1')),
                    (w/'damage_by_type/episode-0001.json',lambda z:z['players']['2']['ticks_by_source_type'].update(Hydralisk=0))):
            with self.subTest(path=path):
                corrupt(path, change)
        for event, change in (('micro_window_closed',lambda z:z.update(interior_checks=6)),
                              ('micro_window_closed',lambda z:z.update(event='lost_window')),
                              ('step',lambda z:z['feedback'].update(micro_submissions=[])),
                              ('skill_step',lambda z:z['local_coord'].update(dispatch=False))):
            with self.subTest(event=event):
                corrupt(w/'events.jsonl',lambda rows:change(next(z for z in rows if z['event']==event)),True)

    def test_fifth_policy_initial_mismatch(self):
        w = self.root/'marginal_near'/(next(iter(COMPOSITIONS))+'_00')
        state = r._read(w/'episodes.json')[0]['initial_state_sha256']
        for path in (w/'episodes.json', w/'events.jsonl'):
            original = path.read_text()
            self.addCleanup(path.write_text, original)
            path.write_text(original.replace(state, '0'*64))
        with self.assertRaisesRegex(ValueError,'initial state mismatch'):
            r.summarize(self.root)

    def test_trace_identity_mode_and_time(self):
        types = ['Zergling']*3 + ['Hydralisk']*7
        rosters = {'1': {str(i+1):t for i,t in enumerate(types)}, '2': {str(i+11):t for i,t in enumerate(types)}}
        for mode, change in [('micro8-hydra',{}), ('micro8-all',{'actor':'u0'}),
                ('micro8-all',{'target_tag':'12'}), ('micro8-all',{'actor_type':'hydralisk'}),
                ('micro8-all',{'expires_loop':10}), ('micro8-all',{'submitted_loop':10}),
                ('micro8-all',{'sc2_response':[True]}), ('micro8-all',{'macro_action_id':1})]:
            events,rows = sample('packs_all',1,types,0)
            obs,decision,window,step = events
            step['micro_window_closed'] = window
            rows[1].update(change)
            with self.subTest(mode=mode,change=change), self.assertRaises(ValueError):
                r._micro_window(obs,decision,step,1,rosters,mode,{u:Counter() for u in types},'test')

    def test_no_resend_is_authorized_but_not_submitted(self):
        types = ['Zergling']*3 + ['Hydralisk']*7
        rosters = {'1': {str(i+1):t for i,t in enumerate(types)}, '2': {'11':'Zergling'}}
        events, rows = sample('packs_all',1,types,0)
        obs,decision,window,step = events
        rows[0].update(status='never_legal', first_legal_loop=None, checked_loops=7)
        rows[1].update(status='already_ordered')
        for row in rows:
            row.update(submitted_action_id=None, submitted_loop=None)
            del row['sc2_response']
        window['states'] = {'already_ordered':1, 'never_legal':1}
        step.update(micro_window_closed=window)
        step['feedback']['micro_submissions'] = []
        by_type = {u:Counter() for u in types}
        r._micro_window(obs, decision, step, 1, rosters, 'micro8-all', by_type, 'test')
        self.assertEqual(by_type, {t:{'authorizations':1} for t in types})


if __name__ == '__main__':
    unittest.main()
