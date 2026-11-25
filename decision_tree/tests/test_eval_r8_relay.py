"""Archive, retry and execution contracts. No StarCraft process is started."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from decision_tree.scripts.linux import eval_r8_relay as r


class RelayTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_archive_report_cohorts_and_no_stage_double_counting(self):
        d, rows = r.archive()
        results = {(p['pair_id'], a): dict(p['R6'], changed_steps=0) for p in rows for a in r.ARMS}
        result = r.report(r.ARCHIVE, d, rows, results)
        self.assertEqual(result['pairs'], 1000)
        self.assertEqual(result['new_games'], 2000)
        self.assertEqual(result['arms']['R6']['wins'], 283)
        self.assertEqual({k:t['pairs'] for k,t in result['strata'].items()}, r.COHORTS)
        self.assertEqual(result['comparisons']['menu_to_full']['net'], 0)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            r.report(r.ARCHIVE,dict(d,pairs=d['pairs']+[d['pairs'][0]]),rows+[rows[0]],results)
        results['pair-001','menu']['initial'] = 'different'
        with self.assertRaisesRegex(ValueError,'initial states'):
            r.report(r.ARCHIVE,d,rows,results)

    def test_archive_smoke_never_launches(self):
        with patch.object(r,'play',side_effect=AssertionError('SC2 forbidden')):
            r.main(['--dry-run','--pilot-ids','1000,1'])
            for ids in ('1,1','0','1001'):
                with self.assertRaisesRegex(ValueError,'pilot'):
                    r.main(['--dry-run','--pilot-ids',ids])

    def test_retries_across_resume_preserve_every_attempt(self):
        p = dict(pair_id='pair-001'); work = self.root/'pair-001/menu'; work.mkdir(parents=True)
        transient = 'at env.reset_record(scenario)\npysc2.lib.remote_controller.ConnectError: connection closed\n'
        def crash(*args, stdout, **kwargs):
            stdout.write(transient)
            return type('Result',(),dict(returncode=1))()
        (work/'worker.log').write_text(transient)
        with patch.object(r,'command',return_value=([],{})), patch.object(r.time,'sleep'), patch.object(r.subprocess,'run',side_effect=crash) as run:
            for _ in range(2):
                with self.assertRaisesRegex(ValueError,'exhausted'):
                    r.play(self.root,p,{},'menu',0)
            self.assertEqual(run.call_count,3)
            root, count = r.attempts(work)
            self.assertEqual(count,3)
            for i in range(1,4): self.assertEqual((root/str(i)/'worker.log').read_text(),transient)
            (root/'1/runner').mkdir()
            (root/'1/runner/events.jsonl').write_text('{"event":"episode_started"}\n')
            with self.assertRaisesRegex(ValueError,'ambiguous prior'):
                r.attempts(work)

    def test_ambiguous_failure_is_not_retryable(self):
        work = self.root/'arm'; (work/'runner').mkdir(parents=True)
        log = work/'worker.log'; log.write_text('ordinary failure')
        self.assertFalse(r.retryable(work))
        log.write_text('env.reset_record(scenario) Address already in use')
        (work/'runner/events.jsonl').write_text('{"event":"run_failed","completed_episodes":0}\n')
        self.assertTrue(r.retryable(work))
        (work/'runner/replays').mkdir(); (work/'runner/replays/partial.SC2Replay').touch()
        self.assertFalse(r.retryable(work))

    def test_recompute_archived_first_guard_and_new_relay(self):
        from decision_tree.zerg.combat_relay_r8 import CombatRelayR8
        path = r.ARCHIVE/'pair-001/R6/runner/events.jsonl'
        with path.open() as f:
            obs = next(json.loads(line)['payload'] for line in f if line.startswith('{"event": "skill_observation"'))
            g = json.loads(next(f))
        for mode in r.ARMS:
            def check(dec):
                return r.check_decision(r.sq.controller_class('guard')(mode='guard'),CombatRelayR8(mode),obs,g,dec,False,mode)
            base = g['actions']; menu = dict(base=base)
            dec = dict(CombatRelayR8(mode).propose(obs,g['avail_actions'],base,menu),mode=mode,
                avail_actions=g['avail_actions'],r6_actions=base,candidates=menu,r6_candidate_id='base',
                opening_handoff=True,owner='r6',changed_actors=[])
            self.assertEqual(check(dec),'r6')
            with self.assertRaisesRegex(ValueError,'replay differs'): check(dict(dec,audit={}))
            with self.assertRaisesRegex(ValueError,'wrong relay inputs'): check(dict(dec,candidates={}))
            with self.assertRaisesRegex(ValueError,'wrong relay owner'): check(dict(dec,owner='menu'))

    @unittest.skipUnless(sys.executable == r.ev.RSI_PYTHON,'historical interpreter required')
    def test_complete_no_intervention_audit_and_corrupt_receipt(self):
        # Reuse a full real R6 trajectory where menu never diverges. Only in this temporary
        # fixture, add relay telemetry; preserve the original engine actions, scores and replay.
        from decision_tree.zerg.combat_relay_r8 import CombatRelayR8
        parent, rows = r.archive(); p = parent['pairs'][5]; ref = rows[5]
        original, _, _, src = r.runtime(parent)
        policy = parent['policies']['R6']; d = dict(sources=src,policies=dict(menu=policy))
        pair = self.root/p['pair_id']; origin = r.ARCHIVE/p['pair_id']
        shutil.copytree(origin/'R6',pair/'menu'); shutil.copyfile(origin/'manifest.json',pair/'manifest.json')
        work = pair/'menu/runner'; read = r.ev.read_json
        def save(name, obj): (work/name).write_text(json.dumps(obj))
        meta = read(work/'run.json'); meta['source_hashes'].update({k:src[k] for k in (*r.BRIDGE_HASHES,r.RELAY,'sc2_skill_adapter/bridge/zerg_distill.py')})
        meta['combat_relay'] = dict(kind='combat_relay_r8',mode='menu',controller_sha256=r.RELAY_SHA,parent_mode='guard',
                                    parent_controller_sha256=src[r.sq.CROSS_SOURCE],trial_id=policy['trial_id'])
        save('run.json',meta)
        ep, = read(work/'episodes.json'); counts = dict(opening_steps=ep['squad_rendezvous']['squad_steps'],
            r6_steps=ep['squad_rendezvous']['v2_steps'],menu_steps=0,custom_steps=0,handoffs_to_r6=0)
        ep['combat_relay'] = counts; save('episodes.json',[ep]); receipt = read(work/'distill/episode-0001.json')
        receipt.update(scope='executed_native_relay_steps_only',combat_relay=counts)
        by_step = {x['step']:x for x in receipt['decisions']}
        shadow = {x['game_loop']:x for x in map(json.loads,(pair/'menu/shadow/shadow.jsonl').read_text().splitlines())}
        path = work/'events.jsonl'; events=[]; relay=CombatRelayR8('menu'); handoff=False
        for e in map(json.loads,path.read_text().splitlines()):
            events.append(e); k = e['event']
            if k == 'skill_observation': obs = e['payload']
            if k == 'squad_rendezvous_decision':
                i=e['step']; loop=e['game_loop']; actions=e['actions']; edge=e['handoff'] and not handoff; handoff=e['handoff']
                stamp=dict(step=i,game_loop=loop,episode=1); menu={}; cid=None; plans=[]; owner='opening'
                if handoff:
                    row=by_step[i]; cid=row['selected_candidate']; owner='r6'; plans=row['plan_ids']
                    row.update(execution_owner=owner,r6_candidate_id=cid)
                    menu={x['candidate_id']:[x['proposed_actions'][f'u{j}'] for j in range(10)] for x in shadow[loop]['candidates'] if x['candidate_id'] in row['scores']}
                dec=relay.propose(obs,e['avail_actions'],actions,menu)
                self.assertEqual(dec['actions'],actions)
                events.append(dict(event='combat_relay_decision',**stamp,**dec,owner=owner,opening_handoff=edge,
                    avail_actions=e['avail_actions'],r6_candidate_id=cid,r6_actions=actions,changed_actors=[],candidates=menu,mode='menu'))
            if k == 'micro_window_closed': grants=e['records']
            if k == 'step':
                events.append(dict(event='combat_relay_receipt',**stamp,owner=owner,actions=actions,plan_ids=plans,
                    micro_grants=grants,micro_submissions=e['feedback']['micro_submissions'],handoff_to_r6=False,dropped_r6_micro_grants=[]))
            if k == 'episode_completed': e.update(combat_relay=counts)
        save('distill/episode-0001.json',receipt)
        path.write_text(''.join(json.dumps(e)+'\n' for e in events))
        def check(): return r.audit(self.root,p,d,ref,'menu',original)
        self.assertEqual(check()['changed_steps'],0)
        next(e for e in events if e['event']=='combat_relay_receipt')['handoff_to_r6']=True
        path.write_text(''.join(json.dumps(e)+'\n' for e in events))
        with self.assertRaisesRegex(ValueError,'handback'): check()

    @unittest.skipUnless(sys.executable == r.ev.RSI_PYTHON,'historical interpreter required')
    def test_frozen_pilot_and_real_worker_contract_before_reset(self):
        from bridge import runner
        class NoGame(Exception): pass
        out = self.root/'pilot'; history = self.root/'history'; history.mkdir()
        parent,_ = r.archive(); name = r.ev.digest(parent)+'.json'
        shutil.copyfile(r.ev.HISTORY/name,history/name)
        with patch.object(r.ev,'HISTORY',history), patch.object(r,'play',side_effect=AssertionError('SC2 forbidden')):
            r.main(['--out',str(out),'--dry-run','--pilot-ids','1000','2'])
            r.main(['--resume',str(out),'--dry-run'])
            with self.assertRaisesRegex(ValueError,'dry-run reservation'): r.main(['--resume',str(out)])
            d = r.ev.read_json(out/'design.json'); p = d['pairs'][0]
            self.assertEqual(d['workers'],16); self.assertEqual(d['pairs'],[parent['pairs'][999],parent['pairs'][1]])
            self.assertFalse((out/'report.json').exists())
            for mode in r.ARMS:
                cmd, env = r.command(out,p,d,mode)
                with patch.dict(os.environ,env,clear=True),patch.object(sys,'argv',cmd[3:]),patch('bridge.environment.ReplayableEnv',side_effect=NoGame):
                    with self.assertRaises(NoGame): runner.main()
                meta = r.ev.read_json(Path(cmd[-1])/'run.json')
                self.assertEqual(meta['combat_relay']['mode'],mode)
                self.assertEqual(meta['combat_relay']['controller_sha256'], r.RELAY_SHA)
                self.assertEqual(meta['squad_rendezvous']['controller_sha256'], parent['sources'][r.sq.CROSS_SOURCE])


if __name__ == '__main__': unittest.main()
