"""Independent worker wiring and paired evidence. All tests are offline."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from decision_tree.scripts.linux import eval_zerg_swarm as trial

ev = trial.ev


@unittest.skipUnless(sys.executable == ev.RSI_PYTHON, 'needs historical deps and manifests')
class SwarmEvaluation(unittest.TestCase):
    def test_resume_audits_one_arm_and_preserves_attempt(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(ev,'HISTORY',Path(tmp)/'history'):
            out = Path(tmp)/'trial'
            d = trial.prepare(out,2,'fresh',[1])
            p, = d['pairs']
            complete = out/p['pair_id']/'swarm/runner'
            complete.mkdir()
            summary = complete/'summary.json'
            ev.write_json(summary,dict(status='complete',completed_episodes=1))
            failed = out/p['pair_id']/'v2'
            (failed/'worker.log').write_text('partial attempt')
            before = {f:f.read_bytes() for f in out.rglob('*') if f.is_file()}
            with patch.object(trial,'audited_arm',return_value=({'initial':'state'},[])) as audit, \
                 patch.object(trial,'play') as play, patch.object(trial,'compare',return_value={}) as compare:
                trial.main(['--resume',str(out),'--resume-check'])
                self.assertEqual(audit.call_args.args[2],'swarm')
                play.assert_not_called()
                compare.assert_not_called()
                self.assertEqual(before,{f:f.read_bytes() for f in out.rglob('*') if f.is_file()})
                with patch.object(trial,'audited_arm',side_effect=ValueError('bad receipt')):
                    with self.assertRaisesRegex(ValueError,'bad receipt'):
                        trial.resume(out,1,False)
                self.assertEqual(before,{f:f.read_bytes() for f in out.rglob('*') if f.is_file()})
                trial.main(['--resume',str(out),'--workers','1'])
                play.assert_called_once()
                self.assertEqual(play.call_args.args[:4],(out,p,'v2',d['policies']['v2']))
                compare.assert_called_once_with(out)
                archived, = out.glob('resume-attempts-*/pair-001/v2/worker.log')
                self.assertEqual(archived.read_text(),'partial attempt')
                self.assertEqual(list(failed.iterdir()),[])
                for path,data in before.items():
                    if path != failed/'worker.log':
                        self.assertEqual(path.read_bytes(),data)

    def test_resume_preflight_refuses_drift_or_lock(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(ev,'HISTORY',Path(tmp)/'history'), \
             patch.object(trial,'play',side_effect=AssertionError('SC2 forbidden')):
            out = Path(tmp)/'trial'
            d = trial.prepare(out,2,'fresh',[1])
            with trial.run_lock(out), self.assertRaisesRegex(ValueError,'already locked'):
                trial.resume(out,1,True)
            with patch.object(ev,'frozen_sources',return_value=dict(d['sources'], **{trial.SRC:'changed'})), \
                 self.assertRaisesRegex(ValueError,'changed frozen source'):
                trial.resume(out,1,False)
            with self.assertRaisesRegex(ValueError,'workers'):
                trial.resume(out,0,True)
            manifest = out/d['pairs'][0]['manifest']
            m = ev.read_json(manifest)
            m['seed'] += 1
            manifest.write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                trial.resume(out,1,True)

    def test_missing_or_broken_summary_does_not_erase_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            runner = work/'runner'
            runner.mkdir()
            summary = runner/'summary.json'
            ev.write_json(summary,dict(status='failed',completed_episodes=0))
            self.assertFalse(trial.completed(work))
            (runner/'episodes.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'terminal evidence'):
                trial.completed(work)
            summary.unlink()
            with self.assertRaisesRegex(ValueError,'terminal evidence'):
                trial.completed(work)
            summary.write_text('{')
            with self.assertRaises(ValueError):
                trial.completed(work)

    def test_both_full_archive_aliases_and_clean_workers(self):
        from bridge import runner
        # Exercise real CLI preflight, stop BEFORE an environment is constructed.
        class OfflineBoundary(Exception):
            pass
        with tempfile.TemporaryDirectory() as tmp, patch.object(ev, 'HISTORY', Path(tmp)/'reservations'), \
             patch.dict(os.environ, {'ZERG_DISTILL_MODEL':'stale','ZERG_FORK_SPEC':'stale'}):
            for alias in ('regression','fresh'):
                out = Path(tmp)/alias
                d = trial.prepare(out,2,alias,[240],True)
                p, = d['pairs']
                archived = trial.old.archive(trial.old.ALIASES[alias])
                self.assertEqual(240,len(archived['pairs']))
                self.assertEqual(p,archived['pairs'][239])
                cv, ve = trial.command(out,p,'v2',d['policies']['v2'])
                self.assertEqual(ve['ZERG_DISTILL_OPENING_SHA256'],archived['policies']['v2']['opening_sha256'])
                self.assertIn('micro8-all',cv)
                cmd, env = trial.command(out,p,'swarm',d['policies']['swarm'])
                self.assertEqual({k for k in env if k.startswith('ZERG_')}, {'ZERG_SWARM_TRIAL_ID','ZERG_SWARM_SHA256'})
                self.assertEqual(env['SC2PATH'],ve['SC2PATH'])
                self.assertEqual(env['SMAC_DAMAGE_MAP_DIR'],ve['SMAC_DAMAGE_MAP_DIR'])
                with patch.dict(os.environ,env,clear=True), patch.object(sys,'argv',cmd[3:]), \
                     patch('bridge.environment.ReplayableEnv',side_effect=OfflineBoundary), \
                     patch('bridge.skill_runtime.load_manager',side_effect=AssertionError('native skill on swarm')):
                    with self.assertRaises(OfflineBoundary):
                        runner.main()
                meta = ev.read_json(out/p['pair_id']/'swarm/runner/run.json')
                self.assertEqual(meta['swarm'],d['policies']['swarm'])
                self.assertFalse(meta['rule_controller_executed'])
                self.assertNotIn('zerg_shadow',meta)
                ref = ev.reference()
                keys = set(ref['source_hashes'])-set(ref['skill_source_hashes']) | {trial.SRC} | {
                    k for k in d['sources'] if k.startswith('sc2_skill_adapter/bridge/')}
                self.assertEqual(meta['source_hashes'],{k:d['sources'][k] for k in keys})
                with self.assertRaisesRegex(ValueError,'invalid frozen panel'):
                    trial.compare(out)

    def test_matched_prefix_and_real_engine_acknowledgements(self):
        runner = ev.RUNS/trial.old.FRESH/'pair-001/v2/runner'
        with (runner/'events.jsonl').open() as stream:
            seen = []
            for line in stream:
                e = json.loads(line)
                if e['event'] == 'skill_observation':
                    obs = e['payload']
                if e['event'] == 'step':
                    seen.append((ev.digest(obs),e['actions']))
                    if len(seen) == 1:
                        masks = [[int(str(a) in obs['questions'][f'u{i}']['criteria']) for a in range(16)] for i in range(10)]
                        self.assertEqual(10,trial.check_macro(e['actions'],masks,e['feedback'],e['game_loop']))
                        e['feedback']['actors']['u0']['response']['code'] = 0
                        with self.assertRaisesRegex(ValueError,'missing/rejected'):
                            trial.check_macro(e['actions'],masks,e['feedback'],e['game_loop'])
        self.assertEqual('v2_micro_only',trial.prefix(runner,seen)['status'])
        obs, actions = seen[0]
        self.assertEqual('macro',trial.prefix(runner,[(obs,[0]*10)])['status'])
        self.assertEqual('unmatched_observation',trial.prefix(runner,[('bad',actions)])['status'])

    def test_new_panel_and_exact_repeat(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(ev,'HISTORY',Path(tmp)/'history'), \
             patch.object(trial,'play',side_effect=AssertionError('SC2 forbidden')):
            root = Path(tmp)/'fresh'
            trial.main(['--out',str(root),'--workers','1','--dry-run'])
            fresh = ev.read_json(root/'design.json')
            self.assertEqual(len({p['seed'] for p in fresh['pairs']}),240)
            copy = trial.prepare(Path(tmp)/'copy',1,ids=[240,1],dry=True,repeat=root)
            self.assertEqual(copy['pairs'],[fresh['pairs'][239],fresh['pairs'][0]])

    @unittest.skipUnless((ev.RUNS/'zerg_radical_smoke_r1/report.json').is_file(), 'external smoke archive not present')
    def test_external_real_smoke_and_tampered_decision(self):
        out = ev.RUNS/'zerg_radical_smoke_r1'
        report = trial.compare(out)
        row, = report['paired_outcomes']
        self.assertEqual(row['first_order_difference']['status'], 'macro')
        self.assertEqual(report['intervention']['controller_coverage'], 1.0)
        d = ev.read_json(out/'design.json')
        p, = d['pairs']
        source = out/p['pair_id']/'swarm/runner'
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            runner = work/'runner'
            runner.mkdir()
            for entry in source.iterdir():
                if entry.name != 'events.jsonl':
                    (runner/entry.name).symlink_to(entry)
            with (source/'events.jsonl').open() as src, (runner/'events.jsonl').open('x') as dst:
                for line in src:
                    if line.startswith('{"event": "swarm_decision",'):
                        e = json.loads(line)
                        e['decision']['actions'][0] = -1
                        line = json.dumps(e)+'\n'
                    dst.write(line)
            with self.assertRaisesRegex(ValueError,'step differs'):
                trial.swarm_arm(work,p,ev.read_json(out/p['manifest']),d,ev.reference())


if __name__ == '__main__':
    unittest.main()
