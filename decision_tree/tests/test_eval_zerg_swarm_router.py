"""Frozen old-panel routing and resume, without launching any game."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from decision_tree.scripts.linux import eval_zerg_swarm_router as r


@unittest.skipUnless(sys.executable == r.ev.RSI_PYTHON, 'historical runtime required')
class TestRouterLauncher(unittest.TestCase):
    def test_pilot_freeze_and_resume_no_game(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(r.ev, 'HISTORY', Path(tmp)/'history'), \
                patch.object(r, 'play', side_effect=AssertionError('SC2 forbidden')), \
                patch.object(r, 'offline_choice', side_effect=AssertionError('post-game only')), \
                patch.dict(os.environ, {'ZERG_FORCE_CHOICE':'swarm'}):
            out = Path(tmp)/'pilot'
            r.main(['--out',str(out),'--dry-run','--pilot-ids','36','347','810'])
            d = r.ev.read_json(out/'design.json')
            self.assertEqual((d['workers'],d['arms']),(60,[r.ARM]))
            for p in d['pairs']:
                self.assertEqual(r.ev.read_json(out/p['manifest']),r.ev.read_json(r.ARCHIVE/p['manifest']))
                cmd,env = r.command(out,p,d)
                self.assertEqual(env['ZERG_DISTILL_VERSION'],'v2')
                self.assertIn('micro8-all',cmd)
                self.assertEqual(cmd[-1],str(out/p['pair_id']/r.ARM/'runner'))
                self.assertNotIn('ZERG_FORCE_CHOICE',env)
                self.assertEqual(r.ev.sha(Path(env['ZERG_SWARM_GATE'])),env['ZERG_SWARM_GATE_SHA256'])
                self.assertFalse((out/p['pair_id']/'v2').exists())
            r.main(['--resume',str(out),'--dry-run'])
            with self.assertRaisesRegex(ValueError,'dry-run design'):
                r.main(['--resume',str(out)])
            (out/'gate.json').write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError,'changed frozen'):
                r.main(['--resume',str(out),'--dry-run'])

    def test_archived_anchors_agree_with_sealed_predictions(self):
        predictions = r.ev.read_json(r.GATE.parent/'holdout_predictions.json')
        rows = {row['scenario_sha256']:row for row in r.ev.read_json(r.ARCHIVE/'report.json')['paired_outcomes']}
        gate = r.ev.read_json(r.GATE)
        for branch in ('v2','swarm'):
            pred = next(p for p in predictions if p['choice'] == branch)
            row = rows[pred['scenario']]
            self.assertEqual(r.offline_choice(row,row,gate),branch)

    def test_full_dry_run_freezes_only_900_real_manifests(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(r.ev,'HISTORY',Path(tmp)/'history'), \
                patch.object(r,'execute',side_effect=AssertionError('launch forbidden')):
            out = Path(tmp)/'full'
            r.main(['--out',str(out),'--dry-run'])
            pairs = r.ev.read_json(out/'design.json')['pairs']
            self.assertEqual(pairs,r.ev.read_json(r.ARCHIVE/'design.json')['pairs'])
            self.assertEqual(len(list(out.glob('pair-*/online_router'))),900)

    def test_live_worker_contract_before_environment_creation(self):
        from bridge import runner
        class StopBeforeGame(Exception): pass
        with tempfile.TemporaryDirectory() as tmp, patch.object(r.ev,'HISTORY',Path(tmp)/'history'):
            out = Path(tmp)/'pilot'; d = r.prepare(out,[36],1,True)
            cmd,env = r.command(out,d['pairs'][0],d)
            with patch.dict(os.environ,env,clear=True),patch.object(sys,'argv',cmd[3:]), \
                    patch('bridge.environment.ReplayableEnv',side_effect=StopBeforeGame):
                with self.assertRaises(StopBeforeGame): runner.main()
            meta = r.ev.read_json(Path(cmd[-1])/'run.json')
            self.assertEqual(meta['swarm_route']['gate_sha256'],r.GATE_HASH)
            self.assertEqual(meta['swarm_route']['trial_id'],d['policies'][r.ARM]['trial_id'])
