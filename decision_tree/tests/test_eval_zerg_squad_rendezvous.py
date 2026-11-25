"""Old-900 squad launcher contracts, without creating an SC2 environment."""
from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from decision_tree.scripts.linux import eval_zerg_squad_rendezvous as s


class Reports(unittest.TestCase):
    def test_frozen_online_not_v2_or_offline_selection(self):
        _, online, old = s.baseline()
        self.assertEqual(sum(r['online_outcome']=='win' for r in online),238)
        self.assertEqual(sum(r['archived_v2']=='win' for r in online),234)
        self.assertEqual(sum(r['online_outcome']!=r['offline_selected_outcome'] for r in online),1)
        self.assertEqual(Counter(g for r in old for g in s.groups(r)),dict(both_loss=609,both_severe=30,both_win=104))

    def test_actual_online_order_prefix(self):
        _, online, _ = s.baseline()
        ref = online[0]; work = s.ONLINE/ref['pair_id']/s.router.ARM/'runner'
        trace = []
        with (work/'events.jsonl').open() as stream:
            for line in stream:
                event = json.loads(line)
                if event['event'] == 'skill_observation': obs = event['payload']
                if event['event'] == 'step': trace.append(s.orders(obs,event))
        compare = lambda: s.first_difference(work,trace,ref)
        self.assertEqual(compare(),dict(status='identical',step=ref['steps']))
        idx = next(i for i,r in enumerate(trace) if r[2]); obs, act, _ = trace[idx]
        trace[idx] = obs, act, []
        self.assertEqual(compare()['status'],'micro')
        trace[0][1][0] += 1
        self.assertEqual(compare()['status'],'macro')
        trace[0] = 'other',trace[0][1],[]
        self.assertEqual(compare(),dict(status='unmatched_observation',step=0))


@unittest.skipUnless(sys.executable == s.ev.RSI_PYTHON,'historical runtime required')
class Launcher(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for mock in (patch.object(s.ev,'HISTORY',self.root/'history'),
                     patch.object(s,'play',side_effect=AssertionError('game launch forbidden'))):
            mock.start(); self.addCleanup(mock.stop)

    def test_full_dry_run(self):
        out = self.root/'full'
        s.main(['--out',str(out),'--mode','fixed','--dry-run'])
        d = s.ev.read_json(out/'design.json')
        self.assertEqual(d['pairs'],s.ev.read_json(s.ONLINE/'design.json')['pairs'])
        self.assertEqual((d['workers'],d['arms']),(60,[s.ARM]))
        self.assertFalse((out/'report.json').exists())
        with self.assertRaisesRegex(ValueError,'dry-run design'): s.main(['--resume',str(out)])

    def test_pilot_and_isolated_worker(self):
        from bridge import runner
        class NoGame(Exception): pass
        for mode in ('adaptive','fixed','guard','front'):
            out = self.root/mode
            s.main(['--out',str(out),'--mode',mode,'--pilot-ids','347','1','--dry-run'])
            s.main(['--resume',str(out),'--dry-run'])
            d = s.ev.read_json(out/'design.json'); p = d['pairs'][0]
            with patch.dict(os.environ,{'ZERG_SWARM_GATE':'poison','ZERG_SQUAD_STRATA':'both_loss'}): cmd, env = s.command(out,p,d)
            self.assertNotIn('ZERG_SWARM_GATE',env)
            self.assertNotIn('ZERG_SQUAD_STRATA',env)
            self.assertEqual(s.ev.read_json(out/p['manifest']),s.ev.read_json(s.ONLINE/p['manifest']))
            self.assertEqual(cmd[-1],str(out/p['pair_id']/s.ARM/'runner'))
            with patch.dict(os.environ,env,clear=True), patch.object(sys,'argv',cmd[3:]), \
                    patch('bridge.environment.ReplayableEnv',side_effect=NoGame):
                with self.assertRaises(NoGame): runner.main()
            meta = s.ev.read_json(Path(cmd[-1])/'run.json')
            expected = dict(kind='squad_rendezvous_v1',mode=mode,controller_sha256=d['sources'][s.SRC],
                            trial_id=d['policies'][s.ARM]['trial_id'])
            if mode in ('guard','front'):
                self.assertEqual(meta['source_hashes'][s.CROSS_SOURCE],env['ZERG_SQUAD_CROSS_SHA256'])
                expected.update(kind='squad_cross_support_v1', controller_sha256=d['sources'][s.CROSS_SOURCE],
                                parent_mode='adaptive', parent_controller_sha256=d['sources'][s.SRC])
            else:
                self.assertNotIn('ZERG_SQUAD_CROSS_SHA256',env)
            self.assertEqual(meta[s.ARM],expected)
            with self.assertRaisesRegex(ValueError,'frozen panel/mode'):
                s.main(['--resume',str(out),'--mode','fixed' if mode=='adaptive' else 'adaptive','--dry-run'])

    def test_incomplete_compare_cannot_launch_or_write(self):
        out = self.root/'pending'; s.prepare(out,'fixed',[2],1,False)
        with self.assertRaisesRegex(ValueError,'incomplete comparison'): s.main(['--comparison',str(out)])
        self.assertFalse((out/'report.json').exists())
        (out/'model.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'changed frozen'): s.main(['--resume',str(out),'--dry-run'])


if __name__ == '__main__': unittest.main()
