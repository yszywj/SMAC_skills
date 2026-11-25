"""Offline launcher/auditor checks. Never start a runner process."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from decision_tree.scripts.linux import eval_zerg_policy as ev


class PolicyEvalTest(unittest.TestCase):
    def test_arm_isolation(self):
        out = Path('/frozen')
        pair = dict(pair_id='pair-001', roster='zerg', manifest='pair-001/manifest.json')
        with patch.dict('os.environ', {'ZERG_DISTILL_MODEL': '/wrong', 'ZERG_WHOLE_AC_ARM': 'C',
                                       'SMAC_TEST': 'on', 'PYTHONPATH': 'bad'}):
            a, ae = ev.command(out, pair, 'baseline')
            b, be = ev.command(out, pair, 'tree')
        self.assertEqual(a[:-1], b[:-1])
        self.assertEqual(a[0], ev.RSI_PYTHON)
        self.assertNotIn('ZERG_DISTILL_MODEL', ae)
        self.assertEqual(be['ZERG_DISTILL_MODEL'], str(out/'model.json'))
        for env in [ae, be]:
            for key in ('ZERG_WHOLE_AC_ARM', 'SMAC_TEST', 'PYTHONPATH'):
                self.assertNotIn(key, env)

    @unittest.skipUnless(sys.executable == ev.RSI_PYTHON, 'manifest generation needs RSI Python')
    def test_full_dry_run_and_collision(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(ev, 'play', side_effect=AssertionError) as play:
            root = Path(tmp)
            with patch.object(ev, 'HISTORY', root/'reservations'):
                ev.main(['--out', str(root/'run'), '--seed', '100000000', '--dry-run'])
                design = ev.read_json(root/'run/design.json')
                self.assertEqual(design['workers'], 60)
                self.assertEqual(set(ev.Counter(p['roster'] for p in design['pairs']).values()), {20})
                self.assertEqual(len({p['manifest_sha256'] for p in design['pairs']}), 240)
                self.assertEqual(len(list((root/'run').glob('pair-*/manifest.json'))), 240)
                with self.assertRaisesRegex(ValueError, 'previously used seed'):
                    ev.prepare(root/'dup', ev.MODEL, 60, 100000000, True)
                with self.assertRaisesRegex(ValueError, 'not a complete'):
                    ev.compare(root/'run')
                play.assert_not_called()

    def test_archived_baseline(self):
        original = ev.reference()
        worker = ev.ORIGINAL.parents[1]
        manifest = ev.read_json(worker/'manifest.json')
        pair = dict(pair_id='pair-001', seed=manifest['seed'], roster=original['zerg_shadow']['scene_config_id'],
                    scenario_sha256=manifest['episodes'][0]['sha256'])
        design = dict(model_contract=ev.model_contract(ev.MODEL, original), sources=original['source_hashes'])
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)/pair['pair_id']
            work.mkdir()
            (work/'baseline').symlink_to(worker)
            arm = ev.audit_arm(Path(tmp), pair, 'baseline', manifest, original, design)
            self.assertEqual((arm['outcome'], arm['steps']), ('loss', 35))
            design['model_contract']['source_hash'] = 'f'*64
            with self.assertRaisesRegex(ValueError, 'wrong parent provenance'):
                ev.audit_arm(Path(tmp), pair, 'baseline', manifest, original, design)

    def test_model_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'other.json'
            path.write_bytes(ev.MODEL.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'model SHA256'):
                ev.model_contract(path, ev.reference())

    def test_decision_and_engine_coverage(self):
        events = [dict(actions=[3], game_loop=2), dict(actions=[4], game_loop=10)]
        rows = [dict(step=i, game_loop=e['game_loop'], final_actions={'u0': e['actions'][0]}, parent_actions={'u0': 3},
                     committed=True, mode='model_switch' if i else 'fallback', macro_receipt='unknown' if i else 'accepted',
                     micro_receipt='not_submitted', fallback_reason=None if i else 'singleton', filtered=[])
                for i, e in enumerate(events)]
        stats = dict(steps=2, model_base=0, model_switch=1, fallback=1, filtered_candidates=0, engine_unconfirmed=1)
        receipt = dict(stats=stats, decisions=rows)
        self.assertEqual(ev.audit_decisions(receipt, events, stats), (stats, {'singleton': 1}))
        for field, invalid in [('step', 0), ('game_loop', 2), ('final_actions', {'u0': 3}), ('committed', False),
                               ('macro_receipt', 'rejected'), ('mode', 'fallback')]:
            broken = copy.deepcopy(receipt)
            broken['decisions'][1][field] = invalid
            with self.subTest(field=field), self.assertRaises(ValueError):
                ev.audit_decisions(broken, events, stats)
        with self.assertRaises(ValueError):
            ev.audit_decisions(receipt, events[:1], stats)

    def test_paired_estimands(self):
        def row(a, b, delta):
            return dict(baseline=dict(outcome=a, reward=2), tree=dict(outcome=b, reward=2+delta,
                        decisions={'fallback': 1}, fallback_reasons={'singleton': 1}))
        result = ev.aggregate([row('win', 'loss', -2), row('draw', 'win', 1), row('loss', 'win', 4)])
        self.assertEqual(result['win_flips'], dict(gained=2, lost=1, net=1))
        self.assertEqual((result['baseline_win_rate'], result['tree_win_rate']), (1/3, 2/3))
        self.assertEqual(result['mean_return_delta'], 1)
        self.assertEqual(result['fallback_reasons'], {'singleton': 3})


if __name__ == '__main__':
    unittest.main()
