"""Offline test of panel construction, actual historical receipts and fail-closed comparison."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from decision_tree.scripts.linux import eval_zerg_two_trees as two

ev = two.ev


class TwoTrees(unittest.TestCase):
    def test_metrics_exclude_draws_from_literal_flips(self):
        def arm(outcome, reward, kills):
            return dict(outcome=outcome, reward=reward, enemy_elimination_fraction=kills / 10)
        rows = [dict(v1=arm('win', 20, 10), v2=arm('loss', 8, 5)),
                dict(v1=arm('draw', 12, 5), v2=arm('win', 20, 10)),
                dict(v1=arm('loss', 10, 6), v2=arm('win', 20, 10))]
        result = two.aggregate(rows)
        self.assertEqual((result['win_to_loss'], result['loss_to_win']), (1, 1))
        self.assertEqual(result['arms']['v2']['severe_losses'], 1)
        self.assertEqual(result['mean_return_delta'], 2)

    @unittest.skipUnless(sys.executable == ev.RSI_PYTHON, 'historical Python required')
    def test_full_and_replay_designs(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(ev, 'play', side_effect=AssertionError):
            root = Path(tmp)
            with patch.object(ev, 'HISTORY', root/'reservations'):
                two.main(['--out', str(root/'fresh'), '--dry-run', '--seed', '1356100100'])
                d = ev.read_json(root/'fresh/design.json')
                self.assertEqual(d['arms'], ['v1', 'v2'])
                self.assertEqual(len(list((root/'fresh').glob('pair-*/v[12]'))), 480)
                for a in two.ARMS:
                    _, env = ev.command(root/'fresh', d['pairs'][0], a, d['policies'][a])
                    self.assertEqual(env['ZERG_DISTILL_VERSION'], a)
                    self.assertEqual(env['ZERG_DISTILL_MODEL_SHA256'], ev.MODEL_HASH)
                self.assertNotEqual(d['policies']['v1']['selector_sha256'], d['policies']['v2']['selector_sha256'])
                for batch in two.OLD:
                    r = two.prepare(root/batch, batch=batch, dry=True)
                    self.assertEqual(r['pairs'], two.old_pairs(batch))
                    self.assertEqual(r['replay_design_sha256'], two.OLD[batch])
                subset = two.prepare(root/'targeted', batch=batch, dry=True, ids=[15,90,91,101])
                self.assertEqual([p['pair_id'] for p in subset['pairs']], ['pair-015','pair-090','pair-091','pair-101'])
                for collision in (d['seed']+238, r['pairs'][-1]['seed']):
                    with self.assertRaisesRegex(ValueError, 'previously used seed'):
                        two.prepare(root/'duplicate', seed=collision, dry=True)
                with self.assertRaisesRegex(ValueError, 'output already exists'):
                    two.prepare(root/'fresh', dry=True)
                with self.assertRaisesRegex(ValueError, 'not a complete'):
                    two.compare(root/'fresh')

    def test_archive_step_evidence_and_corruption(self):
        batch = list(two.OLD)[-1]
        archive = ev.RUNS/batch
        old = ev.read_json(archive/'design.json')
        p = old['pairs'][0]
        m = ev.read_json(archive/p['manifest'])
        worker = archive/p['pair_id']/'tree'
        original = ev.reference()
        contract = dict(old['model_contract'], policy_version='v1', kind='zerg_distill_v1',
                        selector_sha256=old['sources'][two.SELECTORS['v1']])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root/p['pair_id']/'v1'
            runner = work/'runner'
            (runner/'distill').mkdir(parents=True)
            (work/'shadow').symlink_to(worker/'shadow')
            for name in ('manifest.json', 'episodes.json', 'score_details', 'events.jsonl'):
                (runner/name).symlink_to(worker/'runner'/name)
            for name in ('run.json', 'summary.json'):
                meta = ev.read_json(worker/'runner'/name)
                meta['zerg_distill'].update(contract)
                ev.write_json(runner/name, meta)
            path = runner/'distill/episode-0001.json'
            # Annotate one real old v1 receipt with new audit fields, in temp only.
            receipt = ev.read_json(worker/'runner/distill/episode-0001.json')
            receipt.update(contract)
            for row in receipt['decisions']:
                row.update(policy_version='v1', selector_sha256=contract['selector_sha256'], selection_reason='v1_cart_argmax')
            ev.write_json(path, receipt)
            def audit():
                return ev.audit_arm(root, p, 'v1', m, original, old, contract)
            result = audit()
            self.assertEqual(result['selection_reasons'], {'v1_cart_argmax': result['steps']})
            for key, value in [('policy_version','v2'), ('selector_sha256','0'*64), ('selected_candidate','bad'),
                               ('selection_reason',''), ('committed',False), ('macro_receipt','unknown')]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    bad = copy.deepcopy(receipt)
                    bad['decisions'][0][key] = value
                    path.unlink()
                    ev.write_json(path, bad)
                    audit()
            receipt['decisions'].pop()
            path.unlink()
            ev.write_json(path, receipt)
            with self.assertRaisesRegex(ValueError, 'missing decision receipt'):
                audit()
