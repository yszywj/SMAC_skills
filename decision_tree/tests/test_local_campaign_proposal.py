"""Synthetic gain/veto proposal checks; no game process or experiment data."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from decision_tree.local.authored import main as authored_main
from decision_tree.scripts.local_campaign_proposal import main, read, verify
from decision_tree.tests.test_local_authored import structure
from decision_tree.tests.test_local_pipeline import paired_batch


class CampaignProposalTests(unittest.TestCase):
    def test_gain_and_veto_proposals_match_models_and_detect_ledger_tamper(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            shape = root / 'structure.json'
            shape.write_text(json.dumps(structure()), encoding='utf-8')
            for kind, measured_gain in [('gain', 2), ('veto', -2)]:
                dataset, _, _ = paired_batch(root/(kind+'-data'), 'proposal-'+kind, measured_gain)
                prepared, output = root/(kind+'-prepared'), root/(kind+'-model')
                old = ['--old-model', str(root/'gain-model/model.json')] if kind == 'veto' else []
                authored_main(['prepare', '--dataset', str(dataset), '--output', str(prepared), '--kind', kind, *old])
                path = root/(kind+'-proposal.json')
                main(['propose', '--prepared', str(prepared), '--structure', str(shape), '--output', str(path)])
                item = read(path)
                self.assertEqual(item['leaf_statistics'][0]['sign_counts'], {'positive': 6, 'negative': 0, 'zero': 0})
                self.assertEqual(item['leaf_statistics'][0]['value'], 2)
                authored_main([kind, '--dataset', str(dataset), '--structure', str(shape), '--split',
                               str(prepared/'split.json'), '--output', str(output), *old])
                self.assertTrue(verify(item, output)['verified'])
            report_path = output/'report.json'
            report = read(report_path)
            report['leaf_statistics'][0]['records'].pop()
            report_path.write_text(json.dumps(report), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'evidence or support'):
                verify(item, output)

    def test_different_scene_mass_and_fixed_routes(self):
        from decision_tree.local.authored import training_summary
        from decision_tree.scripts.local_campaign_proposal import proposal
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for i, (group, ready, gain) in enumerate([('A', 0, 4), ('A', 0, 4), ('B', 0, -2), ('A', 1, 0)]):
                rows.append({'group_id': group, 'anchor_id': str(i), 'candidate_id': 'r', 'gain': gain,
                             'candidate_family': 'retarget', 'features': {'actor_weapon_ready': ready}})
            split = {'training_group_ids': ['A', 'B'], 'validation_group_ids': ['C']}
            for name, value in [('training_rows.json', rows), ('training_summary.json', training_summary(rows, ['actor_weapon_ready'])),
                                ('preparation.json', {'kind':'gain', 'dataset_sources': []}), ('split.json', split)]:
                (root/name).write_text(json.dumps(value), encoding='utf-8')
            node = {'feature':'actor_weapon_ready', 'threshold':0.5, 'left':{'leaf':True}, 'right':{'leaf':True}}
            value = proposal(root, structure(node), min_groups=2)
            left, right = value['leaf_statistics']
            self.assertEqual((left['route'], left['value'], left['groups'], left['samples'], left['supported']),
                             ('L', 1, 2, 3, True))
            self.assertEqual((right['route'], right['value'], right['sign_counts']['zero'], right['supported']),
                             ('R', 0, 1, False))
            rows[0]['group_id'] = 'C'
            (root/'training_rows.json').write_text(json.dumps(rows), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'non-training'):
                proposal(root, structure(node))


if __name__ == '__main__':
    unittest.main()
