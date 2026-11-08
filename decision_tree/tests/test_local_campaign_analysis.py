"""Synthetic evidence tests for the training-only campaign reader."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from decision_tree.local.authored import main as prepare_main
from decision_tree.scripts.local_campaign_analysis import (cases, load_training,
                                                          summarize, validation_report)
from decision_tree.tests.test_local_pipeline import paired_batch


class CampaignAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path, self.data, self.base = paired_batch(self.root/'data', 'analysis-only', 2)
        self.prepared = self.root/'prepared'
        with redirect_stdout(io.StringIO()):
            prepare_main(['prepare', '--dataset', str(self.path), '--output', str(self.prepared)])

    def tearDown(self):
        self.temporary.cleanup()

    def test_summary_uses_fixed_ordered_cuts_and_training_scene_counts_only(self):
        result = summarize(self.prepared, 'retarget')
        self.assertEqual(result['scope'], 'training_only')
        self.assertEqual(result['training_cohort_groups'], 6)
        self.assertEqual(result['families']['retarget']['overall']['mean'], 2)
        first = result['families']['retarget']['conditions'][0]
        self.assertEqual((first['feature'], first['threshold']), ('actor_weapon_ready', 0.5))
        self.assertNotIn('best', result)
        self.assertNotIn('validation', result)

    def test_resolved_cases_include_only_selected_training_episodes_and_actual_pair(self):
        evidence = load_training(self.prepared)
        result = cases(self.prepared, family='retarget', sign='positive', limit=2, radius=1)
        self.assertEqual(len(result['cases']), 2)
        groups = {case['group_id'] for case in result['cases']}
        self.assertEqual(len(groups), 2)
        self.assertFalse(groups & evidence['validation_ids'])
        for case in result['cases']:
            self.assertEqual(case['original_candidate_gain'], 2)
            self.assertEqual(case['training_label'], 2)
            self.assertEqual({r['arm'] for r in case['trace']}, {'BASE', 'CANDIDATE'})
            anchor = [r for r in case['trace'] if r['step'] == 1]
            self.assertEqual([r['final_action'] for r in anchor], [6, 7])
            self.assertTrue(all(r['submission'] == 'accepted' for r in case['trace']))
        with self.assertRaisesRegex(ValueError, 'not in training'):
            cases(self.prepared, group_id=next(iter(evidence['validation_ids'])), sign='positive')

    def test_injected_validation_row_is_rejected_before_any_summary(self):
        evidence = load_training(self.prepared)
        row = next(r for r in self.data['rows'] if r['group_id'] in evidence['validation_ids'])
        path = self.prepared/'training_rows.json'
        rows = json.loads(path.read_text())
        rows.append(row)
        path.write_text(json.dumps(rows), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'non-training'):
            summarize(self.prepared)

    def test_changed_pair_evidence_rejected_and_validation_requires_explicit_built_report(self):
        with (self.base/'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'event stream changed'):
            cases(self.prepared, sign='positive', limit=1)
        report = self.root/'report.json'
        report.write_text(json.dumps({'validation': {'mean_gain': 123}}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'already-built'):
            validation_report(report)
        report.write_text(json.dumps({'construction': 'LLM_supplied_predicates_and_topology',
                                      'model_hash': 'hash', 'validation': {'mean_gain': 1}}), encoding='utf-8')
        self.assertEqual(validation_report(report)['scope'], 'post_construction_validation_only')


if __name__ == '__main__':
    unittest.main()
