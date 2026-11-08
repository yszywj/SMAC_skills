"""Offline checks for planning watchdog and recovered-input training compatibility."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from decision_tree.local.authored import load_datasets, prepare
from decision_tree.local.dataset import build_dataset
from decision_tree.local.study import save_once
from decision_tree.scripts import local_campaign_plan as planner
from decision_tree.tests.test_local_pipeline import paired_batch


class CampaignPlanTests(unittest.TestCase):
    def test_idle_watchdog_kills_only_the_launched_command(self):
        command = ['test-planner']
        child = Mock(pid=1234)
        child.wait.side_effect = subprocess.TimeoutExpired(command, 5)
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(planner.time, 'monotonic', side_effect=[0, 10, 31]), \
                patch.object(planner, 'terminate_owned') as terminate:
            with self.assertRaisesRegex(TimeoutError, 'no progress'):
                planner.watch(child, command, Path(folder), 30)
            self.assertEqual(terminate.call_args.args[0:2], (child, command))
            self.assertEqual(terminate.call_args.args[2]['pid'], 1234)

    def test_base_round_can_span_multiple_idle_windows_while_progressing(self):
        command = ['test-planner']
        child = Mock(pid=1234)
        child.wait.side_effect = [subprocess.TimeoutExpired(command, 5) for _ in range(2)] + [0]
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(planner.time, 'monotonic', side_effect=[0, 25, 50]), \
                patch.object(planner, 'progress_marker', side_effect=[None, (1, 1), (2, 2)]), \
                patch.object(planner, 'terminate_owned') as terminate:
            self.assertEqual(planner.watch(child, command, Path(folder), 30), 0)
            terminate.assert_not_called()

    def test_recovered_and_deferred_entries_reaudit_in_authored_prepare(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            _, fixture, base = paired_batch(folder / 'synthetic', 'recovery_contract', 1.0)
            entries = []
            for i, source in enumerate(fixture['provenance']['arms']):
                entry = dict(source['entry'])
                if i == 0:
                    entry.update(path=str(folder/'absent'), recovery_status='deferred', attempts=[])
                else:
                    entry.update(recovery_status='recovered', attempts=[{'status': 'exited', 'returncode': 0}])
                entries.append(entry)
            effective_plan = {'arms': entries, 'scenes': [{'group_id': r['group_id']} for r in fixture['rows']],
                              'reference_policy': None, 'recovery_protocol_sha256': 'synthetic-contract'}
            dataset = build_dataset(base, entries, effective_plan)
            self.assertEqual(len(dataset['rows']), 7)
            self.assertEqual(len(dataset['rejected']), 1)
            path = folder/'recovered-paired.json'
            save_once(path, dataset)
            audited = load_datasets([path])
            prepared = prepare(audited)
            self.assertEqual(len(prepared['training_rows']) + len(prepared['validation_rows']), 7)
            self.assertEqual(len(prepared['split']['training_group_ids'])
                             + len(prepared['split']['validation_group_ids']), 8)


if __name__ == '__main__':
    unittest.main()
