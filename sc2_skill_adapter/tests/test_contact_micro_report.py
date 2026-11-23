"""Real batch preparation + synthetic runner artifacts. No game is launched."""
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from contact_micro_report import MICRO_LABELS, TYPES, micro_report
from run_contact_zerg_ab import micro_args, play_micro, prepare_micro, write_json


class MicroReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.batch = Path(tmp.name)
        cls.jobs = prepare_micro(cls.batch, 123, os.environ)
        design = json.loads((cls.batch / 'design.json').read_text())
        cls.files = design['snapshot_files']
        for label, name, path in cls.jobs:
            directory = cls.batch / label / path.stem
            for subdir in ('damage_by_type', 'micro_triggers'):
                (directory / subdir).mkdir(parents=True)
            manifest = json.loads(path.read_text())
            write_json(directory / 'manifest.json', manifest)
            switch = label == 'micro8-hydra'
            timing = dict(mode=label, macro_loops=8, engine_step_loops=1)
            common = dict(method='adaptive-skill', skill_version='tactical_v1', execution_mode=label,
                          manifest_sha256=manifest['sha256'])
            skills = {k: v for k, v in cls.files.items() if k.startswith('skills/tactical_v1/') and k.endswith('.py')}
            write_json(directory / 'run.json', dict(common, macro_loops=8, engine_step_loops=1,
                       damage_by_type_enabled=True, env_args=manifest['env_args'], skill_parameters={},
                       versions={}, source_hashes=dict(cls.files), skill_source_hashes=skills))
            write_json(directory / 'summary.json', dict(common, status='complete', completed_episodes=5))
            games = []
            for i, row in enumerate(manifest['episodes'], 1):
                state = 'submitted' if switch else 'legal_control_off'
                record = dict(episode=i, steps=3, scenario_sha256=row['sha256'], initial_state_sha256='a'*64,
                              outcome='win' if switch else 'loss', micro_trigger=dict(timing, windows=3,
                              authorizations=2, states={state: 1, 'already_ordered': 1}))
                games.append(record)
                write_json(directory / 'micro_triggers' / f'episode-{i:04d}.json', dict(timing, episode=i,
                    authorizations=[dict(episode=i, status=state, submitted_loop=5 if switch else None),
                                    dict(episode=i, status='already_ordered', submitted_loop=None)]))
                write_json(directory / 'damage_by_type' / f'episode-{i:04d}.json', dict(episode=i,
                    scenario_sha256=row['sha256'], manifest_sha256=manifest['sha256'], players={
                        p: dict(damage_dealt_by_source_type=dict(zip(TYPES, [int(p)+switch, 0, 10*int(p)-switch, 0])))
                        for p in ('1', '2')}))
            write_json(directory / 'episodes.json', games)

    def test_snapshot_and_only_mode_changes(self):
        off, hydra = MICRO_LABELS
        self.assertEqual(self.jobs[0][1:], self.jobs[1][1:])
        first = self.jobs[0][2]
        a, b = (micro_args(label, first, 'OUTPUT') for label in MICRO_LABELS)
        self.assertEqual([(x, y) for x, y in zip(a, b) if x != y], [(off, hydra)])
        for name, checksum in self.files.items():
            paths = [self.batch / label / name for label in ('source_snapshot', off, hydra)]
            for path in paths:
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), checksum)
            self.assertFalse(paths[1].samefile(paths[2]))
        with patch('run_contact_zerg_ab.subprocess.run') as child:
            play_micro(self.batch, self.jobs[1], {})
        args, kw = child.call_args
        self.assertEqual(args[0][args[0].index('run'):], micro_args(hydra, first, self.batch / hydra / first.stem))
        self.assertEqual(Path(kw['env']['SMAC_DAMAGE_MAP_DIR']), kw['cwd'] / 'sc2_skill_adapter/damage_maps')

    def test_full_paired_summary(self):
        report = micro_report(self.batch)
        self.assertEqual(len(report['episodes']), 150)
        self.assertEqual([v['pairs'] for v in report['by_composition'].values()], [75, 75])
        total = report['total']
        self.assertEqual((total['gained_wins'], total['dropped_wins']), (150, 0))
        off, hydra = (total['versions'][label] for label in MICRO_LABELS)
        self.assertEqual(off['micro_trigger']['submitted'], 0)
        self.assertEqual(hydra['micro_trigger']['submitted'], 150)
        self.assertEqual(hydra['micro_trigger']['authorizations'], 300)
        self.assertEqual(hydra['damage_by_type']['enemy']['Hydralisk'], 2850)
        self.assertEqual(total['damage_delta_hydra_minus_off']['ally']['Zergling'], 150)

    def test_corrupt_artifacts_fail_closed(self):
        label, _, path = self.jobs[0]
        out = self.batch / label / path.stem
        for filename, keys, value in [
                ('episodes.json', (0, 'initial_state_sha256'), 'b'*64),
                ('episodes.json', (0, 'scenario_sha256'), 'b'*64),
                ('episodes.json', (0, 'micro_trigger', 'authorizations'), 0),
                ('run.json', ('execution_mode',), 'legacy8'),
                ('run.json', ('source_hashes', 'sc2_skill_adapter/bridge/micro_trigger.py'), 'b'*64),
                ('micro_triggers/episode-0001.json', ('authorizations', 0, 'submitted_loop'), 10),
                ('damage_by_type/episode-0001.json', ('manifest_sha256',), 'b'*64),
                ('damage_by_type/episode-0001.json', ('players', '2', 'damage_dealt_by_source_type'), {})]:
            with self.subTest(file=filename, keys=keys):
                data_path = out / filename
                original = data_path.read_text()
                obj = cursor = json.loads(original)
                for k in keys[:-1]:
                    cursor = cursor[k]
                cursor[keys[-1]] = value
                write_json(data_path, obj)
                try:
                    with self.assertRaises(ValueError):
                        micro_report(self.batch)
                finally:
                    data_path.write_text(original)


if __name__ == '__main__':
    unittest.main()
