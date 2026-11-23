"""Offline boundary checks; no replay decoder or SC2 process required."""
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import contact_early_report as report


class ContactEarlyReportTest(unittest.TestCase):
    rosters = {'1': {'z': 'Zergling', 'a': 'Hydralisk', 'b': 'Hydralisk'}, '2': {'c': 'Hydralisk'}}

    def test_first_death_and_cutoff(self):
        for deaths, side in (({}, 'none'), ({'a': 18}, 'ally'), ({'c': 24}, 'enemy'),
                             ({'c': 18, 'a': 18}, 'same_frame'), ({'c': 17, 'a': 18}, 'enemy')):
            game = report._opening(self.rosters, deaths, {})
            self.assertEqual(game['first_hydra_death_side'], side)
            self.assertIsNone(game['participation_before'])
        for cut, before, through in ((10, 0, 0.5), (17, 0.5, 0.5), (18, 0.5, 1), (20, 1, 1)):
            game = report._opening(self.rosters, {'z': cut}, {'a': 10, 'b': 18})
            self.assertEqual((game['participation_before'], game['participation_through']), (before, through))

    def test_paired_design_and_output(self):
        def run(folder, name, manifest):
            game = dict(initial_state_sha256='0'*64, outcome='loss' if folder.parent.name == 'baseline' else 'win',
                        damage_by_type={s: dict.fromkeys(report.TYPES, 1) for s in report.SIDES},
                        **report._opening(self.rosters, {}, {}))
            return [game] * len(manifest['episodes'])

        with TemporaryDirectory() as tmp, patch.object(report, '_run', side_effect=run):
            batch, jobs = Path(tmp), []
            for name, composition in report.COMPOSITIONS.items():
                team = [u.lower() for u, n in composition.items() for _ in range(n)]
                for shard in range(15):
                    path = batch / f'{name}_{shard}.json'
                    episodes = []
                    for i in range(5):
                        seed = 5 * shard + i
                        cfg = dict(team_gen=dict(ally_team=team, enemy_team=team), index=seed)
                        episodes.append(dict(sc2_seed=seed, episode_config=cfg, sha256=report._digest(cfg)))
                    m = dict(version=1, episodes=episodes)
                    path.write_text(json.dumps(dict(m, sha256=report._digest(m))))
                    jobs.extend((label, name, path) for label in report.LABELS)
            result = report.summarize(batch, list(reversed(jobs)))
            self.assertEqual(json.loads((batch / 'comparison.json').read_text()), result)
            self.assertEqual(result['total']['gained_wins'], 150)
            self.assertEqual(result['total']['dropped_wins'], 0)
            self.assertEqual(result['total']['paired_outcomes']['loss']['win'], 150)
            stats = result['by_composition'][name]['versions']['early']
            self.assertEqual(stats['damage_by_type']['enemy']['Hydralisk'], 75)
            self.assertEqual(stats['no_ally_dog_death'], 75)
            self.assertIsNone(stats['mean_participation_before'])
            with self.assertRaisesRegex(ValueError, 'missing policy'):
                report.summarize(batch, jobs[:-1])
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                report.summarize(batch, jobs + jobs[:1])
            m['episodes'][0]['sc2_seed'] = 0
            path.write_text(json.dumps(dict(m, sha256=report._digest(m))))
            with self.assertRaisesRegex(ValueError, 'repeated seed'):
                report.summarize(batch, jobs)
