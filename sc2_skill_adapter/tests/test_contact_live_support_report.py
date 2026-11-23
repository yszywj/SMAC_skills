import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import contact_live_support_report as report


class LiveSupportReportTest(unittest.TestCase):
    def test_submitted_actions_only(self):
        state = dict(game_loop=10, allies={'u0': dict(type='Zergling', alive=True),
                                           'u1': dict(type='Hydralisk', alive=True)})
        obs = dict(event='skill_observation', episode=1, step=0, payload=dict(state=state))
        dec = dict(event='skill_step', episode=1, step=0, game_loop=10, actions={'u0': 7, 'u1': 3})
        step = dict(event='step', episode=1, step=1, game_loop=10, actions=[7, 3])
        end = dict(event='episode_completed', episode=1, steps=1)
        with TemporaryDirectory() as tmp:
            def audit():
                (Path(tmp) / 'events.jsonl').write_text('\n'.join(map(json.dumps, (obs, dec, step, end))))
                return report._audit(Path(tmp), 'support')

            self.assertEqual(audit(), [])
            dec['live_support'] = {}
            self.assertEqual(audit(), [])
            dec['live_support'] = {'u1': dict(before=3, after=3)}
            self.assertEqual(audit(), [])
            dec['live_support']['u1']['before'] = 4
            self.assertEqual([c['actor'] for c in audit()], ['u1'])
            step['game_loop'] += 8
            with self.assertRaisesRegex(ValueError, 'clock'):
                audit()
            step['game_loop'] = 10
            dec['live_support']['u1']['after'] = 2
            with self.assertRaisesRegex(ValueError, 'step.actions'):
                audit()
            dec['live_support'] = {'u0': dict(before=2, after=7)}
            with self.assertRaisesRegex(ValueError, 'changed Zergling'):
                audit()

    def test_paired_report(self):
        def run(folder, name, m):
            support = folder.parent.name == 'support'
            seed = m['episodes'][0]['sc2_seed']
            return [dict(initial_state_sha256='a'*64, outcome='win' if support == (seed < 5) else 'loss',
                         damage_by_type={s: {u: i + support * j for i, u in enumerate(report.TYPES)}
                                         for j, s in enumerate(report.SIDES, 1)})]

        with TemporaryDirectory() as tmp, patch.object(report, '_run', side_effect=run), patch.object(
                report, '_audit', side_effect=lambda f, l: [{}]*2 if l == 'support' and f.stem.endswith('_0') else []):
            batch, jobs = Path(tmp), []
            for name, roster in report.COMPOSITIONS.items():
                team = [u.lower() for u, n in roster.items() for _ in range(n)]
                for seed in range(15):
                    x = 16 if seed < 5 else 8
                    cfg = dict(ally_start_positions=dict(item=[[x, 16]]*10),
                               enemy_start_positions=dict(item=[[32-x, 16]]*10),
                               team_gen=dict(ally_team=team, enemy_team=team))
                    m = dict(version=1, episodes=[dict(sc2_seed=seed, episode_config=cfg, sha256=report._digest(cfg))])
                    path = batch / f'{name}_{seed}.json'
                    path.write_text(json.dumps(dict(m, sha256=report._digest(m))))
                    jobs.extend((v, name, path) for v in report.LABELS)
            result = report.summarize(batch, reversed(jobs))
            self.assertEqual(json.loads((batch/'comparison.json').read_text()), result)
            total = result['total']
            self.assertEqual((total['gained_wins'], total['dropped_wins']), (10, 20))
            self.assertEqual((total['support_changed_games'], total['support_submitted_changes']), (2, 4))
            self.assertEqual(result['by_layout']['center']['pairs'], 10)
            self.assertEqual(result['by_composition'][name]['by_layout']['scattered']['pairs'], 10)
            self.assertEqual(result['episodes'][0]['damage_delta_support_minus_baseline'],
                             {s: dict(Zergling=j, Hydralisk=j) for j, s in enumerate(report.SIDES, 1)})
            with self.assertRaisesRegex(ValueError, 'missing policy'):
                report.summarize(batch, jobs[:-1])
