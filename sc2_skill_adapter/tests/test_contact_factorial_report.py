"""Offline integration tests: synthetic completed shards, no runner or SC2."""
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from contact_factorial_report import COMPOSITIONS, LABELS, MODE, TYPES, digest, summarize, write_json


class FactorialReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.batch = batch = Path(tmp.name)
        (batch / 'manifests').mkdir()
        design = dict(seed=100, labels=list(LABELS), policy_files={}, shards={})
        skill = 'skills/tactical_v1/policy.py'
        for label in LABELS:
            files = design['policy_files'][label] = {}
            for name in (skill, 'sc2_skill_adapter/bridge/micro_trigger.py', 'sc2_skill_adapter/damage_maps/10gen_zerg.SC2Map'):
                path = batch / label / name
                path.parent.mkdir(parents=True)
                data = (label if name == skill else name).encode()
                path.write_bytes(data)
                files[name] = hashlib.sha256(data).hexdigest()
        env = dict(map_name='10gen_zerg', unit_balance='native', step_mul=8, use_unit_ranges=True,
                   min_attack_range=2, capability_config=dict(n_units=10, n_enemies=10, start_positions={},
                   team_gen=dict(unit_types=['zergling', 'hydralisk'])))
        timing = dict(mode=MODE, macro_loops=8, engine_step_loops=1)
        outcomes = [('win', 'loss', 'draw', 'timeout', 'win'), ('draw', 'win', 'draw', 'timeout', 'win'),
                    ('loss',)*5, ('win',)*5]
        for group, (name, roster) in enumerate(COMPOSITIONS.items()):
            team = [unit.lower() for unit, count in roster.items() for _ in range(count)]
            for offset in range(0, 75, 5):
                shard = f'{name}_{offset:02d}'
                episodes = []
                for i in range(5):
                    x = 16 if i == 0 else i
                    cfg = dict(team_gen=dict(ally_team=team, enemy_team=team),
                               ally_start_positions=dict(item=[[x, 16]]*10),
                               enemy_start_positions=dict(item=[[32-x, 16]]*10))
                    episodes.append(dict(index=offset+i, sc2_seed=100+group*75+offset+i,
                                         episode_config=cfg, sha256=digest(cfg)))
                m = dict(version=1, seed=100+group*75, env_args=env, episodes=episodes)
                m['sha256'] = digest(m)
                design['shards'][shard] = dict(composition=name, manifest_sha256=m['sha256'])
                write_json(batch / 'manifests' / f'{shard}.json', m)
                for n, label in enumerate(LABELS):
                    work = batch / label / shard
                    for sub in ('damage_by_type', 'micro_triggers'):
                        (work / sub).mkdir(parents=True)
                    files = design['policy_files'][label]
                    meta = dict(method='adaptive-skill', skill_version='tactical_v1', execution_mode=MODE,
                                manifest_sha256=m['sha256'])
                    write_json(work / 'manifest.json', m)
                    write_json(work / 'run.json', dict(meta, macro_loops=8, engine_step_loops=1, env_args=env,
                               source_hashes=files, skill_source_hashes={skill: files[skill]}, skill_parameters={},
                               damage_by_type_enabled=True, versions={}))
                    write_json(work / 'summary.json', dict(meta, status='complete', completed_episodes=5))
                    records = []
                    for i, row in enumerate(episodes, 1):
                        filename = f'episode-{i:04d}.json'
                        records.append(dict(episode=i, scenario_sha256=row['sha256'], steps=2,
                            initial_state_sha256=digest(row['sc2_seed']), outcome=outcomes[n][i-1],
                            micro_trigger=dict(timing, windows=2, authorizations=n, states={'submitted': n} if n else {})))
                        write_json(work / 'micro_triggers' / filename, dict(timing, episode=i, authorizations=[
                            dict(episode=i, status='submitted', submitted_loop=7)]*n))
                        write_json(work / 'damage_by_type' / filename, dict(meta, episode=i, scenario_sha256=row['sha256'],
                            players={p: dict(damage_dealt_by_source_type=dict(zip(TYPES, (n+1, 0, 10*int(p), 0))))
                                     for p in ('1', '2')}))
                    write_json(work / 'episodes.json', records)
        write_json(batch / 'design.json', design)

    def test_four_way_report(self):
        report = summarize(self.batch)
        self.assertEqual(json.loads((self.batch / 'comparison.json').read_text()), report)
        self.assertEqual(len(report['episodes']), 150)
        self.assertEqual(report['total']['pairs'], 150)
        self.assertEqual(report['total']['paired_vs_baseline'], {
            'simple_hydra': dict(dropped_wins=30, gained_wins=30),
            'simple_ling': dict(dropped_wins=60, gained_wins=0),
            'simple_both': dict(dropped_wins=0, gained_wins=90)})
        self.assertEqual(report['total']['versions']['simple_hydra']['win_rate'], 0.4)
        self.assertEqual(report['by_layout']['center']['versions']['simple_hydra']['win_rate'], 0)
        self.assertEqual(report['by_layout']['scattered']['versions']['simple_hydra']['win_rate'], 0.5)
        for group in report['by_composition'].values():
            self.assertEqual(group['pairs'], 75)
            self.assertEqual(group['by_layout']['center']['pairs'], 15)
        both = report['total']['versions']['simple_both']
        self.assertEqual(both['micro_trigger'], dict(windows=300, authorizations=450, submitted=450, states={'submitted': 450}))
        self.assertEqual(both['damage_by_type']['ally']['Zergling'], 600)
        self.assertEqual(both['damage_by_type']['enemy']['Hydralisk'], 3000)

    def test_corruption_rejected_before_writing(self):
        shard = f'{next(iter(COMPOSITIONS))}_00'
        work = self.batch / 'simple_both' / shard
        comparison = self.batch / 'comparison.json'
        comparison.write_text('previous report')
        for path, keys, value in [
                (self.batch / 'design.json', ['shards', shard, 'manifest_sha256'], 'bad'),
                (self.batch / 'design.json', ['policy_files', 'baseline', 'skills/tactical_v1/policy.py'], 'bad'),
                (work / 'run.json', ['execution_mode'], 'micro8-off'),
                (work / 'run.json', ['skill_source_hashes', 'skills/tactical_v1/policy.py'], 'bad'),
                (work / 'episodes.json', [0, 'initial_state_sha256'], '0'*64),
                (work / 'episodes.json', [0, 'micro_trigger', 'authorizations'], 0),
                (work / 'damage_by_type/episode-0001.json', ['players', '2', 'damage_dealt_by_source_type', 'Hydralisk'], -1)]:
            with self.subTest(path=path, keys=keys):
                original = path.read_text()
                data = obj = json.loads(original)
                for key in keys[:-1]:
                    obj = obj[key]
                obj[keys[-1]] = value
                write_json(path, data)
                try:
                    with self.assertRaises(ValueError):
                        summarize(self.batch)
                    self.assertEqual(comparison.read_text(), 'previous report')
                finally:
                    path.write_text(original)


if __name__ == '__main__':
    unittest.main()
