"""PS5.1 launcher integration in isolated repositories; no native game imports.

Only the interpreter path is substituted. The fixture runner signs/checks real
manifest structure and writes inert records/replays, never starting SC2.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

LAUNCHER = Path(__file__).resolve().parents[1] / 'scripts/run_tactical_matrix.ps1'
GROUPS = ('p_6z4s', 'p_4z5s1c', 'p_2z6s2c', 't_6m4r', 't_4m5r1h',
          't_2m6r2h', 'z_6l4h', 'z_4l5h1b', 'z_2l6h2b')
PRIOR = 'tactical_all_races_10v10_9x20_mechanisms_20261115'
COMPARISON = 'tactical_10x3x3_20261115_212459'
POWERSHELL = shutil.which('powershell.exe')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(',', ':')).encode()).hexdigest()


@unittest.skipUnless(os.name == 'nt' and POWERSHELL, 'Windows PowerShell 5.1 required')
class TacticalMatrixLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='tactical-matrix-')
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / 'repo with spaces'
        self.adapter = self.repo / 'sc2_skill_adapter'
        self.prior = self.repo / 'runs' / PRIOR
        self.comparison = self.repo / 'runs' / COMPARISON
        self.output = self.repo / 'runs/mock_batch'
        self.native = Path(self.tmp.name) / 'native/smacv2/env/starcraft2'
        frozen = {}
        for relative in ('skills/tactical_v1', 'skills/current',
                         'sc2_skill_adapter/bridge', 'sc2_skill_adapter/display_guard'):
            path = self.repo / relative / '__init__.py'
            path.parent.mkdir(parents=True)
            path.write_text('# inert fixture\n')
            frozen[relative + '/__init__.py'] = sha(path)
        defaults = self.repo / 'skills/tactical_v1/defaults.json'
        write_json(defaults, {'test': True})
        frozen['skills/tactical_v1/defaults.json'] = sha(defaults)
        self.script = self.adapter / 'scripts/run_tactical_matrix.ps1'
        self.script.parent.mkdir()
        source = LAUNCHER.read_text(encoding='utf-8')
        original = "$python = Join-Path $adapterRoot '.venv\\Scripts\\python.exe'"
        self.assertEqual(source.count(original), 1)
        interpreter = sys.executable.replace("'", "''")
        self.script.write_text(source.replace(original, f"$python = '{interpreter}'"), encoding='utf-8')
        self.native.mkdir(parents=True)
        for name in ('starcraft2.py', 'wrapper.py', 'distributions.py'):
            path = self.native / name
            path.write_text('# inert native dependency\n')
            frozen['dependencies/smacv2/' + name] = sha(path)
        maps = {}
        for race in ('protoss', 'terran', 'zerg'):
            path = self.native / f'maps/SMAC_Maps/10gen_{race}.SC2Map'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('not a game map: ' + race).encode())
            maps[race[0]] = path
            frozen['dependencies/smacv2/maps/SMAC_Maps/' + path.name] = sha(path)
        design = {'groups': {}}
        for group in GROUPS:
            episodes = []
            for i in range(20):
                config = {'group': group, 'index': i, 'positions': [1.0, 0.037123456789, 2.25]}
                episodes.append({'index': i, 'sc2_seed': 170+i,
                                 'episode_config': config, 'sha256': digest(config)})
            manifest = {'version': 1, 'seed': 170, 'env_args': {'step_mul': 8, 'difficulty': '7',
                        'unit_balance': 'native', 'capability_config': {'n_units': 10, 'n_enemies': 10}},
                        'episodes': episodes}
            manifest['sha256'] = digest(manifest)
            path = self.prior / f'manifests/{group}.json'
            write_json(path, manifest)
            design['groups'][group] = {'manifest_file_sha256': sha(path),
                                      'map': str(maps[group[0]]), 'map_sha256': sha(maps[group[0]])}
            rows = [{'episode': i+1, 'scenario_sha256': episodes[i]['sha256'],
                     'initial_state_sha256': f'{group}:initial:{i}',
                     'outcome': 'win' if i == 1 else 'loss'} for i in range(20)]
            metadata = {'method': 'adaptive-skill', 'skill_version': 'tactical_v1',
                        'manifest_sha256': manifest['sha256'],
                        'skill_parameters': {'posture': 'balanced', 'max_steps': 4},
                        'skill_source_hashes': {k: v for k, v in frozen.items()
                                                if k.startswith('skills/tactical_v1/') and k.endswith('.py')}}
            write_json(self.comparison / f'current/{group}/episodes.json', rows)
            write_json(self.comparison / f'current/{group}/summary.json',
                       dict(metadata, status='complete', completed_episodes=20))
            write_json(self.comparison / f'current/{group}/run.json',
                       dict(metadata, env_args=manifest['env_args'], scenario_count=20,
                            source_hashes={'sc2_skill_adapter/bridge/__init__.py':
                                                     frozen['sc2_skill_adapter/bridge/__init__.py']}))
        write_json(self.prior / 'design.json', design)
        write_json(self.prior / 'source_freeze.json', {'source_hashes': frozen})
        write_json(self.comparison / 'source_freeze.json', {'source_hashes': frozen})
        self.set_runner()

    def set_runner(self, mode='normal'):
        body = textwrap.dedent('''\
            import hashlib, json, struct, sys
            from pathlib import Path
            args = sys.argv[1:]
            repo = Path(__file__).resolve().parents[1]
            out = Path(args[args.index('--output-dir')+1])
            group = out.name
            manifest_path = Path(args[args.index('--manifest')+1])
            assert args[0] == 'run'
            for flag, expected in [('--method', 'adaptive-skill'), ('--skill-version', 'tactical_v1'),
                                   ('--adaptive-posture', 'balanced'), ('--skill-max-steps', '4')]:
                assert args[args.index(flag)+1] == expected
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            count = len(manifest['episodes'])
            body = {k:v for k,v in manifest.items() if k != 'sha256'}
            assert manifest['sha256'] == hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=True,
                                                        separators=(',', ':')).encode()).hexdigest()
            assert [e['sc2_seed'] for e in manifest['episodes']] == list(range(170, 170+count))
            with (repo / 'calls.jsonl').open('a') as log:
                log.write(json.dumps({'group': group, 'argv': args, 'count': count}) + '\\n')
            out.mkdir(parents=True, exist_ok=False)
            (out / 'partial.txt').write_text('preserve evidence')
            print('fixture stdout ' + group, flush=True)
            print('fixture harmless stderr ' + group, file=sys.stderr, flush=True)
            if MODE == 'exit': sys.exit(7)
            frozen = json.loads((out.parents[1] / 'source_freeze.json').read_text())['source_hashes']
            metadata = {'method': 'adaptive-skill', 'skill_version': 'tactical_v1',
                        'manifest_sha256': manifest['sha256'],
                        'skill_parameters': {'posture': 'balanced', 'max_steps': 4},
                        'skill_source_hashes': {k:v for k,v in frozen.items()
                                               if k.startswith('skills/tactical_v1/') and k.endswith('.py')}}
            run = dict(metadata, env_args=manifest['env_args'], scenario_count=count,
                       source_hashes={'sc2_skill_adapter/bridge/__init__.py':
                                                frozen['sc2_skill_adapter/bridge/__init__.py']})
            if MODE == 'wrong_source': run['skill_source_hashes'] = {'not_frozen.py': 'bad'}
            (out / 'run.json').write_text(json.dumps(run))
            partial = (MODE == 'partial' and group == 'z_2l6h2b' or
                       MODE == 'partial_first' and group == 'p_6z4s')
            completed = count - 1 if partial else count
            rows, events = [], []
            for i in range(completed):
                ep = i+1
                replay = out / 'replays' / ('episode-%04d.SC2Replay' % ep)
                replay.parent.mkdir(exist_ok=True)
                archive = struct.pack('<4sIIHHIIII', b'MPQ\\x1a', 32, 32, 0, 3, 32, 32, 0, 0)
                replay.write_bytes(struct.pack('<4sIII', b'MPQ\\x1b', 16, 32, 16) + bytes(16) + archive)
                row = {'episode': ep, 'scenario_sha256': manifest['episodes'][i]['sha256'],
                       'initial_state_sha256': f'{group}:initial:{i}',
                       'outcome': {0:'win', 2:'draw', 3:'timeout'}.get(i, 'loss'),
                       'steps': 1, 'replay': str(replay)}
                if MODE == 'mismatch' and i == 0: row['initial_state_sha256'] = 'wrong'
                events.append(dict(event='episode_started', **row))
                events.append({'event': 'step', 'episode': ep, 'step': 1, 'game_loop': 2,
                               'terminated': True, 'feedback': {'from_game_loop': 2,
                                  'to_game_loop': 9 if MODE == 'bad_clock' and i == 0 else 10}})
                events.append(dict(event='episode_completed', **row))
                rows.append(row)
            if partial:
                events.append({'event': 'episode_started', 'episode': count})
                events.append({'event': 'run_failed', 'error': 'invalid native interval; not a loss'})
            if MODE == 'duplicate': rows[1]['episode'] = 1
            (out / 'episodes.json').write_text(json.dumps(rows))
            (out / 'events.jsonl').write_text(''.join(json.dumps(e)+'\\n' for e in events))
            summary = dict(metadata, status='failed' if partial else 'complete',
                           completed_episodes=completed, error='invalid native interval' if partial else None)
            (out / 'summary.json').write_text(json.dumps(summary))
            if MODE == 'source_drift':
                (repo / 'skills/tactical_v1/__init__.py').write_text('# changed while running')
            if MODE == 'map_drift':
                design = json.loads((repo / 'runs' / PRIOR / 'design.json').read_text())
                Path(design['groups'][group]['map']).write_text('changed map')
            if partial: sys.exit(7)
            if MODE == 'complete_exit_first' and group == 'p_6z4s': sys.exit(7)
            ''')
        (self.adapter / 'run.py').write_text(f'MODE = {mode!r}\nPRIOR = {PRIOR!r}\n'+body, encoding='utf-8')

    def launch(self, dry=False, count=None, resume=False, explicit_name=True):
        args = [POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass',
                '-File', str(self.script)]
        if explicit_name: args += ['-OutputName', 'mock_batch']
        if count is not None: args += ['-GamesPerGroup', str(count)]
        if resume: args.append('-Resume')
        if dry: args.append('-DryRun')
        env = {key: value for key, value in os.environ.items() if key.lower() != 'psmodulepath'}
        env['PSModulePath'] = str(Path(POWERSHELL).parent / 'Modules')
        return subprocess.run(args, cwd=self.repo, capture_output=True, text=True,
                              encoding='utf-8', errors='replace', timeout=50, env=env)

    def calls(self):
        path = self.repo / 'calls.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def summary(self):
        return json.loads((self.output / 'matrix_summary.json').read_text(encoding='utf-8'))

    def clear_output(self):
        self.assertTrue(self.output.resolve().is_relative_to(Path(self.tmp.name).resolve()))
        shutil.rmtree(self.output)

    def test_dry_run_checks_dependencies_without_output_or_runner(self):
        result = self.launch(dry=True, count=10)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('90 games', result.stdout)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.calls(), [])
        (self.native / 'wrapper.py').write_text('# drift')
        result = self.launch(dry=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Native dependency differs', result.stderr)

    def test_default_twenty_remains_180_and_retains_both_log_streams(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual((self.summary()['status'], self.summary()['completed_games']), ('complete', 180))
        self.assertEqual([row['group'] for row in self.calls()], list(GROUPS))
        log_bytes = (self.output / 'logs/p_6z4s.log').read_bytes()
        log = log_bytes.decode('utf-16' if log_bytes.startswith(b'\xff\xfe') else 'utf-8-sig')
        self.assertIn('fixture stdout', log)
        self.assertIn('fixture harmless stderr', log)
        self.assertNotEqual(self.launch().returncode, 0)
        self.assertEqual(len(self.calls()), 9)

    def test_twenty_games_against_latest_ten_game_baseline_marks_ninety_unpaired(self):
        for group in GROUPS:
            directory = self.comparison / 'current' / group
            selected = json.loads((self.prior / f'manifests/{group}.json').read_text())
            selected['episodes'] = selected['episodes'][:10]
            selected['sha256'] = digest({k: v for k, v in selected.items() if k != 'sha256'})
            rows = json.loads((directory / 'episodes.json').read_text())[:10]
            write_json(directory / 'episodes.json', rows)
            for name in ('summary.json', 'run.json'):
                value = json.loads((directory / name).read_text())
                value['manifest_sha256'] = selected['sha256']
                if name == 'summary.json':
                    value['completed_episodes'] = 10
                else:
                    value['scenario_count'] = 10
                write_json(directory / name, value)
        result = self.launch(count=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = self.summary()
        self.assertEqual(summary['completed_games'], 180)
        self.assertEqual(summary['available_baseline_pairs'], 90)
        for row in summary['results']:
            self.assertEqual(row['paired_games'], 10)
            self.assertEqual(row['unpaired_seeds'], list(range(180, 190)))
            self.assertEqual(row['baseline_wins'], 1)

    def test_ten_games_all_ninety_signed_prefixes_and_exact_snapshot(self):
        original_hashes = {p: sha(p) for p in self.prior.rglob('*.json')}
        result = self.launch(count=10)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        summary = self.summary()
        self.assertEqual((summary['completed_games'], summary['total_games']), (90, 90))
        self.assertEqual(summary['baseline'], str(self.comparison))
        self.assertEqual(summary['available_baseline_pairs'], 90)
        for row in summary['results']:
            self.assertEqual((row['wins'], row['losses'], row['draws'], row['timeouts']), (1, 7, 1, 1))
            self.assertEqual((row['baseline_wins'], row['win_delta'], row['win_rate']), (1, 0, .1))
            self.assertEqual((row['paired_games'], row['unpaired_seeds']), (10, []))
            self.assertEqual(row['gained_seeds'], [170])
            self.assertEqual(row['lost_seeds'], [171])
            original = json.loads((self.prior / f"manifests/{row['group']}.json").read_text())
            subset = json.loads((self.output / f"manifests/{row['group']}.json").read_text())
            self.assertEqual(subset['episodes'], original['episodes'][:10])
            self.assertNotEqual(subset['sha256'], original['sha256'])
            self.assertEqual(subset['sha256'], digest({k:v for k,v in subset.items() if k != 'sha256'}))
        self.assertEqual({p: sha(p) for p in original_hashes}, original_hashes)
        frozen = json.loads((self.output / 'source_freeze.json').read_text())
        for key, expected in frozen['source_hashes'].items():
            path = self.output / 'source_snapshot' / key
            self.assertTrue(path.resolve().is_relative_to(self.output.resolve()))
            self.assertEqual(sha(path), expected)

    def test_missing_previous_seed_is_unpaired_without_older_fallback(self):
        path = self.comparison / 'current/z_2l6h2b'
        write_json(path / 'episodes.json', json.loads((path / 'episodes.json').read_text())[:9])
        summary = json.loads((path / 'summary.json').read_text())
        summary.update(status='failed', completed_episodes=9)
        write_json(path / 'summary.json', summary)
        result = self.launch(count=10)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        summary = self.summary()
        self.assertEqual((summary['completed_games'], summary['available_baseline_pairs']), (90, 89))
        last = summary['results'][-1]
        self.assertEqual((last['paired_games'], last['unpaired_seeds']), (9, [179]))
        self.assertNotIn(179, last['lost_seeds'])
        self.assertNotIn(179, last['gained_seeds'])

    def test_failed_last_group_retains_valid_partial_nine_not_invalid_tenth(self):
        self.set_runner('partial')
        result = self.launch(count=10)
        self.assertNotEqual(result.returncode, 0)
        summary = self.summary()
        self.assertEqual((summary['status'], summary['completed_games']), ('failed', 89))
        self.assertEqual(len(summary['results']), 9)
        last = summary['results'][-1]
        self.assertEqual(last['excluded_incomplete_seeds'], [179])
        self.assertEqual((last['completed_games'], last['losses']), (9, 6))
        self.assertIn('exit 7', summary['error'])
        self.assertTrue((self.output / 'current/z_2l6h2b/partial.txt').exists())

    def test_invalid_seed_stops_preflight_before_runner(self):
        path = self.prior / 'manifests/p_6z4s.json'
        manifest = json.loads(path.read_text())
        manifest['episodes'][3]['sc2_seed'] = 999
        manifest['sha256'] = digest({k:v for k,v in manifest.items() if k != 'sha256'})
        write_json(path, manifest)
        design_path = self.prior / 'design.json'
        design = json.loads(design_path.read_text())
        design['groups']['p_6z4s']['manifest_file_sha256'] = sha(path)
        write_json(design_path, design)
        result = self.launch(dry=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Seed mismatch', result.stderr)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.calls(), [])

    def test_native_failure_before_records_preserves_artifacts_counts_zero(self):
        self.set_runner('exit')
        result = self.launch(count=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.calls()), 1)
        summary = self.summary()
        self.assertEqual((summary['status'], summary['completed_games']), ('failed', 0))
        self.assertIn('exit 7', summary['error'])
        self.assertTrue((self.output / 'current/p_6z4s/partial.txt').exists())

    def test_inconsistent_previous_completion_or_protocol_fails_preflight(self):
        path = self.comparison / 'current/p_6z4s/summary.json'
        value = json.loads(path.read_text())
        value['completed_episodes'] = 19
        write_json(path, value)
        result = self.launch(dry=True, count=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('baseline completion evidence', result.stderr)
        value['completed_episodes'] = 20
        write_json(path, value)
        path = self.comparison / 'current/p_6z4s/run.json'
        value = json.loads(path.read_text())
        value['env_args']['step_mul'] = 4
        write_json(path, value)
        result = self.launch(dry=True, count=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('environment/scenario count mismatch', result.stderr)
        self.assertEqual(self.calls(), [])
        self.assertFalse(self.output.exists())

    def test_duplicate_wrong_initial_source_or_clock_cannot_count(self):
        for mode, message in [('duplicate', 'duplicate episode'), ('mismatch', 'Paired episode mismatch'),
                              ('wrong_source', 'source hash mismatch'), ('bad_clock', 'clock/termination')]:
            with self.subTest(mode=mode):
                self.set_runner(mode)
                result = self.launch(count=10)
                self.assertNotEqual(result.returncode, 0)
                summary = self.summary()
                self.assertIn(message, summary['error'])
                self.assertEqual(summary['completed_games'], 0)
                self.clear_output()

    def test_runtime_and_native_map_drift_stop_before_next_group(self):
        for mode in ('source_drift', 'map_drift'):
            with self.subTest(mode=mode):
                self.set_runner(mode)
                source_path = self.repo / 'skills/tactical_v1/__init__.py'
                map_path = self.native / 'maps/SMAC_Maps/10gen_protoss.SC2Map'
                before_source, before_map = source_path.read_bytes(), map_path.read_bytes()
                before_calls = len(self.calls())
                result = self.launch(count=10)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(len(self.calls()), before_calls+1)
                summary = self.summary()
                self.assertIn('Runtime source changed', summary['error'])
                self.assertEqual(summary['completed_games'], 0)
                source_path.write_bytes(before_source)
                map_path.write_bytes(before_map)
                self.clear_output()

    def completed_prefix(self):
        # A complete native group can coexist with failed outer orchestration.
        # This runner remains byte-identical when the later groups resume.
        self.set_runner('complete_exit_first')
        result = self.launch(count=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.summary()['status'], self.summary()['completed_games']), ('failed', 10))
        self.assertEqual(len(self.calls()), 1)

    def output_hashes(self):
        return {p.relative_to(self.output).as_posix(): sha(p)
                for p in self.output.rglob('*') if p.is_file()}

    def test_resume_requires_explicit_existing_output(self):
        result = self.launch(resume=True, dry=True, count=10, explicit_name=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('explicit existing OutputName', result.stderr)
        result = self.launch(resume=True, dry=True, count=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('does not exist', result.stderr)
        self.assertEqual(self.calls(), [])

    def test_resume_dry_run_checks_prefix_and_launcher_upgrade_without_writes(self):
        self.completed_prefix()
        # Match the real audit-error case: stale matrix totals are not evidence
        # against an independently audited complete group's run artifacts.
        old = self.summary(); old.update(completed_games=0, results=[])
        write_json(self.output / 'matrix_summary.json', old)
        self.script.write_text(self.script.read_text(encoding='utf8') + '\n# audit-only launcher update\n', encoding='utf8')
        before = self.output_hashes()
        result = self.launch(resume=True, dry=True, count=10)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('1 complete prefix groups (10 games)', result.stdout)
        self.assertIn('scheduling 80 games', result.stdout)
        self.assertEqual(self.output_hashes(), before)
        self.assertEqual(len(self.calls()), 1)

    def test_resume_skips_complete_prefix_retains_freeze_and_rebuilds_totals(self):
        self.completed_prefix()
        old = self.summary(); old.update(completed_games=0, results=[])
        write_json(self.output / 'matrix_summary.json', old)
        original = self.output_hashes()
        old_launcher = sha(self.output / 'source_snapshot/sc2_skill_adapter/scripts/run_tactical_matrix.ps1')
        self.script.write_text(self.script.read_text(encoding='utf8') + '\n# corrected offline audit\n', encoding='utf8')
        result = self.launch(resume=True, count=10)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual([row['group'] for row in self.calls()], list(GROUPS))
        self.assertEqual((self.summary()['status'], self.summary()['completed_games']), ('complete', 90))
        for relative, value in original.items():
            if relative != 'matrix_summary.json':
                self.assertEqual(sha(self.output / relative), value, relative)
        path = self.output / self.summary()['resume_audit']
        audit = json.loads(path.read_text(encoding='utf8'))
        self.assertEqual(audit['retained_groups'], ['p_6z4s'])
        self.assertEqual(audit['remaining_groups'], list(GROUPS[1:]))
        self.assertEqual(audit['original_launcher_sha256'], old_launcher)
        self.assertEqual(audit['launcher_sha256'], sha(self.script))
        self.assertEqual(sha(path.parent / 'run_tactical_matrix.ps1'), sha(self.script))
        self.assertEqual(sha(path.parent / 'matrix_summary_before.json'), original['matrix_summary.json'])

    def test_resume_rejects_policy_config_runner_and_dependency_drift_before_any_write(self):
        self.completed_prefix()
        before = self.output_hashes()
        for path in (self.repo / 'skills/tactical_v1/__init__.py',
                     self.repo / 'skills/tactical_v1/defaults.json', self.adapter / 'run.py',
                     self.native / 'wrapper.py'):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(original + b'\n ')
                try:
                    result = self.launch(resume=True, dry=True, count=10)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertTrue('drift' in result.stderr or 'Native dependency differs' in result.stderr, result.stderr)
                    self.assertEqual(self.output_hashes(), before)
                    self.assertEqual(len(self.calls()), 1)
                finally:
                    path.write_bytes(original)

    def test_resume_rejects_partial_and_never_overwrites_incomplete_group(self):
        self.set_runner('partial_first')
        result = self.launch(count=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.summary()['completed_games'], 9)
        before = self.output_hashes()
        result = self.launch(resume=True, count=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('refuses partial group', result.stderr)
        self.assertEqual(self.output_hashes(), before)
        self.assertEqual(len(self.calls()), 1)

    def test_resume_rejects_hole_or_pending_group_artifacts(self):
        self.completed_prefix()
        extra = self.output / ('current/' + GROUPS[2])
        extra.mkdir()
        result = self.launch(resume=True, dry=True, count=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('contiguous complete prefix', result.stderr)
        extra.rmdir()
        pending = self.output / ('manifests/' + GROUPS[1] + '.json')
        pending.write_text('incomplete launch evidence')
        before = self.output_hashes()
        result = self.launch(resume=True, count=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('partial/unstarted group artifacts', result.stderr)
        self.assertEqual(self.output_hashes(), before)
        self.assertEqual(len(self.calls()), 1)

    def test_resume_rejects_count_change_or_original_snapshot_tampering(self):
        self.completed_prefix()
        result = self.launch(resume=True, dry=True, count=20)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Resume protocol', result.stderr)
        original = self.output / 'source_snapshot/skills/tactical_v1/__init__.py'
        original.write_text('# original evidence was altered')
        before = self.output_hashes()
        result = self.launch(resume=True, dry=True, count=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('original snapshot changed', result.stderr)
        self.assertEqual(self.output_hashes(), before)
        self.assertEqual(len(self.calls()), 1)


if __name__ == '__main__':
    unittest.main()
