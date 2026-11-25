"""Replay both one-step arms of a Zerg fork plan using the original scenes.

Use the RSI Python from collect_zerg_shadow. --out must be a new directory;
--dry-run validates provenance and writes the design without starting a runner.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from decision_tree.scripts.linux.collect_zerg_shadow import (
    ADAPTER, MAP_DIR, RSI_PYTHON, historical_config, read_json, write_json)
from decision_tree.scripts.linux.plan_zerg_forks import arm as planned_arm
from decision_tree.scripts.linux.fit_zerg_shadow import change_count
from decision_tree.zerg.schema import build_model_features, digest
from decision_tree.zerg.scorer import ScorerModel

SC2_PATH = '/home/amax/cz/SC2_RSI/StarCraftII'
PROTOCOL = dict(method='adaptive-skill', skill_version='tactical_v1', execution_mode='micro8-all')
MAP_FILE = MAP_DIR / '10gen_zerg.SC2Map'


def require_fields(record, expected, where):
    for key, value in expected.items():
        # Compare canonical JSON, so e.g. True cannot masquerade as action 1.
        if key not in record or digest(record[key]) != digest(value):
            raise ValueError(f'{where}: unexpected or missing {key}')


def validate_anchor(path, pair, plan, scorer, steps):
    found = False
    count = 0
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            count += 1
            if (row['snapshot_id'], row['game_loop']) != (pair['snapshot_id'], pair['game_loop']):
                continue
            if found:
                raise ValueError(f'duplicate decision in {path}')
            found = True
            require_fields(row, dict(run_id=pair['run_id'], episode_id=pair['episode_id'],
                snapshot_id=pair['snapshot_id'], game_loop=pair['game_loop'], seed=pair['seed'],
                scene_config_id=pair['roster'], source_hash=plan['source_hash'],
                execution_context_hash=plan['collection_context'],
                teacher_choice=pair['A']['candidate_id'], final_actions=pair['A']['actions']), path)
            for arm in ('A', 'B'):
                candidate, = [c for c in row['candidates'] if c['legal'] is True and
                              c['candidate_family'] == 'joint_plan' and
                              c['candidate_id'] == pair[arm]['candidate_id']]
                features = build_model_features(row['global_features'],
                    {'candidate_action_change_count': change_count(candidate)})
                expected = planned_arm(candidate, scorer.estimate(features)[0]['value'])
                require_fields(pair[arm], expected, f'{path} arm {arm}')
            a, b = (pair[k]['actions'] for k in ('A', 'B'))
            if a.keys() != b.keys() or a == b:
                raise ValueError(f'not two distinct actions on the same actors: {path}')
            delta = {u: dict(unit=row['global_features']['unit_types'].get(u), A=a[u], B=b[u])
                     for u in a if a[u] != b[u]}
            require_fields(pair, dict(changed=delta), path)
    if not found or count != steps:
        raise ValueError(f'missing decision or incomplete original episode: {path}')


def validate_plan(plan, config, versions):
    from bridge.scenarios import validate_manifest

    require_fields(plan, dict(kind='zerg_fork_review_plan_v1', horizon_steps=1), 'plan')
    if not isinstance(plan['pairs'], list) or not plan['pairs'] or type(plan['seed']) is not int:
        raise ValueError('expected a nonempty fork plan and integer selection seed')
    fit = ROOT / plan['fit']
    split, model, report = (read_json(fit / name) for name in ('split.json', 'model.json', 'report.json'))
    ctx = dict(projection='joint_team_skill_int_action_changes_v1', collection_context=plan['collection_context'])
    require_fields(plan, dict(split_hash=digest(split), model_id=model['model_id']), fit)
    require_fields(model, dict(target='imitation', scorer='joint_scorer', execution_context_hash=digest(ctx),
                               source_fingerprint=plan['source_hash']), fit)
    require_fields(report, dict(offline_feature_context=ctx, model_sha256=digest(model), model_id=model['model_id']), fit)
    training = {g for g, ep in split.items() if ep['partition'] == 'train'}
    held = sorted(g for g, ep in split.items() if ep['partition'] == 'validation')
    if (training != {r['group_id'] for r in report['training']['per_scenario']} or
            len(training) + len(held) != len(split)):
        raise ValueError(f'{fit}: original training episodes disagree with split')
    require_fields(model['training_configuration'], dict(validation_episodes=held), fit)
    scorer = ScorerModel(model)
    seen_groups, seen_scenarios = set(), set()
    for i, pair in enumerate(plan['pairs'], 1):
        group = pair['episode_group']
        if group != f"{pair['run_id']}:{pair['episode_id']}" or group in seen_groups:
            raise ValueError(f'pair {i}: invalid or repeated episode group')
        seen_groups.add(group)
        ep = split[group]
        require_fields(ep, dict(partition='train', roster=pair['roster'], seed=pair['seed']), group)
        batch = ROOT / ep['batch']
        work = batch / ep['worker']
        path = work / 'manifest.json'
        if Path(pair['manifest']) != path or not path.is_absolute():
            raise ValueError(f'pair {i}: not the original split manifest: {path}')
        design = read_json(batch / 'design.json')
        require_fields(design, dict(dry_run=False, versions=versions, python=RSI_PYTHON,
            execution_mode='micro8-all', damage_map_dir=str(MAP_DIR), sc2_path=SC2_PATH), batch)
        job, = [j for j in design['jobs'] if j['worker'] == ep['worker']]
        manifest = read_json(path)
        validate_manifest(manifest)
        scenario, = manifest['episodes']
        require_fields(manifest, dict(seed=pair['seed'], env_args=config), path)
        require_fields(scenario, dict(sc2_seed=pair['seed'], index=0), path)
        require_fields(job, dict(sc2_seed=pair['seed'], roster=pair['roster'],
            manifest_sha256=manifest['sha256'], scenario_sha256=scenario['sha256']), batch)
        team = design['rosters'][pair['roster']]
        require_fields(scenario['episode_config']['team_gen'], dict(ally_team=team, enemy_team=team), path)
        if scenario['sha256'] in seen_scenarios:
            raise ValueError(f'pair {i}: repeated scenario')
        seen_scenarios.add(scenario['sha256'])
        original_run = read_json(work / 'runner/run.json')
        require_fields(original_run, dict(**PROTOCOL, versions=versions, env_args=config,
            map_file=str(MAP_FILE), manifest_sha256=manifest['sha256'], damage_by_type_enabled=True,
            macro_loops=8, engine_step_loops=1, native_teacher_control='off'), work)
        summary = read_json(work / 'runner/summary.json')
        require_fields(summary, {**PROTOCOL, 'status': 'complete', 'manifest_sha256': manifest['sha256']}, work)
        require_fields(summary['zerg_shadow'], dict(run_id=pair['run_id'], scene_config_id=pair['roster'],
            source_fingerprint=plan['source_hash'], execution_context_hash=plan['collection_context']), work)
        episode, = read_json(work / 'runner/episodes.json')
        require_fields(episode, dict(scenario_sha256=scenario['sha256'], zerg_shadow_decisions=episode['steps']), work)
        validate_anchor(work / 'shadow/shadow.jsonl', pair, plan, scorer, episode['steps'])


def validate_arm(arm_dir, arm, pair, plan):
    """Return the completed receipt, or raise; an unknown engine receipt is not acceptance."""
    runner = Path(arm_dir) / 'runner'
    path = Path(pair['manifest'])
    original_run = read_json(path.parent / 'runner/run.json')
    run = read_json(runner / 'run.json')
    require_fields(run, {k: original_run[k] for k in (*PROTOCOL, 'macro_loops', 'engine_step_loops',
        'map_file', 'damage_by_type_enabled', 'skill_parameters', 'native_teacher_control',
        'versions', 'env_args', 'skill_source_hashes', 'manifest_sha256')}, runner)
    map_key = MAP_FILE.relative_to(ROOT).as_posix()
    require_fields(run['source_hashes'], {map_key: original_run['source_hashes'][map_key]}, runner)
    original, = read_json(path.parent / 'runner/episodes.json')
    episode, = read_json(runner / 'episodes.json')
    summary = read_json(runner / 'summary.json')
    require_fields(summary, {**PROTOCOL, 'status': 'complete', 'completed_episodes': 1,
        'expected_episodes': 1, 'manifest_sha256': read_json(path)['sha256']}, runner)
    meta = summary['zerg_shadow']
    require_fields(meta, dict(source_fingerprint=plan['source_hash'], scene_config_id=pair['roster'],
                              execution_context_hash=plan['collection_context']), runner)
    identity = {k: original[k] for k in ('scenario_sha256', 'initial_state_sha256')}
    steps = episode['steps']
    require_fields(episode, dict(**identity, zerg_shadow_decisions=steps), runner)
    require_fields(meta['stats'], dict(decisions_seen=steps, decisions_recorded=steps, model_errors=0), runner)
    keys = set()
    expected_row = dict(run_id=meta['run_id'], episode_id=pair['episode_id'], scene_config_id=pair['roster'],
        seed=pair['seed'], source_hash=plan['source_hash'], execution_context_hash=plan['collection_context'],
        outcome=episode['outcome'], **{'return': episode['reward']})
    with (runner.parent / 'shadow/shadow.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            require_fields(row, expected_row, runner)
            key = (row['snapshot_id'], row['game_loop'])
            if key in keys:
                raise ValueError(f'{runner}: repeated shadow record {key}')
            keys.add(key)
    if type(steps) is not int or steps != len(keys) or (pair['snapshot_id'], pair['game_loop']) not in keys:
        raise ValueError(f'{runner}: incomplete shadow episode or missing anchor')
    receipt = read_json(runner / 'fork_receipt.json')
    expected = {k: pair[k] for k in ('episode_group', 'seed', 'roster', 'snapshot_id', 'game_loop')}
    require_fields(receipt, dict(**expected, **identity, kind='zerg_one_shot_fork_receipt_v1',
        status='complete', arm=arm, prefix_verified=True, anchor_commits=1, error=None,
        source_hash=plan['source_hash'], collection_context=plan['collection_context'],
        candidate_id=pair[arm]['candidate_id'], source_plan_id=pair[arm]['source_plan_id'],
        final_actions=pair[arm]['actions']), runner)
    if receipt['engine_receipt'] not in ('accepted', 'unknown'):
        raise ValueError(f'{runner}: rejected or invalid engine receipt')
    return receipt


def run_worker(arm_dir, pair, spec_path):
    """Run a prepared fork spec using the historical protocol."""
    env = dict(os.environ, SC2PATH=SC2_PATH, SMAC_DAMAGE_MAP_DIR=str(MAP_DIR), OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
               ZERG_FORK_SPEC=str(spec_path.resolve()), ZERG_SHADOW_ROSTER=pair['roster'],
               ZERG_SHADOW_DIR=str(arm_dir / 'shadow'))
    env.pop('PYTHONPATH', None)
    for key in ('NO_PROXY', 'no_proxy'):
        env[key] = ','.join(filter(None, (env.get(key), '127.0.0.1,localhost,::1')))
    cmd = [RSI_PYTHON, '-B', '-u', str(ADAPTER / 'run.py'), 'run', '--method', PROTOCOL['method'],
           '--skill-version', PROTOCOL['skill_version'], '--execution-mode', PROTOCOL['execution_mode'],
           '--manifest', pair['manifest'], '--output-dir', str(arm_dir / 'runner')]
    with (arm_dir / 'worker.log').open('x', encoding='utf-8') as log:
        subprocess.run(cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def play(arm_dir, arm, pair, plan):
    spec = dict(arm=arm, pair=pair, source_hash=plan['source_hash'], collection_context=plan['collection_context'])
    spec_path = arm_dir / 'fork_spec.json'
    write_json(spec_path, spec)
    run_worker(arm_dir, pair, spec_path)
    validate_arm(arm_dir, arm, pair, plan)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--plan', type=Path, required=True)
    cli.add_argument('--out', type=Path, required=True, help='new directory (never overwritten)')
    cli.add_argument('--workers', type=int, default=48, help='maximum concurrent games (default: 48)')
    cli.add_argument('--pair-index', type=int, help='run both arms of just this original 1-based pair')
    cli.add_argument('--dry-run', action='store_true')
    args = cli.parse_args(argv)
    out = args.out.expanduser().resolve()
    if out.exists() or args.workers <= 0:
        cli.error('require a new --out directory and positive --workers')
    try:
        raw_plan = args.plan.read_bytes()
        plan = json.loads(raw_plan)
        if 'abc_protocol' in plan:
            cli.error('ABC plans require run_zerg_abc.py; refusing to drop arm C')
        n = len(plan['pairs'])
        if args.pair_index is not None and not 1 <= args.pair_index <= n:
            cli.error(f'--pair-index must be in 1..{n}')
        config, versions = historical_config()
        validate_plan(plan, config, versions)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        cli.error(f'preflight failed: {exc}')

    indices = [args.pair_index] if args.pair_index is not None else list(range(1, n + 1))
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'plan.json').open('xb') as f:
        f.write(raw_plan)
    write_json(out / 'design.json', dict(pair_indices=indices, dry_run=args.dry_run, workers=args.workers,
        plan_hash=digest(plan), python=RSI_PYTHON, versions=versions, sc2_path=SC2_PATH,
        damage_map_dir=str(MAP_DIR), **PROTOCOL))
    jobs = []
    for i in indices:
        for arm in ('A', 'B'):
            work = out / f'pair-{i:03d}' / arm
            work.mkdir(parents=True)
            jobs.append((work, arm, plan['pairs'][i-1], plan))
    print(f'{len(indices)} pair(s), {len(jobs)} independent arms; design: {out / "design.json"}', flush=True)
    if args.dry_run:
        print('Dry-run: no SC2 processes launched.')
        return
    failures = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        pending = {pool.submit(play, *job): job[0] for job in jobs}
        for future in as_completed(pending):
            try:
                future.result()
            except Exception as exc:
                failures.append(pending[future])
                print(f'FAILED {pending[future]}: {exc}', file=sys.stderr, flush=True)
    print('Compare separately:', shlex.join([
        RSI_PYTHON, '-B', str(Path(__file__).with_name('compare_zerg_forks.py')), '--run', str(out)]))
    if failures:
        raise SystemExit(f'{len(failures)}/{len(jobs)} arms failed; see worker.log in the paths above')


if __name__ == '__main__':
    main()
