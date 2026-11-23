"""Evaluate one frozen policy per race on six distinct 10v10 compositions."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from compare_protoss_parallel import (ADAPTER, ROOT, digest, generate_manifest,
                                      load_config, validate_manifest, write_json)

RUNS = ROOT / 'runs'
OLD = RUNS / 'contact_540_20261120_225957_302289'
POLICIES = {'protoss': (OLD / 'nov16', 'colossus'),
            'terran': (OLD / 'nov17', 'medivac'),
            'zerg': (RUNS / 'hydra_supported_20261122_100630_476962/both', 'baneling')}


def prepare(batch, seed, env):
    rng = random.Random(seed)
    (batch / 'manifests').mkdir()
    jobs, design = [], {}
    for i, (race, (work, special)) in enumerate(POLICIES.items()):
        subprocess.run([sys.executable, '-B', '-c', 'from bridge.environment import ReplayableEnv; '
                        'from bridge.skill_runtime import load_manager; load_manager("tactical_v1")()'],
                       cwd=work/'sc2_skill_adapter', env=env, check=True)
        config = load_config(10, race)
        config['unit_balance'] = 'native'
        dist = config['capability_config']['team_gen']
        names, weights = dist['unit_types'], dist['weights']
        seen = set()
        for g in range(6):
            for _ in range(10000):
                team = rng.choices(names, weights=weights, k=10)
                key = tuple(team.count(u) for u in names)
                if key in seen or all(u in dist.get('exception_unit_types', ()) for u in team):
                    continue
                if g < 3 and team.count(special) != g:
                    continue
                break
            else:
                raise ValueError(f'Cannot sample composition for {race}/{g}')
            seen.add(key)
            name = race + '_' + '_'.join(f'{n}{u}' for u, n in zip(names, key))
            design[name] = dict(race=race, policy=str(work), composition=dict(zip(names,key)),
                                special_quota=g if g < 3 else None)
            m = generate_manifest(config, 20, seed + (6*i+g)*20)
            for row in m['episodes']:
                row['episode_config']['team_gen'].update(ally_team=team.copy(), enemy_team=team.copy())
                row['sha256'] = digest(row['episode_config'])
            for offset in range(0, 20, 5):
                part = {k:v for k,v in m.items() if k != 'sha256'}
                part['episodes'] = m['episodes'][offset:offset+5]
                part['sha256'] = digest(part)
                validate_manifest(part)
                path = batch/'manifests'/f'{name}_{offset:02d}.json'
                write_json(path,part)
                jobs.append((race,name,path))
    write_json(batch/'design.json', dict(seed=seed,groups=design))
    return jobs


def play(batch, job, env):
    race, name, path = job
    out = batch/path.stem
    opts = ['--execution-mode','micro8-hydra'] if race == 'zerg' else []
    with out.with_suffix('.log').open('w') as log:
        subprocess.run([sys.executable,'-B','-u','sc2_skill_adapter/run.py','run','--method',
                        'adaptive-skill','--skill-version','tactical_v1',*opts,'--manifest',str(path),
                        '--output-dir',str(out)], cwd=POLICIES[race][0], env=env, stdout=log,
                       stderr=subprocess.STDOUT,check=True)
    rows=json.loads((out/'episodes.json').read_text())
    if [r['scenario_sha256'] for r in rows] != [r['sha256'] for r in json.loads(path.read_text())['episodes']]:
        raise ValueError(f'Missing or wrong scenarios: {out}')
    return Counter(r['outcome'] for r in rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=20261122)
    p.add_argument('--workers',type=int,default=60,choices=range(1,61),metavar='1..60')
    p.add_argument('--output-root',type=Path,default=RUNS)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
             MKL_NUM_THREADS='1',SMAC_DAMAGE_MAP_DIR=str(ADAPTER/'damage_maps'))
    env.pop('PYTHONPATH',None)
    if a.dry_run:
        with TemporaryDirectory() as tmp:
            jobs=prepare(Path(tmp),a.seed,env)
            print(f'Validated {len(jobs)} shards x 5 games; no SC2 launched.')
        return
    batch=a.output_root.resolve()/('best_races_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    batch.mkdir(parents=True)
    print(f'Output: {batch}; seed: {a.seed}',flush=True)
    jobs=prepare(batch,a.seed,env)
    counts,errors={},[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(play,batch,j,env):j for j in jobs}
        for i,f in enumerate(as_completed(futures),1):
            race,name,path=futures[f]
            try:
                outcomes=f.result()
                for key in (name,race,'total'):
                    counts.setdefault(key,Counter()).update(outcomes)
                print(f'[{i}/{len(jobs)}] {path.stem} complete',flush=True)
            except Exception as exc:
                errors.append(path.stem)
                print(f'FAILED {path.stem}: {exc}',flush=True)
    if errors:
        raise SystemExit(f'{len(errors)} failed shards; no final win rate.')
    report={k:dict(games=sum(n.values()),wins=n['win'],win_rate=n['win']/sum(n.values()),outcomes=n) for k,n in counts.items()}
    write_json(batch/'comparison.json',report)
    for k in (*POLICIES,'total'):
        print(f"{k}: {report[k]['wins']}/{report[k]['games']} ({report[k]['win_rate']:.1%})")


if __name__=='__main__':
    main()
