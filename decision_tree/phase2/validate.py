"""Roll out frozen phase-2 candidates on the ten predeclared validation scenes."""
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import subprocess
import time

from decision_tree.local import study
from decision_tree.local.dataset import read_run, episode, audit_pair, comparable_observation, require
from decision_tree.local.runtime import digest
from decision_tree.scripts.local_campaign_recover import terminate_owned, failure_kind
from decision_tree.scripts.local_campaign_validate import validate_cohort, validation_lock, summarize
from .evidence import file_sha
from .policy import normalize_policy, accepted_attack, Selector
from .run import source_hashes, carrier_policy


def launch(cmd, dest, idle_seconds=120):
    study.save_once(dest.with_suffix('.command.json'), cmd)
    with dest.with_suffix('.console.log').open('x', encoding='utf-8') as stream:
        proc = subprocess.Popen(cmd, cwd=study.ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONUNBUFFERED':'1'},
                                **({'creationflags':subprocess.CREATE_NO_WINDOW} if os.name == 'nt'
                                   else {'start_new_session':True}))
        owner = {'pid':proc.pid, 'command_sha256':digest(cmd)}
        study.save_once(dest.with_suffix('.process.json'), owner)
        last, changed = None, time.monotonic()
        log = dest/'events.jsonl'
        while True:
            try:
                return proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                marker = log.stat().st_size if log.exists() else None
                if marker != last:
                    last, changed = marker, time.monotonic()
                elif time.monotonic()-changed > idle_seconds:
                    terminate_owned(proc, cmd, owner)
                    dest.mkdir(exist_ok=True)
                    study.save_once(dest/'timeout.json', {**owner,'owned_process':True})
                    return -1


def completed_arm(output, arm, policy, manifest, execution):
    for n in range(1, 4):
        dest = output/f'{arm}-{n:02d}'
        if not dest.exists():
            require(not any(p.exists() for p in [dest.with_suffix('.console.log'), dest.with_suffix('.process.json')]),
                    'Unresolved launch already owns '+str(dest))
            cmd = [execution['python'], '-B', '-m', 'decision_tree.phase2.run',
                   '--policy', str(output/(arm+'.json')), '--manifest', str(output/'manifest.json'),
                   '--output-dir', str(dest)]
            if execution.get('sc2_path'):
                cmd.extend(['--sc2-path',execution['sc2_path']])
            launch(cmd, dest)
        status = study.read(dest/'summary.json') if (dest/'summary.json').exists() else {}
        if status.get('status') != 'complete':
            reason = failure_kind(dest)
            require(reason in ('environment_transport','watchdog_timeout'), 'Incomplete validation arm: '+str(dest)+' '+reason)
            continue
        run = read_run(dest)
        binding = run['run.json']['phase2']
        require(run['manifest.json']==manifest and binding['policy']==policy and
                binding['policy_sha256']==digest(policy) and binding['source_hashes']==source_hashes() and
                binding==run['summary.json']['phase2'] and run['run.json']['local_policy']['policy']==carrier_policy(policy),
                'Unbound validation policy')
        return run
    raise RuntimeError('Three failed environmental attempts: '+arm)


def base_row(arm):
    rec = arm['record']
    return {'return':rec['reward'],'outcome':rec['outcome'],'steps':rec['steps'],
            'wait_steps':0,'decline':None,'execution':None,'matched_prefix':True}


def validate_episode(base, arm, policy):
    selector = Selector(policy)
    decisions = sorted((e for e in arm['events'] if e['event']=='local_decision'), key=lambda e:e['step'])
    fired = False
    row = base_row(arm)
    for e in decisions:
        step = e['step']
        expected = selector.decide(e['candidates'], step, fired)
        require(expected==e['phase2'] and e['selected_id']==expected['selected_id'], 'Selector replay mismatch')
        action = expected['action']
        if action == 'wait':
            row['wait_steps'] += 1
        if action in ('execute','decline'):
            item = next(c for c in e['candidates'] if c['id']==expected['considered_id'])
            key = 'execution' if action=='execute' else 'decline'
            row[key] = {'step':step,'id':item['id'],'bucket':expected['bucket'],'features':item['features']}
        fired = fired or accepted_attack(arm['steps'][step+1]['feedback'])
    require(base['start']['initial_state_sha256']==arm['start']['initial_state_sha256'] and
            base['record']['scenario_sha256']==arm['record']['scenario_sha256'] and
            base['start']['game_version']==arm['start']['game_version'], 'Different episode initial state')
    if row['execution']:
        ex = row['execution']
        proof, _, _ = audit_pair(base, arm, ex['step'], ex['id'])
        ex['audit'] = proof
    else:
        require(row['steps']==base['record']['steps'] and row['outcome']==base['record']['outcome'] and
                abs(row['return']-base['record']['reward'])<1e-8,'Unmodified episode differs from BASE')
        for step in range(row['steps']):
            require(comparable_observation(base['observations'][step]['payload'])==
                    comparable_observation(arm['observations'][step]['payload']), 'Unmodified observation changed')
            for field in ('actions','feedback','game_loop','reward','terminated'):
                require(base['steps'][step+1][field]==arm['steps'][step+1][field], 'Unmodified rollout drift: '+field)
    return row


def compare(rows, reference):
    comp = [{**r, 'incumbent':r[reference], 'gain':r['candidate']['return']-r[reference]['return']} for r in rows]
    return {**summarize(comp),
            'rescued':[r['seed'] for r in comp if r['candidate']['outcome']=='win' and r['incumbent']['outcome']!='win'],
            'lost':[r['seed'] for r in comp if r['candidate']['outcome']!='win' and r['incumbent']['outcome']=='win']}


def validate(folder):
    folder = Path(folder).resolve()
    require(folder.is_relative_to(study.ROOT/'decision_tree/artifacts'), 'Require experiment folder')
    proposal = study.read(folder/'proposal.json')
    require(proposal['kind']=='phase2_frozen_proposal' and proposal['phase2_sources']==source_hashes(), 'Changed proposal runtime')
    policies = {arm:normalize_policy(proposal[arm]) for arm in ('incumbent','candidate')}
    manifest, split = study.read(folder/'manifest.json'), study.read(folder/'split.json')
    cohort = validate_cohort(manifest, split)
    holdout = set(split['validation_group_ids'])
    for src in proposal['training_sources']:
        require(file_sha(Path(src['path']))==src['sha256'], 'Training artifact changed')
        groups = {s['group_id'] for s in study.read(src['path'])['scenes']}
        require(not groups & holdout, 'Holdout leaked into proposal')
    execution = study.read(folder/'study/protocol.json')
    require(study.source_hashes(manifest)==execution['sources'], 'Frozen adaptive sources changed')
    reference = read_run(folder/'study/base')
    require(reference['manifest.json']==manifest and reference['run.json']['local_policy']['policy']==study.read(folder/'study/base_policy.json') and
            reference['run.json']['local_policy']['policy']['selection']['mode']=='base', 'Wrong BASE')
    subset = deepcopy(manifest)
    subset['episodes'] = [{**scene,'index':i} for i,(_,scene) in enumerate(cohort)]
    subset['sha256'] = digest({k:v for k,v in subset.items() if k!='sha256'})
    output = folder/'validation'
    output.mkdir(exist_ok=True)
    with validation_lock(output):
        study.save_once(output/'protocol.json',{'proposal_sha256':digest(proposal),'base_hashes':reference['hashes'],
                        'execution_sha256':digest(execution),'split':split,'manifest_sha256':subset['sha256'],
                        'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        study.save_once(output/'manifest.json',subset)
        base_episodes = [episode(reference,n) for n,_ in cohort]
        del reference['events.jsonl']
        rows = [{'seed':s['sc2_seed'],'group_id':s['sha256'],'base':base_row(base)} for (_,s),base in zip(cohort,base_episodes)]
        run_bindings = {}
        for arm,policy in policies.items():
            study.save_once(output/(arm+'.json'),policy)
            if policy['mode']=='base':
                for row in rows: row[arm]=row['base']
                run_bindings[arm]={'reuse':'base'}
            elif arm=='candidate' and policy==policies['incumbent']:
                for row in rows: row[arm]=row['incumbent']
                run_bindings[arm]={'reuse':'incumbent'}
            else:
                run=completed_arm(output,arm,policy,subset,execution)
                require(run['game_version']==reference['game_version'] and
                        run['run.json']['source_hashes']==reference['run.json']['source_hashes'] and
                        run['run.json']['local_policy']['context_hash']==reference['run.json']['local_policy']['context_hash'],
                        'Validation context mismatch')
                for i,(row,base) in enumerate(zip(rows,base_episodes),1):
                    row[arm]=validate_episode(base,episode(run,i),policy)
                run_bindings[arm]={'path':run['path'],'hashes':run['hashes']}
            print('Audited '+arm+' on all ten scenes.', flush=True)
        study.check_sources(execution['sources'])
        research, production = compare(rows,'incumbent'),compare(rows,'base')
        supported = proposal['supported_change']
        require(isinstance(supported,bool), 'Missing declared support decision')
        adopted = supported and research['decision']=='adopt_candidate'
        result={'kind':'phase2_internal_validation','rows':rows,'bindings':run_bindings,'versus_incumbent':research,
                'versus_base':production,'adopt_research_candidate':adopted,
                'beats_adaptive_on_this_batch':supported and len(production['rescued'])>len(production['lost'])
                  and production['mean_gain']>=-1e-9,'proposal_sha256':digest(proposal)}
        study.save_once(output/'result.json',result)
        return {k:v for k,v in result.items() if k not in ('rows','bindings')}


if __name__=='__main__':
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', required=True, type=Path)
    print(json.dumps(validate(parser.parse_args().round),ensure_ascii=False,indent=2))
