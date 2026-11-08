"""Freeze outcome-blind, one-intervention experiments for second-round SMAC sampling.

Run BASE with the frozen adaptive configuration. Replaying predeclared selectors
on its observations identifies the exact steps to force; each tested menu is
complete. Candidate execution and paired-prefix audits use the original
local_campaign_recover and decision_tree.local.dataset machinery.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess

from decision_tree.local import study
from decision_tree.local.dataset import read_run
from decision_tree.local.runtime import digest, load_policy
from decision_tree.phase2 import policy as phase2
from decision_tree.scripts.local_campaign_plan import watch
from decision_tree.scripts.local_campaign_recover import terminate_owned
from decision_tree.scripts.local_campaign_validate import single_manifest


def require(ok, message):
    if not ok:
        raise ValueError(message)


def probe_sources():
    files = {'planner': Path(__file__), 'selector': Path(phase2.__file__),
             'regroup_policy': phase2.POLICY_PATH}
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}


def base(template, directory, idle, supplement):
    """Reuse a completed, correctly bound BASE. Preserve incomplete evidence."""
    output = directory / 'study'
    reference = study.read(template / 'protocol.json')
    manifest = study.read(directory / 'manifest.json')
    original = study.read(template / 'manifest.json')
    policy = load_policy(template / 'base_policy.json')
    require(reference['sources'] == study.source_hashes(manifest)
            and original['env_args'] == manifest['env_args']
            and digest(policy) == reference['policy_sha256'], 'Frozen training context changed')
    require(len(manifest['episodes']) == 40, 'Expected forty BASE scenes')
    protocol = {**reference, 'manifest_sha256': digest(manifest), 'anchors': 4, 'min_gap': 1,
                'anchor_rule': 'union_of_frozen_policy_opportunities', 'reference_policy': None,
                'supplement_empty_scene': supplement}
    output.mkdir(parents=True, exist_ok=True)
    study.save_once(output/'protocol.json', protocol)
    study.save_once(output/'manifest.json', manifest)
    study.save_once(output/'base_policy.json', policy)
    study.snapshot_sources(protocol['sources'], output/'source_snapshot')
    if not (output/'base').exists():
        cmd = study.command(protocol, output/'manifest.json', output/'base_policy.json', output/'base')
        study.save_once(output/'base.command.json', cmd)
        kwargs = {'cwd': study.ROOT, 'env': {**os.environ, 'PYTHONUNBUFFERED': '1'},
                  'stdin': subprocess.DEVNULL, 'stderr': subprocess.STDOUT}
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs['start_new_session'] = True
        with (output/'base.console.log').open('x', encoding='utf-8') as log:
            process = subprocess.Popen(cmd, stdout=log, **kwargs)
            try:
                code = watch(process, cmd, output, idle)
            finally:
                if process.poll() is None:
                    terminate_owned(process, cmd, {'pid': process.pid, 'command_sha256': digest(cmd)})
            require(code == 0, 'BASE failed; original evidence retained')
    data = read_run(output/'base')
    require(data['manifest.json'] == manifest and data['run.json']['local_policy']['policy'] == policy,
            'Cannot reuse BASE from a different experiment')
    return data


def select(steps, receipts, policies, *, supplement=True):
    """Keep the first proposal and first consumed action of each alternative.

    This also tests a declined choice's forced branch. WAIT rows after the first
    are not forced unless another selector consumes at that step. This avoids
    a biased cutoff that would silently drop the later actions of patient policies.
    """
    selected = {}
    for name, policy in policies.items():
        gate = phase2.Selector(policy)
        fired = False
        first = True
        for event in steps:
            k = event['step']
            verdict = gate.decide(event['candidates'], k, fired)
            if verdict['considered_id'] != 'BASE' and (first or gate.consumed):
                selected.setdefault(k, []).append({'policy': name, 'id': verdict['considered_id'],
                                                   'action': verdict['action'], 'bucket': verdict['bucket']})
                first = False
            if gate.consumed:
                break
            # Receipts number the completed environment step from one;
            # local decision indices start at zero.
            fired = fired or phase2.accepted_attack(receipts[k + 1]['feedback'])
    if not selected and supplement:
        for event in study.select_anchors(steps, 1, min_gap=1, rule='family_coverage'):
            selected[event['step']] = [{'source': 'outcome_blind_family_coverage_supplement'}]
    require(len(selected) <= 4, 'More than four anchor steps needed; no candidate budget was truncated')
    return selected


def extend_menu(output, manifest, base_policy, number, event, reasons, plan):
    k = event['step']
    scene = manifest['episodes'][number-1]
    ids = [c['id'] for c in event['candidates']]
    require(ids and len(set(ids)) == len(ids), 'Invalid canonical menu')
    anchor = {'base_episode': number, 'anchor_step': k, 'reference_first_proposal': False,
              'available_candidate_ids': ids, 'phase2_selection': reasons}
    plan['anchors'].append(anchor)
    for rank, candidate in enumerate(event['candidates']):
        name = f'ep{number:04d}_s{k:05d}_c{rank:03d}'
        single = single_manifest(manifest, scene)
        forced = deepcopy(base_policy)
        forced['selection'] = {'mode':'forced', 'candidate_id':candidate['id'], 'anchor_step':k,
                               'max_interventions':1, 'min_gain':0.0}
        forced = load_policy(forced)
        mp = output/'inputs'/(name+'_manifest.json')
        pp = output/'inputs'/(name+'_policy.json')
        study.save_once(mp,single)
        study.save_once(pp,forced)
        plan['arms'].append({'name':name, 'path':str(output/name), 'base_episode':number,
                             'arm_episode':1,'anchor_step':k,'reference_first_proposal':False,
                             'candidate_id':candidate['id'], 'candidate_family':candidate['family'],
                             'manifest':str(mp), 'policy':str(pp), 'manifest_sha256':digest(single),
                             'policy_sha256':digest(forced)})


def plan(directory, cohort, policies, supplement):
    output = directory/'study'
    manifest = cohort['manifest.json']
    base_policy = cohort['run.json']['local_policy']['policy']
    plan = {'schema_version':1, 'kind':'local_patch_study_plan', 'base_path':str(output/'base'),
            'base_hashes':cohort['hashes'], 'reference_policy':None, 'anchor_rule':{
                'name':'union_of_frozen_policy_opportunities','maximum_count':4,
                'minimum_step_gap':1,'policies':policies,'supplement_empty_scene':supplement,
                'source_sha256':hashlib.sha256(Path(phase2.__file__).read_bytes()).hexdigest()},
            'scenes':[], 'anchors':[], 'arms':[]}
    for n, scene in enumerate(manifest['episodes'],1):
        ep=study.episode(cohort,n)
        steps=sorted((e for e in ep['events'] if e['event']=='local_decision'),key=lambda e:e['step'])
        selected = select(steps,ep['steps'],policies,supplement=supplement)
        plan['scenes'].append({'base_episode':n,'group_id':scene['sha256'],'sc2_seed':scene['sc2_seed'],
                               'anchors':len(selected)})
        for event in steps:
            k=event['step']
            if k in selected:
                extend_menu(output,manifest,base_policy,n,event,selected[k],plan)
    plan['candidate_games']=len(plan['arms'])
    plan['total_games']=len(plan['scenes'])+plan['candidate_games']
    study.save_once(output/'plan.json',plan)
    return {'scenes':len(plan['scenes']), 'anchors':len(plan['anchors']), 'candidates':len(plan['arms'])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--round-dir',type=Path,required=True)
    p.add_argument('--template',type=Path,required=True)
    p.add_argument('--policies',type=Path,required=True,help='JSON mapping fixed before BASE to probe on each scene')
    p.add_argument('--idle-seconds',type=int,default=120)
    p.add_argument('--no-supplement',action='store_true',
                   help='Keep scenes with no probe opportunity in the cohort with zero forced arms')
    args=p.parse_args()
    d=args.round_dir.resolve()
    require(d.is_relative_to(study.ROOT/'decision_tree/artifacts/precontact_training_5x40')
            and 30<=args.idle_seconds<=3600,'Invalid second-round root or watchdog')
    policies = {name: phase2.normalize_policy(pol) for name, pol in study.read(args.policies).items()}
    require(bool(policies),'At least one probe policy is required')
    sources = probe_sources()
    study.save_once(d/'sampling_policies.json', policies)
    study.save_once(d/'sampling_protocol.json', {'policies_sha256': digest(policies),
                    'sources': sources, 'supplement_empty_scene': not args.no_supplement,
                    'maximum_anchors': 4, 'minimum_step_gap': 1})
    lock = d/'.phase2_collection.lock'
    with lock.open('x', encoding='utf-8') as handle:
        handle.write(str(os.getpid()))
    try:
        cohort = base(args.template.resolve(), d, args.idle_seconds, not args.no_supplement)
        require(probe_sources() == sources, 'Anchor planner or selector changed during BASE collection')
        print(json.dumps(plan(d, cohort, policies, not args.no_supplement)), flush=True)
    finally:
        lock.unlink()


if __name__=='__main__':
    main()
