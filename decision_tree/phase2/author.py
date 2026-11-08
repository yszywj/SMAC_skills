"""Freeze one human/LLM-specified policy and its measured training support."""
import argparse
import json
from pathlib import Path

from decision_tree.local.study import read, save_once
from decision_tree.local.runtime import digest
from .evidence import compare, file_sha, load_cohorts, stats
from .policy import Selector, normalize_policy
from .run import source_hashes


def first_difference(scene, candidate, incumbent):
    a, b = Selector(candidate), Selector(incumbent)
    for frame in scene['frames']:
        args = frame['candidates'], frame['step'], frame['team_fired_before']
        x, y = a.decide(*args), b.decide(*args)
        if (x['action'],x['considered_id']) != (y['action'],y['considered_id']):
            return {'step':frame['step'],'bucket':x['bucket'] or y['bucket'],
                    'candidate_action':x['action'],'incumbent_action':y['action']}
        if x['consumed']:
            break
    return None


def freeze(folder, training, candidate, incumbent, rationale, support_training=()):
    candidate, incumbent = normalize_policy(candidate), normalize_policy(incumbent)
    scenes = load_cohorts(training)
    # Older batches sometimes did not sample the newly introduced policy.
    # Keep their incomplete comparisons visible; require complete measurement
    # in the cohorts explicitly designated for learning the changed condition.
    support_paths = list(support_training) or training
    if not {p.resolve() for p in support_paths} <= {p.resolve() for p in training}:
        raise ValueError('Support inputs must be a subset of the declared training inputs')
    support_ids = {s['group_id'] for s in scenes} if support_paths==training else {
        s['group_id'] for s in load_cohorts(support_paths)}
    comparison = compare(scenes,candidate,incumbent)
    bucket_rows, affected, missing = {}, [], []
    for scene,row in zip(scenes,comparison['rows']):
        diff = first_difference(scene,candidate,incumbent)
        if diff is None:
            continue
        bucket = diff['bucket']
        in_scope = row['group_id'] in support_ids
        if in_scope:
            bucket_rows.setdefault(bucket, [])
        a, b = row['candidate'], row['incumbent']
        if a['return'] is None or b['return'] is None:
            missing.append({'seed':row['sc2_seed'],'in_support_cohort':in_scope,**diff})
            continue
        if in_scope:
            bucket_rows[bucket].append({'group_id':row['group_id'],'gain':a['return']-b['return']})
        affected.append({'seed':row['sc2_seed'],'group_id':row['group_id'],'gain':a['return']-b['return'],
                         'candidate_outcome':a['outcome'],'incumbent_outcome':b['outcome'],'in_support_cohort':in_scope,**diff})
    support = {k:stats(v) for k,v in bucket_rows.items()}
    # No semantic group can borrow support from a different changed branch.
    supported = bool(support) and not any(m['in_support_cohort'] for m in missing) and all(v['groups']>=4 and v['scene_equal_mean_gain']>1e-9
                                                      for v in support.values())
    proposal={'kind':'phase2_frozen_proposal','author':'conversation LLM','rationale':rationale,
              'candidate':candidate,'incumbent':incumbent,'training_scene_count':len(scenes),
              'training_sources':[{'path':str(p.resolve()),'sha256':file_sha(p)} for p in training],
              'designated_support_cohorts':[str(p.resolve()) for p in support_paths],
              'support_by_changed_bucket':support,'affected':affected,'missing_affected':missing,'supported_change':supported,
              'versus_incumbent':comparison['differences']['incumbent'],'versus_base':comparison['differences']['base'],
              'phase2_sources':source_hashes()}
    save_once(folder/'proposal.json',proposal)
    save_once(folder/'training_comparison.json',comparison)
    return {'proposal_sha256':digest(proposal),'supported':supported,'support':support,
            'missing_affected':len(missing),'training_total':len(scenes)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--round',type=Path,required=True)
    p.add_argument('--training',type=Path,action='append',required=True)
    p.add_argument('--support-training',type=Path,action='append',default=[])
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--incumbent',type=Path,required=True)
    p.add_argument('--rationale',required=True)
    args=p.parse_args()
    print(json.dumps(freeze(args.round,args.training,read(args.candidate),read(args.incumbent),args.rationale,args.support_training),
                     ensure_ascii=False,indent=2))
