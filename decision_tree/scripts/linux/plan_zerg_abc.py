"""Freeze a train-only, outcome-blind, single-step A/B/C development screen."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from decision_tree.scripts.linux.fit_zerg_shadow import change_count
from decision_tree.scripts.linux.plan_zerg_forks import arm
from decision_tree.scripts.linux.zerg_fork_preflight import excluded_groups, verify_archived_arm
from decision_tree.zerg.schema import build_model_features, digest
from decision_tree.zerg.scorer import ScorerModel


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def native_prefix(work):
    """One cheap native BASE replay per visited episode; strict Fork replay still certifies the finalists."""
    from decision_tree.zerg.factory import build_zerg_skill_manager
    manager = build_zerg_skill_manager().manager
    manager.reset(); manager.micro_all_attackers = True
    with (work/'runner/events.jsonl').open() as stream:
        for line in stream:
            e = json.loads(line)
            if e.get('episode') != 1: continue
            if e['event']=='skill_observation':
                if not manager.active: manager.select('adaptive_focus_fire',posture='balanced')
                actions, summary = manager.step(e['payload'],10)
                yield e['step']+1,manager,summary,e['payload']
                manager.commit_final_actions(actions)
            elif e['event']=='skill_step' and summary['actions'] != e['actions']:
                raise ValueError(f'BASE replay diverged at {work}:{e["step"]}')
            elif e['event']=='step':
                for sub in e.get('feedback',{}).get('micro_submissions',[]):
                    if sub.get('sc2_response')==[1]: manager.record_micro_action(sub['actor'],sub['submitted_action_id'])


def scan(fit, previous, seed, budget, *, shard=None):
    from decision_tree.zerg.rules_v1 import RULE_ID, make_entry, may_trigger, rule_hash
    split, model, report = (read(fit/name) for name in ('split.json','model.json','report.json'))
    if (digest(model) != report['model_sha256'] or model['target'] != 'imitation' or
            {k for k,v in split.items() if v['partition']=='validation'} != set(model['training_configuration']['validation_episodes'])):
        raise ValueError('frozen fit and split disagree')
    scorer = ScorerModel(model)
    plan = dict(kind='zerg_fork_review_plan_v1', abc_protocol='zerg_abc_v1', fit=str(fit.resolve()), seed=seed,
                horizon_steps=1, split_hash=digest(split), model_id=model['model_id'], source_hash=model['source_fingerprint'],
                collection_context=report['offline_feature_context']['collection_context'], rule_id=RULE_ID, rule_hash=rule_hash())
    card = read(ROOT/'decision_tree/zerg/configs/dual_lane_trial_v1.json')
    if (card['rule_id'],card['parent_context'],card['parent_source']) != (RULE_ID,plan['collection_context'],plan['source_hash']):
        raise ValueError('rule card not frozen for the selected parent')
    plan.update(rule_card=card,rule_card_hash=digest(card))
    avoid, plan['excluded_plan_hashes'] = excluded_groups(previous, plan, split)
    eligible = [g for g, ep in split.items() if ep['partition']=='train' and g not in avoid]
    plan['selection'] = dict(episode_order='sha256(seed, episode_group); earliest eligible point per episode',
         B='first archived legal ranked alternative with distinct complete actions; no fallback by outcome',
         budget=budget, eligible_games=len(eligible), excluded_games=len(avoid))
    plan['screening'] = dict(min_complete=budget, mean_delta_C_A_must_exceed=0, mean_delta_C_B_must_exceed=0,
        net_win_flips_C_A_minimum=0, max_C_A_drops_at_least_5=1,
        next_if_pass='independent whole-game evaluation; never deploy from this screen')
    plan['whole_game_design_if_screen_passes'] = dict(status='design_only_not_run', intervention='candidate addition, not automatic override',
        frozen_selector_required=True, fresh_paired_scenarios=120, allocation='10 per each of the original 12 rosters',
        primary='mean paired full-episode return gain > 0', win_constraint='nonnegative net wins',
        severe_constraint='return delta <= -5 in at most 6 of 120', count_nontriggers=True,
        confirmation='repeat on a separate untouched 120 paired scenes before adoption; no retuning between runs')
    pairs, counts, failures = [], Counter(), Counter()
    ordered = sorted(eligible, key=lambda g:digest([seed,g]))
    if shard is None:
        # Ordered blocks keep publication deterministic regardless of worker speed.
        from concurrent.futures import ProcessPoolExecutor
        import os
        with ProcessPoolExecutor(max_workers=min(12,os.cpu_count() or 1)) as pool:
            tasks = [(fit,previous,seed,budget,k) for k in range(0,len(ordered),4)]
            for chunk in pool.map(_block,tasks):
                for pair, coverage, rejected in chunk:
                    counts.update(coverage); failures.update(rejected)
                    if pair is not None: pairs.append(pair)
                    if len(pairs)==budget: break
                if len(pairs)==budget: break
        ordered = []
    else:
        ordered = ordered[shard:shard+4]; chunks=[]
    for group in ordered:
        counts,failures = Counter(),Counter(); found = None
        ep = split[group]; work = ROOT/ep['batch']/ep['worker']; counts['inspected_games'] += 1
        with (work/'shadow/shadow.jsonl').open() as f:
            records = {r['snapshot_id']:r for r in map(json.loads,f)}
        from contextlib import closing
        with (work/'runner/events.jsonl').open() as trace, closing(native_prefix(work)) as native:
            for line in trace:
                event = json.loads(line)
                if event['event'] != 'skill_observation': continue
                sid = event['step']+1; r = records[sid]; obs = event['payload']; a = r['final_actions']
                if (r['game_loop'] != obs['state']['game_loop'] or f"{r['run_id']}:{r['episode_id']}" != group or
                        r['execution_context_hash'] != plan['collection_context'] or r['source_hash'] != plan['source_hash']):
                    raise ValueError(f'archive context changed: {group}:{sid}')
                counts['inspected_points'] += 1
                if not may_trigger(obs, a): continue
                counts['observable_trigger_points'] += 1
                legal = [c for c in r['candidates'] if c['candidate_family']=='joint_plan' and c['legal']]
                base = next((c for c in legal if c['candidate_id']==r['teacher_choice'] and c['proposed_actions']==a),None)
                alt = next((c for c in legal if c['proposed_actions'].keys()==a.keys() and c['proposed_actions']!=a),None)
                if base is None or alt is None:
                    counts['no_distinct_retained_B'] += 1; continue
                for index, manager, summary, payload in native:
                    if index == sid: break
                else: raise ValueError(f'truncated prefix: {work}:{sid}')
                if summary['actions'] != a: raise ValueError(f'BASE does not match at {work}:{sid}')
                if make_entry(manager,summary,payload) is None:
                    counts['C_abstained_or_duplicate'] += 1; continue
                def pack(c):
                    features = build_model_features(r['global_features'], {'candidate_action_change_count':change_count(c)})
                    return arm(c,scorer.estimate(features)[0]['value'])
                pair = dict(episode_group=group, seed=ep['seed'], roster=ep['roster'], manifest=str(work/'manifest.json'),
                    run_id=r['run_id'],episode_id=r['episode_id'],snapshot_id=sid,game_loop=r['game_loop'],A=pack(base),B=pack(alt),
                    changed={u:dict(unit=r['global_features']['unit_types'].get(u),A=v,B=alt['proposed_actions'][u])
                             for u,v in a.items() if v!=alt['proposed_actions'][u]})
                try:
                    proposal = verify_archived_arm(pair,plan,'A',peek=True)
                    if proposal is None:
                        counts['C_abstained_or_duplicate'] += 1; continue
                    pair['C'] = proposal
                    for mode in ('B','C'): verify_archived_arm(pair,plan,mode)
                except ValueError as exc:
                    if not str(exc).startswith('Zerg fork:'): raise
                    failures[str(exc)] += 1; continue
                found=pair; break
        chunks.append((found,counts,failures))
    if shard is not None: return chunks
    plan.update(pairs=pairs, screen_eligible=(len(pairs)==budget),
                stage='development_screen' if len(pairs)==budget else 'feasibility_only_insufficient_coverage',
                coverage=dict(**counts,replay_rejections=dict(failures),archive_complete_anchors=len(pairs)))
    return plan


def _block(args):
    fit,previous,seed,budget,shard=args
    return scan(fit,previous,seed,budget,shard=shard)


def discovery_smoke(original):
    """Known training anchor, solely to test mechanical engine execution, never independent evidence."""
    from decision_tree.zerg.rules_v1 import RULE_ID,rule_hash
    prior=read(original)
    facts=read(ROOT/'runs/zerg_workflow/round-001/review/llm_train_facts.json')
    fact, = [f for f in facts['pairs'] if f['pair_id']=='pair-023' and f['partition']=='train']
    p, = [p for p in prior['pairs'] if p['episode_group']==fact['episode_group'] and p['snapshot_id']==fact['snapshot_id']]
    card=read(ROOT/'decision_tree/zerg/configs/dual_lane_trial_v1.json')
    plan={k:prior[k] for k in ('kind','fit','seed','model_id','split_hash','source_hash','collection_context','horizon_steps')}
    plan.update(rule_id=RULE_ID,rule_hash=rule_hash(),rule_card=card,rule_card_hash=digest(card),abc_protocol='zerg_abc_v1',
                stage='discovery_anchor_engine_smoke_only',screen_eligible=False,pairs=[p])
    p['C']=verify_archived_arm(p,plan,'A',peek=True)
    if p['C'] is None: raise ValueError('C abstained on the discovery anchor')
    for arm_id in ('B','C'): verify_archived_arm(p,plan,arm_id)
    return plan


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--fit',type=Path,default=ROOT/'runs/zerg_joint_imitation_480_20261123_194909')
    cli.add_argument('--exclude-plan',type=Path,action='append',default=[])
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--discovery-smoke',action='store_true',help='one discovery anchor, not independent evidence')
    args = cli.parse_args()
    original = ROOT/'runs/zerg_forks_24x2_20261123_204135/plan.json'
    excluded = [original, *args.exclude_plan]
    if args.out.exists(): cli.error('--out must be a new file')
    if args.discovery_smoke:
        plan=discovery_smoke(original); write(args.out,plan); print('Discovery smoke only: 1 anchor, 3 planned games.'); sys.exit(0)
    plan = scan(args.fit,excluded,2471,24)
    if not plan['pairs']:
        write(args.out.with_name('coverage_only.json'),plan)
        print(json.dumps(plan['coverage'],indent=2)); cli.exit(1,'No executable anchors; saved coverage_only.json, not a runnable plan.\n')
    if not plan['screen_eligible']:
        print('Fewer than 24 eligible anchors; this is a feasibility-only test, not a passing screen.')
    write(args.out,plan); print(json.dumps(plan['coverage'],indent=2)); print(args.out)
