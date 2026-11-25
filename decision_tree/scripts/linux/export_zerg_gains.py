"""Export audited single-step joint gain labels from a completed fork run."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from decision_tree.scripts.linux.run_zerg_forks import historical_config, read_json, validate_arm, validate_plan
from decision_tree.zerg.dataset import build_pair_record, dumps_key, record_key
from decision_tree.zerg.experiment_manifest import make_anchor_manifest
from decision_tree.zerg.pair_dataset import build_gain_dataset, make_execution_receipt
from decision_tree.zerg.schema import digest, dumps


def prefix(runner, pair, expected_actions):
    """Hash every observed payload through the anchor, but no post-anchor outcome."""
    trace = []
    with (runner / 'events.jsonl').open() as stream:
        for line in stream:
            e = json.loads(line)
            if e.get('episode') != 1:
                continue
            if e['event'] == 'skill_observation':
                if e['step'] * 2 != len(trace):
                    raise ValueError(f'{runner}: incomplete pre-anchor observations')
                trace.append(digest(e['payload']))
            elif e['event'] == 'step':
                if e['step'] * 2 - 1 != len(trace):
                    raise ValueError(f'{runner}: incomplete pre-anchor actions')
                if e['game_loop'] == pair['game_loop']:
                    if (e['step'] != pair['snapshot_id'] or
                        {f'u{i}': v for i,v in enumerate(e['actions'])} != expected_actions):
                        raise ValueError(f'{runner}: wrong executed anchor')
                    return digest(trace)
                trace.append([e['game_loop'], e['actions']])
    raise ValueError(f'{runner}: anchor not reached')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run', type=Path, required=True)
    cli.add_argument('--out', type=Path, required=True, help='new directory')
    args = cli.parse_args()
    if args.out.exists():
        cli.error('--out must be a new directory')
    run = args.run.resolve()
    plan, design = (read_json(run / fn) for fn in ('plan.json', 'design.json'))
    if 'abc_protocol' in plan or any('C' in p for p in plan['pairs']):
        raise ValueError('A/B/C evidence cannot be exported with the two-arm gain schema')
    indices = design['pair_indices']
    if (design['dry_run'] or design['plan_hash'] != digest(plan) or not indices or
        indices != sorted(set(indices)) or any(not 1 <= i <= len(plan['pairs']) for i in indices)):
        raise ValueError('invalid fork run design')
    validate_plan(plan, *historical_config())
    experiment = run.name
    records, pairs, receipts, anchors = [], [], [], {}
    for i in indices:
        p = plan['pairs'][i-1]
        orig = Path(p['manifest']).parent
        with (orig/'shadow/shadow.jsonl').open() as stream:
            rec, = (r for r in map(json.loads, stream) if r['snapshot_id'] == p['snapshot_id'] and r['game_loop'] == p['game_loop'])
        if any(c.get('candidate_return') is not None for c in rec['candidates']):
            raise ValueError('anchor already carries training labels')
        records.append(rec)
        key = dumps_key(record_key(rec))
        original_ep, = read_json(orig/'runner/episodes.json')
        anchors[key] = dict(baseline_id=p['A']['candidate_id'], initial_state_hash=original_ep['initial_state_sha256'],
                            prefix_hash=prefix(orig/'runner', p, p['A']['actions']))
        common = dict(experiment_id=experiment, anchor_id=key, baseline_id=p['A']['candidate_id'],
                      intervention_type='single_step', intervention_horizon=1,
                      execution_context_hash=plan['collection_context'])
        ends = {}
        for arm, role in [('A','BASE'), ('B','candidate')]:
            work = run / f'pair-{i:03d}' / arm
            raw = validate_arm(work, arm, p, plan)
            if raw['engine_receipt'] != 'accepted':
                raise ValueError(f'pair {i} {arm}: no accepted engine receipt')
            ep, = read_json(work/'runner/episodes.json')
            ends[arm] = ep
            receipts.append(make_execution_receipt(**common, branch_id=role, candidate_id=raw['candidate_id'],
                initial_state_hash=raw['initial_state_sha256'], prefix_hash=prefix(work/'runner', p, raw['final_actions']),
                source_fingerprint=raw['source_hash'], execution_started=True, execution_completed=True, episode_finished=True,
                outcome=ep['outcome'], return_value=ep['reward'], receipt_id=digest(raw), receipt_source='sc2_runner'))
        pair = build_pair_record(**common, run_id=rec['run_id'], episode_id=rec['episode_id'], scene_config_id=p['roster'],
            pair_id=f'pair-{i:03d}', candidate_id=p['B']['candidate_id'], initial_state_hash=anchors[key]['initial_state_hash'],
            prefix_hash=anchors[key]['prefix_hash'], candidate_return=ends['B']['reward'], baseline_return=ends['A']['reward'],
            outcome=ends['B']['outcome'], completion_status='complete')
        pairs.append(dict(pair, source_fingerprint=plan['source_hash'], scope='joint', decision_key=key))
    manifest = make_anchor_manifest(experiment_id=experiment, source_fingerprint=plan['source_hash'],
                                    execution_context_hash=plan['collection_context'], anchors=anchors)
    bundle = build_gain_dataset(records, pairs, anchor_manifest=manifest, execution_receipts=receipts)
    if bundle.audit_report['pairs_rejected'] or len(bundle.joint_records) != len(indices):
        raise ValueError(bundle.audit_report)
    args.out.mkdir(parents=True, exist_ok=False)
    for name, items in [('joint_records', bundle.joint_records), ('pairs', pairs), ('receipts', receipts)]:
        (args.out / (name + '.jsonl')).write_text(''.join(dumps(r) for r in items))
    for name, value in [('anchor_manifest',manifest), ('audit',bundle.audit_report), ('envelope',bundle.export_envelope())]:
        (args.out / (name + '.json')).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(f'{len(pairs)} audited gain labels, {len(receipts)} accepted receipts: {args.out}')


if __name__ == '__main__':
    main()
