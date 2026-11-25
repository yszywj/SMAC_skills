"""Verify and combine v1 single-step fork exports without trusting their joined targets."""
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re

from decision_tree.zerg.dataset import dumps_key, episode_group, record_key
from decision_tree.zerg.experiment_manifest import make_anchor_manifest
from decision_tree.zerg.pair_dataset import build_gain_dataset, build_receipt_index
from decision_tree.zerg.schema import digest, validate_decision_record, validate_pair_record

ROOT = Path(__file__).resolve().parents[3]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def unlabel(r):
    # build_gain_dataset copies input candidates, including *preexisting* labels.
    # Remove every joint label before replaying the join, not just the expected B.
    candidates = []
    for c in r['candidates']:
        c = dict(c)
        for k in ('pair_id', 'pair_experiment_id', 'pair_anchor_id'):
            c.pop(k, None)
        if c.get('candidate_return') is not None:
            del c['candidate_return']
        candidates.append(c)
    require(not any(c.get('candidate_return') is not None for group in
                    r['skill_proposals'].values() for c in group), 'unexpected internal target')
    return dict(r, candidates=candidates, baseline_return=None)


def fork_facts(run, record, pair, receipts):
    """V1 does not serialize scenario identity in the normalized receipt. Resolve its preimage."""
    key = pair['anchor_id']
    require(type(record['seed']) is int and record['scene_config_id'] and
            record['scene_config_id'] == pair['scene_config_id'], f'{key}: missing seed or wrong roster')
    require(record['teacher_choice'] == record['selected_candidate'] == pair['baseline_id'] and
            pair['candidate_id'] != pair['baseline_id'], f'{key}: wrong A/B candidate')
    members = {c['candidate_id']: c for c in record['candidates'] if c['legal'] and c['candidate_family'] == 'joint_plan'}
    arms = {}
    scenarios = set()
    for arm, role, field, target in [('A', 'BASE', 'baseline_id', 'baseline_return'),
                                     ('B', 'candidate', 'candidate_id', 'candidate_return')]:
        cid = pair[field]
        require(cid in members, f'{key}: missing legal joint candidate {cid}')
        c = members[cid]
        receipt = receipts[(pair['experiment_id'], key, role, cid)]
        runner = run / pair['pair_id'] / arm / 'runner'
        raw = load(runner / 'fork_receipt.json')
        ep, = load(runner / 'episodes.json')
        require(receipt['receipt_source'] == 'sc2_runner' and digest(raw) == receipt['receipt_id'],
                f'{runner}: fork receipt digest or source mismatch')
        expected = dict(kind='zerg_one_shot_fork_receipt_v1', status='complete', error=None,
                        engine_receipt='accepted', anchor_commits=1, prefix_verified=True,
                        arm=arm, episode_group=episode_group(record), seed=record['seed'],
                        roster=record['scene_config_id'], snapshot_id=record['snapshot_id'],
                        game_loop=record['game_loop'], candidate_id=cid, source_plan_id=c['source_plan_id'],
                        final_actions=c['proposed_actions'], initial_state_sha256=pair['initial_state_hash'],
                        source_hash=record['source_hash'], collection_context=record['execution_context_hash'])
        require(digest({k: raw[k] for k in expected}) == digest(expected), f'{runner}: wrong branch evidence')
        scenario = raw['scenario_sha256']
        require(re.fullmatch('[0-9a-f]{64}', scenario) and ep['scenario_sha256'] == scenario and
                ep['initial_state_sha256'] == raw['initial_state_sha256'] and
                ep['reward'] == receipt['return'] == pair[target] and ep['outcome'] == receipt['outcome'],
                f'{runner}: episode/receipt/target mismatch')
        scenarios.add(scenario)
        arms[arm] = dict(candidate_id=cid, source_plan_id=c['source_plan_id'], skill=c['candidate_skill'],
                         actions=c['proposed_actions'], outcome=ep['outcome'], receipt_id=receipt['receipt_id'],
                         **{'return': ep['reward']})
    a, b = arms['A']['actions'], arms['B']['actions']
    require(a == record['final_actions'] and a != b and a.keys() == b.keys() and len(scenarios) == 1,
            f'{key}: invalid one-step fork')
    require(pair['outcome'] == arms['B']['outcome'] and
            pair['gain'] == arms['B']['return'] - arms['A']['return'], f'{key}: wrong paired outcome/gain')
    return dict(experiment_id=pair['experiment_id'], pair_id=pair['pair_id'], decision_key=key,
                episode_group=episode_group(record), snapshot_id=record['snapshot_id'], game_loop=record['game_loop'],
                roster=record['scene_config_id'], seed=record['seed'], scenario_sha256=scenarios.pop(),
                initial_state_hash=pair['initial_state_hash'],
                pre_action_visible_enemy_count=record['global_features']['team'].get('enemy_visible_count'),
                original=arms['A'], alternate=arms['B'], gain=pair['gain'])


@dataclass
class Export:
    path: Path
    envelope: dict
    manifest: dict
    pairs: list
    receipts: list
    clean: list
    records: list
    facts: list


def verify_export(path):
    path = path.resolve()
    env = load(path / 'envelope.json')
    manifest = load(path / 'anchor_manifest.json')
    records = [validate_decision_record(r) for r in lines(path / 'joint_records.jsonl')]
    pairs = [validate_pair_record(p) for p in lines(path / 'pairs.jsonl')]
    receipts = lines(path / 'receipts.jsonl')
    index = build_receipt_index(receipts)
    exp = env['experiment_id']
    require(pairs and len(pairs) == len(records) == len(manifest['anchors']) and
            len({p['pair_id'] for p in pairs}) == len(pairs) and
            len({p['anchor_id'] for p in pairs}) == len(pairs) and
            len(index) == len(receipts) == 2 * len(pairs),
            f'{path}: duplicate or missing pair, receipt, anchor or record')
    for p, r in zip(pairs, records):
        require(p['experiment_id'] == exp and p['anchor_id'] == p['decision_key'] == dumps_key(record_key(r)) and
                (p['scope'], p['intervention_type'], p['intervention_horizon']) == ('joint', 'single_step', 1),
                f'{path}: incompatible experiment, scope, decision or horizon')
    clean = [unlabel(r) for r in records]
    bundle = build_gain_dataset(clean, pairs, anchor_manifest=manifest, execution_receipts=receipts)
    require(not bundle.audit_report['pairs_rejected'], f'{path}: {bundle.audit_report["problems_by_pair"]}')
    require(digest(bundle.export_envelope()) == digest(env) and digest(records) == env['records_hash'] and
            digest(bundle.audit_report) == digest(load(path / 'audit.json')),
            f'{path}: envelope, joined records/targets or audit report mismatch')

    # Archive may be next to the export, its enclosing run, or in the standard runs directory.
    require(exp not in ('.', '..') and all(p['pair_id'] not in ('.', '..') for p in pairs), 'invalid archive identity')
    run = next((d for d in (path.parent, path.parent / exp, ROOT / 'runs' / exp)
                if (d / pairs[0]['pair_id'] / 'A/runner/fork_receipt.json').is_file()), None)
    require(run is not None, f'{path}: original {exp} fork archive required for scenario/receipt verification')
    facts = [fork_facts(run, r, p, index) for r, p in zip(records, pairs)]
    return Export(path, env, manifest, pairs, receipts, clean, records, facts)


def cumulative(exports):
    first = exports[0].envelope
    seen = defaultdict(set)

    def unique(field, value):
        require(value not in seen[field], f'duplicate/overlapping {field}: {value}')
        seen[field].add(value)

    for batch in exports:
        unique('experiment_id', batch.envelope['experiment_id'])
        for k in ('source_fingerprint', 'execution_context_hash'):
            require(batch.envelope[k] == first[k], f'incompatible {k}')
        require(all(r.get('map_id') and r['map_id'] == exports[0].records[0]['map_id'] for r in batch.records),
                'missing or incompatible map_id')
        for f in batch.facts:
            for k in ('decision_key', 'episode_group', 'seed', 'scenario_sha256', 'initial_state_hash'):
                unique(k, f[k])
            for arm in ('original', 'alternate'):
                unique('receipt_id', f[arm]['receipt_id'])
    if len(exports) == 1:
        return exports[0].records, first

    # Re-audit an explicit, derived namespace. Original anchor/episode/candidate identities,
    # raw receipt IDs and evidence stay unchanged; only experiment and local pair IDs change.
    experiment = 'cumulative-' + digest([b.envelope for b in exports])
    manifest = make_anchor_manifest(experiment_id=experiment, source_fingerprint=first['source_fingerprint'],
        execution_context_hash=first['execution_context_hash'],
        anchors={k: v for b in exports for k, v in b.manifest['anchors'].items()})
    pairs = [dict(p, experiment_id=experiment, pair_id=digest([p['experiment_id'], p['pair_id']]))
             for b in exports for p in b.pairs]
    receipts = [dict(r, experiment_id=experiment) for b in exports for r in b.receipts]
    bundle = build_gain_dataset([r for b in exports for r in b.clean], pairs,
                                anchor_manifest=manifest, execution_receipts=receipts)
    require(not bundle.audit_report['pairs_rejected'] and len(bundle.joint_records) == len(pairs),
            f'cumulative audit failed: {bundle.audit_report}')
    return bundle.joint_records, bundle.export_envelope()
