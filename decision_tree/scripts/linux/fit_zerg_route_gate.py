"""Offline only. Re-audit full v2/alternative panels; never train on report.json."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from decision_tree.zerg.policy_route_gate import CUTS, SCHEMA, valid_features


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(x):
    return isinstance(x, str) and re.fullmatch('[0-9a-f]{64}', x) is not None


def load_panel(path):
    design = json.loads((path / 'design.json').read_text())
    require(design.get('schema') == 'zerg_route_pair_v1' and
            design.get('arms') == ['v2', 'alternative'], 'not a full v2/modified-alternative panel')
    hashes = design['policy_sha256']
    base = Path(__file__).resolve().parents[3]
    source_paths = {'v2': base/'decision_tree/zerg/policy_repair_v2.py',
                    'alternative': base/'decision_tree/zerg/policy_route_alt.py',
                    'gate_code': base/'decision_tree/zerg/policy_route_gate.py',
                    'opening': base/'decision_tree/zerg/opening_support_v2.py',
                    'model': path/'model.json'}
    require(isinstance(hashes, dict) and all(digest(v) for v in hashes.values()) and
            all(hashes.get(k) == sha256(p.read_bytes()).hexdigest() for k,p in source_paths.items()),
            'model or policy source fingerprint mismatch')
    for arm in design['arms']:
        policy = design['policies'][arm]
        require(policy['policy_version'] == arm and policy['selector_sha256'] == hashes[arm] and
                policy['trial_id'] == design['trial_id'] and policy['policy_sha256'] == hashes,
                'wrong policy identity or trial')
    # The runner's offline auditor opens both raw trajectories, not a cached report.
    from decision_tree.scripts.linux.eval_zerg_route import audit_panel
    report = audit_panel(path)
    require(hashes == report['policy_sha256'], 'audited policy fingerprint mismatch')
    rows = report['paired_outcomes']
    index = {r['pair_id']: r['scenario_sha256'] for r in design['pairs']}
    require(len(index) == len(rows) == len(design['pairs']) and
            index == {r['pair_id']: r['scenario_sha256'] for r in rows}, 'incomplete pair set')
    for r in rows:
        if r['prefix_verified'] is True and r['micro_verified'] is True:
            if r['anchor'] is None:
                require(r['route_choice'] is None, 'unanchored intervention')
            else:
                require(r['route_choice'] == 'alternative' and r['anchor']['policy_sha256'] == hashes,
                        'no certified modified-plan intervention')
    return hashes, rows


def normalize(row):
    scenario = row['scenario_sha256']
    require(digest(scenario), 'missing scenario identity')
    require(row.get('prefix_verified') is True and row.get('micro_verified') is True,
            'unverified prefix/actions/micro')
    p, a = row['v2'], row['alternative']
    require(digest(p['initial']) and p['initial'] == a['initial'], 'initial state mismatch')
    for arm in (p, a):
        require(arm['outcome'] in ('loss', 'win', 'draw') and
                type(arm['enemy_elimination_fraction']) in (int, float) and
                0 <= arm['enemy_elimination_fraction'] <= 1, 'invalid terminal')
    features = None
    if row['anchor'] is not None:
        anchor = row['anchor']
        before, after = anchor['parent_plan'], anchor['alt_plan']
        require(any(before[k] != after[k] for k in ('actions', 'plan_ids')), 'no executable divergence')
        features = anchor['features']
        require(valid_features(features), 'invalid anchor feature set')
    else:
        require(all(p[k] == a[k] for k in ('outcome', 'enemy_elimination_fraction')),
                'different terminal without verified intervention')
    pw, aw = p['outcome'] == 'win', a['outcome'] == 'win'
    ps, als = [x['outcome'] == 'loss' and x['enemy_elimination_fraction'] <= .5 for x in (p, a)]
    return dict(scenario=scenario, features=features, transition=p['outcome'] + '_to_' + a['outcome'],
                rescued=p['outcome'] == 'loss' and aw, dropped=pw and not aw,
                parent_severe=ps, alternative_severe=als, new_severe=als and not ps,
                relieved_severe=ps and not als, severe_to_win=ps and aw)


def fold(scenario):
    return int.from_bytes(sha256(('route-gate-v1:' + scenario).encode()).digest()[:8], 'big') % 5


def summarize(rows):
    return dict(n=len(rows), no_intervention=sum(r['features'] is None for r in rows),
                transitions=dict(sorted(Counter(r['transition'] for r in rows).items())),
                **{k: sum(r[k] for r in rows) for k in ('rescued', 'dropped', 'parent_severe',
                   'alternative_severe', 'new_severe', 'relieved_severe', 'severe_to_win')})


def selected(rows, rule):
    if rule is None:
        return []
    return [r for r in rows if r['features'] is not None and
            ((r['features'][rule['feature']] > rule['threshold']) == (rule['op'] == 'gt'))]


def safe(s, n, rescues):
    return s['n'] >= n and s['rescued'] >= rescues and not (s['dropped'] or s['new_severe'])


def learn(rows):
    best, best_score = None, None
    for key, cuts in CUTS.items():
        for cut in cuts:
            for op in ('le', 'gt'):
                rule = dict(feature=key, threshold=cut, op=op)
                s = summarize(selected(rows, rule))
                if safe(s, 8, 4):
                    score = (s['rescued'], s['severe_to_win'], s['relieved_severe'], -s['n'])
                    if best_score is None or score > best_score:
                        best, best_score = rule, score
    return best


def fit(rows, hashes):
    rows = list(rows)
    require(len({r['scenario_sha256'] for r in rows}) == len(rows), 'duplicate scenario; never count twice')
    data, excluded = [], {}
    for row in rows:
        if (row.get('prefix_verified') is False or row.get('micro_verified') is False):
            excluded[row['scenario_sha256']] = row.get('exclusion_reason') or 'unverified prefix/actions/micro'
        else:
            data.append(normalize(row))
    data.sort(key=lambda r: r['scenario'])
    bins = [[r for r in data if fold(r['scenario']) == i] for i in range(5)]
    # Fold zero is untouched during search. Folds 1..4 independently refit for OOF.
    train = sum(bins[1:], [])
    candidate = learn(train)
    folds, oof = [], []
    for i in range(1, 5):
        rule = learn(sum((b for j,b in enumerate(bins) if j not in (0, i)), []))
        votes = selected(bins[i], rule)
        folds.append(dict(fold=i, validation_n=len(bins[i]), rule=rule, selected=summarize(votes)))
        oof.extend(votes)
    hold, cv = summarize(selected(bins[0], candidate)), summarize(oof)
    reasons = []
    if excluded:
        reasons.append('incomplete_verification_in_panel')
    if candidate is None:
        reasons.append('no_supported_training_leaf')
    if not safe(cv, 8, 4) or sum(x['selected']['rescued'] > 0 for x in folds) < 2:
        reasons.append('insufficient_safe_cross_fold_rescues')
    if not safe(hold, 4, 2):
        reasons.append('insufficient_safe_held_out_rescues')
    model = dict(schema=SCHEMA, default='v2', tree=None if reasons else candidate,
                 policy_sha256=hashes)
    report = dict(status='empty_default_v2' if reasons else 'supported_stump', reasons=reasons,
                  total_pairs=len(rows), excluded=excluded,
                  cohort=summarize(data), interventions=summarize([r for r in data if r['features'] is not None]),
                  train_candidate=candidate, train_selected=summarize(selected(train, candidate)),
                  cross_validation=folds, oof_selected=cv, holdout_n=len(bins[0]), holdout_selected=hold,
                  split='sha256(route-gate-v1: + scenario_sha256), first 8 bytes big endian mod 5; holdout=0',
                  assignments={r['scenario']: fold(r['scenario']) for r in data},
                  severe_definition='loss and enemy_elimination_fraction <= 0.5',
                  acceptance='train >=8 selected/4 loss-to-win; OOF >=8/4 in >=2 folds; holdout >=4/2; zero dropped wins and new severe losses in each',
                  evaluation='offline paired outcomes, not a played gated policy')
    return model, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, action='append', required=True)
    parser.add_argument('--out', required=True, type=Path, help='gate model JSON')
    parser.add_argument('--report', required=True, type=Path, help='offline diagnostics JSON')
    args = parser.parse_args(argv)
    require(args.out.resolve() != args.report.resolve(), 'separate model and report paths required')
    hashes, rows = None, []
    for panel in args.panel:
        policy, samples = load_panel(panel)
        require(hashes is None or hashes == policy, 'mixed policy fingerprints')
        hashes = policy
        rows.extend(samples)
    model, report = fit(rows, hashes)
    report.update(panels=[str(p) for p in args.panel], policy_sha256=hashes)
    for path, obj in ((args.out, model), (args.report, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], reasons=report['reasons'], pairs=len(rows),
                          tree=model['tree'], model=str(args.out), report=str(args.report))))


if __name__ == '__main__':
    main()
