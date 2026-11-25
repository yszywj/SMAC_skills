"""Fit a single weighted CART on joined, frozen Zerg shadow episodes (no simulator)."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from decision_tree.zerg import policy_distill as policy

FROZEN = ROOT / 'runs/zerg_joint_imitation_480_20261123_194909/split.json'
SPLIT_HASH = '92a8b695f48702f30f314588962fac0df79f188bd64dfafcb66ca035568093b7'


def episode(item):
    gid, meta = item
    work = ROOT / meta['batch'] / meta['worker']
    hashes = {}

    def lines(name):
        h = sha256()
        with (work/name).open('rb') as stream:
            for line in stream:
                h.update(line)
                yield line
        hashes[name] = h.hexdigest()

    observations = {}
    for line in lines('runner/events.jsonl'):
        if b'"skill_observation"' not in line:
            continue
        e = json.loads(line)
        if e['event'] == 'skill_observation':
            key = (e['episode'], e['step'], e['payload']['state']['game_loop'])
            if key in observations:
                raise ValueError('duplicate observation: ' + gid)
            observations[key] = e['payload']['state']
    rows, decisions, contexts = [], [], set()
    for step, line in enumerate(lines('shadow/shadow.jsonl')):
        r = json.loads(line)
        if (gid != '%s:%s' % (r['run_id'], r['episode_id']) or meta['seed'] != r['seed'] or
                meta['roster'] != r['scene_config_id'] or r['snapshot_id'] != step + 1):
            raise ValueError('shadow identity mismatch: ' + gid)
        s = observations.pop((int(r['episode_id']), step, r['game_loop']))
        contexts.add((r['source_hash'], r['execution_context_hash']))
        cands = [c for c in r['candidates'] if c['candidate_family'] == 'joint_plan' and c['legal'] is True]
        teacher, = [policy.vector(s, c['proposed_actions']) for c in cands if c['candidate_id'] == r['teacher_choice']]
        if teacher != policy.vector(s, r['final_actions']):
            raise ValueError('teacher not the submitted vector: ' + gid)
        unique = sorted({policy.vector(s, c['proposed_actions']) for c in cands})
        decisions.append((unique, unique.index(teacher), policy.tie_salt(s), len(cands)))
        rows.extend(policy.project(s, v) for v in unique)
    if observations or len(contexts) != 1 or not decisions:
        raise ValueError('incomplete observation/decision join: ' + gid)
    return gid, np.array(rows, dtype=float), decisions, contexts.pop(), hashes


def fit(games, ids, depth_limit, min_leaf=24):
    x = np.concatenate([games[g][0] for g in ids]); labels, weights = [], []
    for g in ids:
        ds = games[g][1]
        for vs, label, _, _ in ds:
            labels.extend(int(i == label) for i in range(len(vs)))
            weights.extend([1. / (len(ds)*len(vs))] * len(vs))
    y, w = np.array(labels), np.array(weights)
    nodes, importance = [], np.zeros(x.shape[1])
    depths = []

    def grow(ix, depth):
        idx = len(nodes); nodes.append(None)
        weight = w[ix].sum(); pos = (w[ix]*y[ix]).sum()
        best, split = 1e-10, None
        if depth < depth_limit and len(ix) >= 2*min_leaf and 0 < pos < weight:
            for j in range(x.shape[1]):
                col = x[ix,j]
                if col.min() == col.max():
                    continue
                order = col.argsort(kind='stable'); a = col[order]; oi = ix[order]
                cut = np.flatnonzero(a[1:] > a[:-1]); cut = cut[(cut+1 >= min_leaf) & (len(ix)-1-cut >= min_leaf)]
                if not len(cut):
                    continue
                left = w[oi].cumsum()[cut]; p = (w[oi]*y[oi]).cumsum()[cut]
                gain = p*p/left + (pos-p)**2/(weight-left) - pos*pos/weight
                k = gain.argmax()
                if gain[k] > best:
                    best = gain[k]; split = (j, float(a[cut[k]]))
        if split is None:
            depths.append(depth); nodes[idx] = [float(np.clip(pos/weight, 0., 1.))]
        else:
            j, threshold = split; left = x[ix,j] <= threshold
            importance[j] += best
            nodes[idx] = [j, threshold, grow(ix[left], depth+1), grow(ix[~left], depth+1)]
        return idx

    grow(np.arange(len(x)), 0)
    return nodes, dict(depth=max(depths), leaves=len(depths), feature_importances=dict(sorted(
        zip(policy.FEATURES, (importance / max(importance.sum(), 1e-99)).tolist()), key=lambda kv: -kv[1])))


def evaluate(nodes, games, ids):
    totals, macro = defaultdict(list), defaultdict(list)
    counts = Counter()
    for g in ids:
        x, ds = games[g][:2]; offset = 0; by_ep = defaultdict(list)
        for vs, label, salt, nraw in ds:
            n = len(vs); scores = [policy.score(nodes, row) for row in x[offset:offset+n]]; offset += n
            chosen, ties = policy.resolve(salt, vs, scores)
            row = (int(chosen == vs[label]), int(max(scores)-scores[label] <= 1e-12)/ties, 1./n, int(ties > 1))
            counts.update(dict(decisions=1, raw_candidates=nraw, unique_candidates=n, multi_decisions=int(n>1)))
            for scope in ('all','multi') if n>1 else ('all',):
                totals[scope].append(row); by_ep[scope].append(row)
        for scope, rows in by_ep.items():
            macro[scope].append(np.mean(rows, axis=0))
    fields = ('deterministic_top1','tie_neutral_top1','uniform_random_top1','top_tie_rate')
    return dict(episodes=len(ids), counts=counts, **{scope:{kind:dict(zip(fields, np.mean(d[scope],axis=0).tolist()))
                for kind,d in [('per_beat',totals),('episode_equal',macro)]} for scope in totals})


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path); p.add_argument('--workers',type=int,default=8)
    args = p.parse_args(); out = args.out or ROOT/'runs/zerg_policy_distill_v1'
    if not args.out and out.exists():
        out = out.with_name(out.name+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if out.exists():
        p.error('output path must be new')
    raw = FROZEN.read_bytes(); split = json.loads(raw)
    if sha256(raw).hexdigest() != SPLIT_HASH:
        raise ValueError('frozen split changed')
    train, valid = ([g for g in sorted(split) if split[g]['partition'] == s] for s in ('train','validation'))
    strata = defaultdict(list)
    for g in train:
        strata[split[g]['roster']].append(g)
    inner = sorted(g for groups in strata.values() for g in sorted(groups,key=lambda v:policy.digest(['inner-v1',v]))[:6])
    development = sorted(set(train)-set(inner)); games = {}; provenance = {}; contexts = set()

    def load(pool, ids):
        for g, x, ds, context, hashes in pool.map(episode, [(g,split[g]) for g in ids]):
            games[g] = (x,ds); contexts.add(context); provenance[g] = dict(**split[g], file_sha256=hashes)
        if len(contexts) != 1:
            raise ValueError('mixed contexts')

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        load(pool, train); search = {}
        for depth in (6,10,14):
            nodes, _ = fit(games, development, depth)
            search[depth] = evaluate(nodes, games, inner)['multi']['episode_equal']['tie_neutral_top1']
        depth = max(search,key=lambda d:(search[d],-d)); nodes, stats = fit(games,train,depth)
        load(pool, valid)
    src, ctx = contexts.pop(); identity = dict(source_hash=src,ctx_hash=ctx,schema_hash=policy.SCHEMA_HASH,split_hash=SPLIT_HASH)
    m = dict(**identity,version=policy.VERSION,features=policy.FEATURES,nodes=nodes)
    out.mkdir(parents=True,exist_ok=False)

    def save(name, value):
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

    save('model.json',m); policy.load_model(out/'model.json'); (out/'split.json').write_bytes(raw)
    save('schema.json',dict(version=policy.VERSION,features=policy.FEATURES,schema_hash=policy.SCHEMA_HASH,
         inputs=['skill_observation.payload.state','complete joint action'],tie='sha256(pre-state salt, vector); tolerance 1e-12'))
    save('provenance.json',dict(**identity,split_source=str(FROZEN.relative_to(ROOT)),episodes=provenance,
         files={str(path.relative_to(ROOT)):sha256(path.read_bytes()).hexdigest() for path in (Path(__file__).resolve(),Path(policy.__file__).resolve())},
         model_sha256=sha256((out/'model.json').read_bytes()).hexdigest(),python=sys.version,numpy=np.__version__))
    report = dict(**identity,**stats, selection=dict(inner_validation=inner, inner_train=development, candidates=search,
        metric='episode_equal multi-beat tie-neutral top1; smaller max_depth on tie',selected_max_depth=depth,min_leaf_rows=24),
        loss='weighted CART squared error, binary exact joint teacher label',weight='1/(episode beats * unique candidates at beat)',
        train=evaluate(nodes,games,train),validation=evaluate(nodes,games,valid),
        by_roster={r:evaluate(nodes,games,[g for g in valid if split[g]['roster']==r]) for r in sorted(strata)})
    save('report.json',report); print(out, stats['depth'],stats['leaves'],report['validation'],flush=True)


if __name__ == '__main__':
    main()
