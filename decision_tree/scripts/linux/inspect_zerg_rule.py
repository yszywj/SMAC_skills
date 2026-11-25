"""Reproduce the rule coverage diagnosis on the original 18 training anchors only."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from decision_tree.scripts.linux.plan_zerg_abc import native_prefix, read
from decision_tree.zerg.rules_v1 import make_entry


def main():
    evidence = read(ROOT/'runs/zerg_workflow/round-001/review/llm_train_facts.json')
    if evidence['scope'] != 'training_pairs_only': raise ValueError('wrong facts scope')
    plan = read(ROOT/'runs/zerg_forks_24x2_20261123_204135/plan.json')
    split = read(ROOT/plan['fit']/'split.json')
    emitted = 0
    for row in evidence['pairs']:
        group, sid = row['episode_group'], row['snapshot_id']
        ep = split[group]; work = ROOT/ep['batch']/ep['worker']
        if row['partition']!='train' or ep['partition']!='train': raise ValueError('holdout forbidden')
        a,b = row['original']['actions'], row['alternate']['actions']
        from contextlib import closing
        with closing(native_prefix(work)) as prefix:
            for step,mgr,summary,obs in prefix:
                if step!=sid: continue
                if summary['actions']!=a or summary['snapshot_game_loop']!=row['game_loop']:
                    raise ValueError('anchor does not reproduce received training facts')
                cache = list(mgr.coordinator._cache.values())
                uniq = {tuple(sorted(e.actions.items())) for e in cache}
                c = make_entry(mgr,summary,obs)
                if c is not None and tuple(sorted(c.actions.items())) in uniq: raise ValueError('C in old cache')
                print(json.dumps(dict(pair=row['pair_id'],snapshot=sid,loop=row['game_loop'],cache_entries=len(cache),
                    distinct_joint_vectors=len(uniq),A_in_cache=tuple(sorted(a.items())) in uniq,
                    B_in_cache=tuple(sorted(b.items())) in uniq,C_changes=None if c is None else
                    {u:[a[u],v] for u,v in c.actions.items() if a[u]!=v},measured_B_minus_A=row['gain'])))
                emitted += c is not None
                break
            else: raise ValueError('anchor not found')
    print(f'{emitted}/{len(evidence["pairs"])} training anchors emitted a new C; no SC2 execution.')


if __name__=='__main__':
    main()
