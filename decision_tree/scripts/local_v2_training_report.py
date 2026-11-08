"""Compare v1 and v2 action kernels on a fixed selector's audited training scenes."""
import argparse
from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from decision_tree.local import study
from decision_tree.local.runtime import digest
from decision_tree.phase2.evidence import stats
from decision_tree.phase2.policy import Selector, accepted_attack, normalize_policy
from decision_tree.scripts.local_campaign_validate import validate_cohort
from skills.local_v2.explore import FILES, SCOPE, checksum, require
from skills.local_v2.run import parse_binding, source_hashes


EPS = 1e-9


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def compare_sample(rows):
    result = stats(rows)
    result.update(
        positive_rows=sum(r["gain"] > EPS for r in rows),
        zero_rows=sum(abs(r["gain"]) <= EPS for r in rows),
        negative_rows=sum(r["gain"] < -EPS for r in rows),
        rescued_rows=sum(r["left_outcome"] == "win" and r["right_outcome"] != "win" for r in rows),
        lost_rows=sum(r["left_outcome"] != "win" and r["right_outcome"] == "win" for r in rows),
    )
    if rows:
        result["largest_positive_or_best"] = max(rows, key=lambda r:r["gain"])
        result["largest_negative_or_worst"] = min(rows, key=lambda r:r["gain"])
    return result


def diff(left, right, group, step, actor):
    if left is None or right is None:
        return None
    return {"group_id":group, "step":step, "actor":actor, "gain":left["return"]-right["return"],
            "left":left["effect"], "right":right["effect"],
            "left_outcome":left["outcome"], "right_outcome":right["outcome"]}


def old_actions(row, binding):
    """Some v1 families offered multiple targets, not one standalone recommendation."""
    if binding["family"] in ("regroup_step","retreat_step"):
        action = binding.get("legacy_action")
        return [] if action is None else [action]
    family, actor = binding["family"],binding["actor"]
    return [a for f,u,a in map(parse_binding,row["frozen_v1_menu"]) if f==family and u==actor]


def load_labels(collection, groups, protocol):
    """Read only the training scene partitions from the audited ledger."""
    db = (collection/"coverage.sqlite3").resolve()
    labels, states = {}, Counter()
    with closing(sqlite3.connect(db.as_uri()+"?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        require(conn.execute("SELECT value FROM meta WHERE key='protocol'").fetchone()[0]==digest(protocol),
                "Ledger refers to a different collection protocol")
        for group in sorted(groups):
            query = "SELECT scene_hash,seed,step,effect,binding,state,accepted_path,result_json FROM tasks WHERE scene_hash=?"
            for task in conn.execute(query,(group,)):
                key = group,task["step"],task["effect"]
                require(key not in labels, "Duplicate training effect")
                states[task["state"]] += 1
                labels[key] = None
                if task["state"] != "accepted":
                    continue
                record = json.loads(task["result_json"])
                require(all(record[k]==task[k] for k in ("scene_hash","seed","step","effect","binding")),
                        "Audit label differs from its registered binding")
                path = (collection/task["accepted_path"]).resolve()
                require(path.is_relative_to(collection.resolve()), "Branch escaped evidence directory")
                proof=read(path/"proof.json")
                require(proof["audit"]==record and checksum(path/"events.jsonl.gz")==proof["compressed_events_sha256"],
                        "Changed audit proof or compressed events")
                for filename in FILES[:-1]:
                    require(checksum(path/filename)==record["run_hashes"][filename], "Changed branch "+filename)
                ep, = read(path/"episodes.json")
                require(ep["scenario_sha256"]==group and ep["reward"]==record["return"] and
                        ep["outcome"]==record["outcome"] and math.isfinite(ep["reward"]), "Unbound branch result")
                labels[key] = {"effect":record["effect"],"return":record["return"],"outcome":record["outcome"]}
    return labels,dict(states)


def first_selections(base, ordinals, policies, expected_hash):
    """Replay frozen selectors on training BASE prefixes with feedback only from earlier steps."""
    active = {n:{k:Selector(p) for k,p in policies.items()} for n in ordinals}
    selected = {n:{} for n in ordinals}
    fired = defaultdict(bool)
    next_step = Counter()
    ack = Counter()
    h = hashlib.sha256()
    with (base/"events.jsonl").open("rb") as stream:
        for line in stream:
            h.update(line)
            e = json.loads(line)
            n = e.get("episode")
            if n not in ordinals:
                continue
            if e["event"] in ("local_decision","local_v2_decision"):
                require(e["step"]==next_step[n]==ack[n] and e["selected_id"]=="BASE"
                        and e["baseline_actions"]==e["final_actions"], "Invalid BASE prefix")
                for name,sel in active[n].items():
                    if name in selected[n]:
                        continue
                    choice = sel.decide(e["candidates"],e["step"],fired[n])
                    if choice["consumed"]:
                        selected[n][name] = dict(choice, step=e["step"])
                next_step[n] += 1
            elif e["event"]=="step":
                require(e["step"]==next_step[n]==ack[n]+1, "Invalid receipt sequence")
                ack[n]+=1
                fired[n] = fired[n] or accepted_attack(e["feedback"])
    require(h.hexdigest()==expected_hash and all(next_step[n] and ack[n]==next_step[n] for n in ordinals),
            "Incomplete or changed training BASE source")
    return selected


def train(plan, collection, split_path, policy_paths, partial=False):
    summary = read(plan/"summary.json")
    require(summary["kind"]=="local_v2_exploration_plan" and summary["scope"]==SCOPE and
            source_hashes()==summary["source_hashes"], "Plan code or domain changed")
    base = Path(summary["base"])
    for name in FILES[:-1]:
        require(checksum(base/name)==summary["base_hashes"][name], "Changed BASE source "+name)
    manifest,episodes = read(base/"manifest.json"), read(base/"episodes.json")
    split = read(split_path)
    validate_cohort(manifest,split)
    groups = set(split["training_group_ids"])
    by_number = {n:s for n,s in enumerate(manifest["episodes"],1) if s["sha256"] in groups}
    policies = {k:normalize_policy(read(p)) for k,p in policy_paths.items()}
    require(bool(policies), "At least one frozen selector required")
    campaign_path=split_path.parent/"protocol.json"
    if campaign_path.is_file():
        campaign=read(campaign_path)
        if campaign.get("kind")=="full_single_effect_v2_train40":
            require(campaign["split"]==split and campaign["manifest_sha256"]==manifest["sha256"]
                    and campaign["source_hashes"]==summary["source_hashes"]
                    and set(policies)==set(campaign["frozen_selectors"]), "Experiment preregistration differs")
    protocol=read(collection/"protocol.json")
    require(protocol["scope"]==summary["scope"] and protocol["menus_sha256"]==summary["menus_sha256"] and
            protocol["source_hashes"]==summary["source_hashes"] and protocol["base_hashes"]==summary["base_hashes"] and
            protocol["total"]==summary["counts"]["nonbase_effects"],"Collector/plan binding mismatch")
    labels, states = load_labels(collection,groups,protocol)
    selections = first_selections(base,by_number,policies,summary["base_hashes"]["events.jsonl"])
    base_labels = {s["sha256"]:{"effect":"BASE","return":episodes[n-1]["reward"],"outcome":episodes[n-1]["outcome"]}
                   for n,s in by_number.items()}
    selected_rows = {k:[] for k in policies}
    families = defaultdict(lambda:defaultdict(list))
    totals = defaultdict(Counter)
    expected = set()
    steps_seen = Counter()
    menus_sha = hashlib.sha256()
    with (plan/"menus.jsonl").open("rb") as stream:
        for line in stream:
            menus_sha.update(line)
            menu = json.loads(line)
            group,n,step = menu["scene_hash"],menu["episode"],menu["step"]
            if group not in groups:
                continue
            require(n in by_number and by_number[n]["sha256"]==group and by_number[n]["sc2_seed"]==menu["seed"]
                    and step==steps_seen[n],"Training menu scene/step mismatch")
            steps_seen[n]+=1
            base_result = base_labels[group]
            expected.update((group,step,key) for key in menu["effects"] if key!="BASE")
            aliases = {alias:key for key,entry in menu["effects"].items() for alias in entry["aliases"]}
            bindings = {(b["family"],b["actor"]):b for b in menu["bindings"]}

            def outcome(b, action):
                cid = f"{b['family']}:{b['actor']}:{action}"
                effect = aliases[cid]
                return base_result if effect=="BASE" else labels.get((group,step,effect))

            for (family,actor),b in bindings.items():
                totals[family]["actor_steps"] += 1
                for option in b["options"]:
                    if option["action"] == b["base_action"]:
                        continue
                    totals[family]["all_nonbase_options"] += 1
                    pair=diff(outcome(b,option["action"]),base_result,group,step,actor)
                    if pair is not None:
                        families[family]["all_nonbase_options_vs_base"].append(pair)
                    if family in ("retarget","retreat_veto"):
                        # The old target family allowed many bindings. Keep all of
                        # them here, including those outside the old dispatch cap.
                        action = option["action"] if b.get("uses_frozen_target") else b["recommended_action"]
                        contrast = diff(outcome(b,action),outcome(b,option["action"]),group,step,actor)
                        if contrast is not None:
                            families[family]["v2_vs_each_legal_target"].append(contrast)
                versions = old_actions(menu,b)
                if not b.get("uses_frozen_target"):
                    rec = outcome(b,b["recommended_action"])
                    pair = diff(rec,base_result,group,step,actor)
                    if pair is not None:
                        families[family]["recommended_vs_base"].append(pair)
                    if not versions:
                        totals[family]["actor_steps_without_v1_binding"] += 1
                for old in versions:
                    totals[family]["eligible_v1_comparisons"]+=1
                    rec = outcome(b,old if b.get("uses_frozen_target") else b["recommended_action"])
                    prior = outcome(b,old)
                    pair = diff(rec,prior,group,step,actor)
                    if pair is not None:
                        families[family]["recommended_vs_v1"].append(pair)
                        if pair["left"]!=pair["right"]:
                            families[family]["changed_vs_v1"].append(pair)

            for name,event in selections[n].items():
                if event["step"]!=step:
                    continue
                old,new = base_result,base_result
                family,actor,action = parse_binding(event["considered_id"])
                if event["action"]=="execute":
                    binding = bindings[family,actor]
                    old = outcome(binding,action)
                    new = outcome(binding,action if binding.get("uses_frozen_target") else binding["recommended_action"])
                selected_rows[name].append({"group_id":group,"seed":menu["seed"],"step":step,"actor":actor,"family":family,
                                            "frozen":event,"v1":old,"v2":new,"base":base_result})
    require(menus_sha.hexdigest()==summary["menus_sha256"] and set(labels)==expected,
            "Unregistered effect or incomplete ledger initialization")
    for n,s in by_number.items():
        require(steps_seen[n]==episodes[n-1]["steps"],"Missing training step")
        for name in policies:
            if name not in selections[n]:
                b=base_labels[s["sha256"]]
                selected_rows[name].append({"group_id":s["sha256"],"seed":s["sc2_seed"],"step":None,"actor":None,
                                            "family":None,"frozen":None,"v1":b,"v2":b,"base":b})
    complete = len(expected)==states.get("accepted",0)
    require(complete or partial,"Missing training branches; use --partial for coverage-only diagnostics")
    selector_results = {}
    for name,rows in selected_rows.items():
        require(len(rows)==len(groups)==len({r["group_id"] for r in rows}),"Incomplete selector cohort")
        versus={}
        for key in ("base","v1"):
            pairs = [pair for row in rows if (pair:=diff(row["v2"],row[key],row["group_id"],row["step"],row["actor"])) is not None]
            versus[key] = {**compare_sample(pairs),"eligible_scenes":len(rows),"missing_scenes":len(rows)-len(pairs),
                           "different_effects":sum(p["left"]!=p["right"] for p in pairs)}
        selector_results[name]={"policy":policies[name],"policy_sha256":digest(policies[name]),"versus":versus,
                                "rows":sorted(rows,key=lambda row:row["seed"])}
    return {"kind":"v2_action_training_evidence","partition":"training_only",
            "training_groups":sorted(groups),"validation_groups_excluded":sorted(split["validation_group_ids"]),
            "training_complete":complete,"training_registered_nonbase":len(expected),"training_ledger_states":states,
            "plan_sha256":checksum(plan/"summary.json"),"menus_sha256":summary["menus_sha256"],
            "collector_protocol_sha256":checksum(collection/"protocol.json"),"split_sha256":checksum(split_path),
            "frozen_selectors":selector_results,
            "families":{f:{"counts":dict(totals[f]),"statistics":{k:compare_sample(rows) for k,rows in families[f].items()}}
                        for f in totals},
            "interpretation":["Families weight each scene equally after averaging eligible step/actor comparisons.",
                "For target-choice kernels each offered v1 target is a comparison, not a unique v1 deployment choice.",
                "Frozen-selector rows use at most one executed intervention per episode; missing returns remain null.",
                "Family-level binding diagnostics are not whole-policy win rates."]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",required=True,type=Path)
    p.add_argument("--collection",required=True,type=Path)
    p.add_argument("--split",required=True,type=Path)
    p.add_argument("--selector",action="append",nargs=2,metavar=("NAME","PATH"),required=True)
    p.add_argument("--partial",action="store_true",help="explicitly allow incomplete training-only summaries")
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    policies={k:Path(v) for k,v in args.selector}
    require(len(policies)==len(args.selector),"Duplicate selector name")
    report=train(args.plan,args.collection,args.split,policies,args.partial)
    save(args.output,report)
    print(json.dumps({k:report[k] for k in ("training_complete","training_registered_nonbase","training_ledger_states")}))


if __name__=="__main__":
    main()
