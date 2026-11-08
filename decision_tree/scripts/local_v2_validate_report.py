"""Freeze training-only comparisons, then audit ten complete v1/v2 validation rollouts."""
import argparse
import json
import math
from pathlib import Path

from decision_tree.local.runtime import digest
from decision_tree.phase2.policy import Selector, accepted_attack, normalize_policy
from decision_tree.scripts.local_campaign_validate import validate_cohort
from skills.local_v2.explore import anchors, checksum, read_episodes, read_header, require
from skills.local_v2.library import analyze_binding
from skills.local_v2.run import parse_binding, source_hashes
from .local_v2_training_report import EPS, compare_sample, diff, read, save


def freeze(training_report, name):
    report=read(training_report)
    require(report["kind"]=="v2_action_training_evidence" and report["training_complete"],
            "Candidate freeze requires the complete training cohort")
    selector=report["frozen_selectors"][name]
    require(digest(selector["policy"])==selector["policy_sha256"] and len(selector["rows"])==30 and
            selector["versus"]["v1"]["missing_scenes"]==0 and selector["versus"]["base"]["missing_scenes"]==0,
            "Incomplete frozen-selector comparison")
    evidence=selector["versus"]["v1"]
    return {"kind":"local_v2_validation_freeze","training_report":str(training_report.resolve()),
            "training_report_sha256":checksum(training_report),"split_sha256":report["split_sha256"],
            "plan_sha256":report["plan_sha256"],"source_hashes":source_hashes(),
            "training_groups":report["training_groups"],"validation_groups":report["validation_groups_excluded"],
            "selector_name":name,"selector":selector["policy"],"selector_sha256":selector["policy_sha256"],
            "training_supported":evidence["different_effects"]>=4 and evidence["scene_equal_mean_gain"]>EPS
                             and evidence["rescued_rows"]>=evidence["lost_rows"],
            "minimum_distinct_changed_training_scenes":4}


def trajectory(arm,decisions,policy=None,implementation=None):
    """Recompute selector and action kernel online; keep only compact prefix hashes."""
    selector=Selector(policy) if policy else None
    fired=False
    proposal=None
    changed=None
    prefixes=[]
    terminal_hash=None
    for step,event,observation,baseline,anchor in anchors(arm,decisions):
        prefixes.append(anchor)
        sent=arm["steps"][step+1]
        if selector is None:
            require(event["selected_id"]==event["proposed_id"]=="BASE" and baseline["actions"]==sent["actions"],
                    "Modified BASE rollout")
        else:
            choice=selector.decide(event["candidates"],step,fired)
            require(event["phase2"]==choice, "Frozen selector differs from actual decision")
            info=event["local_v2"]
            cid=choice["selected_id"]
            expected_proposed=choice["considered_id"] if choice["action"] in ("decline","execute") else "BASE"
            require(not info["forced"] and info["implementation"]==implementation and
                    info["frozen_selected_id"]==cid and info["frozen_proposed_id"]==expected_proposed
                    and event["vetoed"]==(choice["action"]=="decline"), "Unbound frozen proposal")
            if choice["action"] in ("decline","execute"):
                require(proposal is None,"Repeated proposal")
                proposal={"step":step,"frozen_binding":choice["considered_id"],"action":choice["action"]}
            expected=list(baseline["actions"])
            actual_id=cid
            if cid!="BASE":
                family,actor,action=parse_binding(cid)
                if implementation=="v2":
                    b=analyze_binding(observation,baseline,family,actor)
                    require(b is not None,"Frozen candidate is absent from v2 kernel")
                    if not b.get("uses_frozen_target"):
                        action=b["recommended_action"]
                slot=int(actor[1:])
                if action==expected[slot]:
                    actual_id="BASE"
                else:
                    actual_id=f"{family}:{actor}:{action}"
                    changed={"step":step,"id":actual_id,"actor":actor,"family":family}
                expected[slot]=action
            require(event["selected_id"]==actual_id and event["final_actions"]==sent["actions"]==expected,
                    "Actual committed actions differ from reconstructed implementation")
        fired=fired or accepted_attack(sent["feedback"])
        terminal_hash=digest([anchor,{k:sent[k] for k in ("actions","game_loop","feedback","reward","terminated")}])
    return {"initial_state_sha256":arm["start"]["initial_state_sha256"],"game_version":arm["start"]["game_version"],
            "outcome":arm["record"]["outcome"],"return":arm["record"]["reward"],"steps":len(prefixes),
            "proposal":proposal,"changed":changed,"prefixes":prefixes,"terminal":terminal_hash}


def validate(proposal_path,split_path,plan,v1_path,v2_path):
    proposal=read(proposal_path)
    training=Path(proposal["training_report"])
    require(proposal==freeze(training,proposal["selector_name"]) and proposal["source_hashes"]==source_hashes(),
            "Frozen training decision or code changed")
    summary=read(plan/"summary.json")
    require(proposal["plan_sha256"]==checksum(plan/"summary.json") and proposal["split_sha256"]==checksum(split_path),
            "Unbound validation cohort or plan")
    base=Path(summary["base"])
    base_run=read_header(base)
    manifest=base_run["manifest.json"]
    cohort=validate_cohort(manifest,read(split_path))
    holdout={scene["sha256"] for _,scene in cohort}
    require(holdout==set(proposal["validation_groups"]) and not holdout.intersection(proposal["training_groups"]),
            "Training and validation groups overlap")
    records={"base":{}}
    for n,arm,decs in read_episodes(base,base_run):
        group=manifest["episodes"][n-1]["sha256"]
        if group in holdout:records["base"][group]=trajectory(arm,decs)
    require(base_run["hashes"]==summary["base_hashes"],"Original BASE evidence changed")
    provenance={"base":base_run["hashes"]}
    for implementation,path in (("v1",v1_path),("v2",v2_path)):
        run=read_header(path)
        marker=run["run.json"]["local_v2"]
        policy=normalize_policy(proposal["selector"])
        require(marker["selector_kind"]=="phase2" and marker["implementation"]==implementation and
                marker["selector"]==policy and marker["forced"] is None and
                run["run.json"]["local_policy"]["context_hash"]==summary["context"],"Wrong validation arm")
        actual={(x["sha256"],x["sc2_seed"]) for x in run["manifest.json"]["episodes"]}
        require(len(actual)==10 and actual=={(s["sha256"],s["sc2_seed"]) for _,s in cohort},
                "Validation population differs from the predeclared split")
        records[implementation]={}
        for n,arm,decs in read_episodes(path,run):
            group=run["manifest.json"]["episodes"][n-1]["sha256"]
            records[implementation][group]=trajectory(arm,decs,policy,implementation)
        provenance[implementation]=run["hashes"]
    rows=[]
    for _,scene in cohort:
        g=scene["sha256"]
        arms={label:group[g] for label,group in records.items()}
        original=arms["base"]
        require(arms["v1"]["proposal"]==arms["v2"]["proposal"],"Old tree's first proposal changed")
        for arm in (arms["v1"],arms["v2"]):
            require(arm["initial_state_sha256"]==original["initial_state_sha256"] and arm["game_version"]==original["game_version"],
                    "Games started in different native states")
            changed=arm["changed"]
            if changed:
                n=changed["step"]+1
                require(arm["prefixes"][:n]==original["prefixes"][:n],"Different observed prefix at intervention")
            else:
                require(arm["terminal"]==original["terminal"] and arm["steps"]==original["steps"] and
                        arm["outcome"]==original["outcome"] and abs(arm["return"]-original["return"])<1e-8,
                        "Unmodified trajectory differed from BASE")
        clean={name:{k:v for k,v in arm.items() if k not in ("prefixes","terminal")} for name,arm in arms.items()}
        rows.append(dict(seed=scene["sc2_seed"],group_id=g,**clean))
    metrics={}
    for reference in ("v1","base"):
        pairs=[diff({**r["v2"],"effect":str(r["v2"]["changed"])},{**r[reference],"effect":str(r[reference]["changed"])},
                    r["group_id"],None,None) for r in rows]
        metrics[reference]={**compare_sample(pairs),
            "rescued":[r["seed"] for r in rows if r["v2"]["outcome"]=="win" and r[reference]["outcome"]!="win"],
            "lost":[r["seed"] for r in rows if r["v2"]["outcome"]!="win" and r[reference]["outcome"]=="win"]}
    by_arm={a:{"wins":sum(r[a]["outcome"]=="win" for r in rows),"mean_return":math.fsum(r[a]["return"] for r in rows)/10}
            for a in ("base","v1","v2")}
    v1_gate=metrics["v1"]["scene_equal_mean_gain"]>EPS and len(metrics["v1"]["rescued"])>=len(metrics["v1"]["lost"])
    base_gate=metrics["base"]["scene_equal_mean_gain"]>EPS and len(metrics["base"]["rescued"])>=len(metrics["base"]["lost"])
    return {"kind":"v2_heldout_complete_policy_comparison","scope":"internal version-selection cohort",
            "proposal_sha256":checksum(proposal_path),"selector":proposal["selector"],"arms":by_arm,"versus":metrics,
            "rows":rows,"provenance":provenance,
            "passes_v1_gate":proposal["training_supported"] and v1_gate,
            "passes_preregistered_v1_and_base_gate":proposal["training_supported"] and v1_gate and base_gate,
            "beats_base_on_this_cohort":by_arm["v2"]["wins"]>by_arm["base"]["wins"]
                             and metrics["base"]["scene_equal_mean_gain"]>=-EPS}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest="command",required=True)
    create=subs.add_parser("freeze")
    create.add_argument("--training",required=True,type=Path)
    create.add_argument("--selector-name",required=True)
    check=subs.add_parser("validate")
    for key in ("proposal","split","plan","v1","v2"):
        check.add_argument("--"+key,required=True,type=Path)
    for parser in (create,check):parser.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    result=freeze(args.training,args.selector_name) if args.command=="freeze" else validate(
        args.proposal,args.split,args.plan,args.v1,args.v2)
    save(args.output,result)
    print(json.dumps({k:result[k] for k in ("kind","training_supported","passes_v1_gate","beats_base_on_this_cohort","arms") if k in result}))


if __name__=="__main__":
    main()
