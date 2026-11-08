import json
from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from decision_tree.local.runtime import digest
from decision_tree.scripts.local_v2_training_report import compare_sample, old_actions, train
from decision_tree.scripts.local_v2_validate_report import freeze
from skills.local_v2.explore import checksum, SCOPE, FILES
from skills.local_v2.run import source_hashes


def write(p,obj):
    p.write_text(json.dumps(obj),encoding="utf-8")


class TrainingReport(unittest.TestCase):
    def test_scene_mass_and_repeated_targets(self):
        rows=[dict(group_id=g,gain=v,left_outcome="win",right_outcome="win") for g,v in
              [("one",3),("one",3),("two",-2)]]
        self.assertEqual(compare_sample(rows)["scene_equal_mean_gain"],.5)
        binding={"family":"retarget","actor":"u0"}
        self.assertEqual(old_actions({"frozen_v1_menu":["retarget:u0:6","retarget:u1:7","retarget:u0:8"]},binding),[6,8])

    def test_partial_training_and_holdout_isolation(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            base,collection,plan=(root/n for n in ("base","collection","plan"))
            for path in (base,collection,plan):path.mkdir()
            scenes=[{"sha256":f"g{i}","sc2_seed":i} for i in range(40)]
            write(base/"manifest.json",{"episodes":scenes})
            write(base/"episodes.json",[{"reward":1,"outcome":"win","steps":1} for _ in scenes])
            for n in ("summary.json","run.json"):write(base/n,{})
            with (base/"events.jsonl").open("w",encoding="utf-8") as out:
                for n in range(1,41):
                    out.write(json.dumps(dict(event="local_decision",episode=n,step=0,selected_id="BASE",
                                              candidates=[],baseline_actions=[1],final_actions=[1]))+"\n")
                    out.write(json.dumps(dict(event="step",episode=n,step=1,feedback={"actors":{}}))+"\n")
            hashes={name:checksum(base/name) for name in FILES}
            binding={"family":"regroup_step","actor":"u0","options":[{"action":1},{"action":3}],
                     "base_action":1,"legacy_action":3,"recommended_action":3}
            effects={"BASE":{"id":"BASE","aliases":["regroup_step:u0:1"]},
                     "test-effect":{"id":"regroup_step:u0:3","aliases":["regroup_step:u0:3"]}}
            with (plan/"menus.jsonl").open("w",encoding="utf-8") as out:
                for i,scene in enumerate(scenes):
                    menus=[binding] if i in (0,39) else []
                    out.write(json.dumps(dict(episode=i+1,scene_hash=scene["sha256"],seed=i,step=0,
                                              bindings=menus,effects=effects if menus else {"BASE":{"aliases":[]}},
                                              frozen_v1_menu=[]))+"\n")
            meta={"scope":SCOPE,"source_hashes":source_hashes(),"base_hashes":hashes,
                  "menus_sha256":checksum(plan/"menus.jsonl")}
            write(plan/"summary.json",{**meta,"kind":"local_v2_exploration_plan","base":str(base),"counts":{"nonbase_effects":2}})
            proto={**meta,"total":2};write(collection/"protocol.json",proto)
            groups=[s["sha256"] for s in scenes]
            split=root/"split.json"
            write(split,{"kind":"local_scene_split","schema_version":1,"training_group_ids":groups[:30],
                         "validation_group_ids":groups[30:]})
            policy=root/"base.json";write(policy,{"mode":"base"})
            with closing(sqlite3.connect(collection/"coverage.sqlite3")) as db:
                db.execute("CREATE TABLE meta(key TEXT,value TEXT)")
                db.execute("INSERT INTO meta VALUES('protocol',?)",(digest(proto),))
                db.execute("CREATE TABLE tasks(scene_hash,seed,step,effect,binding,state,accepted_path,result_json)")
                db.executemany("INSERT INTO tasks VALUES(?,?,0,'test-effect','regroup_step:u0:3',?,NULL,?)",[
                    (groups[0],0,'pending',None),(groups[-1],39,'accepted','intentionally invalid holdout label')])
                db.commit()
            with self.assertRaisesRegex(ValueError,"Missing training"):
                train(plan,collection,split,{"baseline":policy})
            report=train(plan,collection,split,{"baseline":policy},partial=True)
            self.assertFalse(report["training_complete"])
            self.assertEqual(report["training_registered_nonbase"],1)
            self.assertEqual(report["frozen_selectors"]["baseline"]["versus"]["v1"]["zero_rows"],30)
            self.assertEqual(report["families"]["regroup_step"]["counts"]["all_nonbase_options"],1)
            self.assertNotIn("intentionally invalid",json.dumps(report))
            out=root/"partial.json";write(out,report)
            with self.assertRaisesRegex(ValueError,"complete training"):
                freeze(out,"baseline")
            run=collection/"attempt"/"run";run.mkdir(parents=True)
            for name in FILES[:-1]:
                write(run/name,[dict(scenario_sha256=groups[0],reward=3,outcome="win")] if name=="episodes.json" else {})
            (run/"events.jsonl.gz").write_bytes(b"archived bytes")
            evidence=dict(scene_hash=groups[0],seed=0,step=0,effect="test-effect",binding="regroup_step:u0:3",
                          outcome="win",**{"return":3,"run_hashes":{n:checksum(run/n) for n in FILES[:-1]}})
            write(run/"proof.json",dict(audit=evidence,compressed_events_sha256=checksum(run/"events.jsonl.gz")))
            with closing(sqlite3.connect(collection/"coverage.sqlite3")) as db:
                db.execute("UPDATE tasks SET state='accepted',accepted_path='attempt/run',result_json=? WHERE scene_hash=?",
                           (json.dumps(evidence),groups[0]))
                db.commit()
            report=train(plan,collection,split,{"baseline":policy})
            self.assertTrue(report["training_complete"])
            self.assertEqual(report["families"]["regroup_step"]["statistics"]["recommended_vs_base"]["scene_equal_mean_gain"],2)
            self.assertEqual(report["families"]["regroup_step"]["statistics"]["recommended_vs_v1"]["scene_equal_mean_gain"],0)
            out=root/"complete.json";write(out,report)
            self.assertFalse(freeze(out,"baseline")["training_supported"])
            (run/"events.jsonl.gz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"Changed audit proof"):
                train(plan,collection,split,{"baseline":policy})


if __name__ == "__main__":
    unittest.main()
