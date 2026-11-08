"""Keep the complete action catalogue separate from deployment conditions."""
from copy import deepcopy

from skills.local_v2 import library as v2
from . import retarget


def enumerate_bindings(payload, baseline):
    rows = v2.enumerate_bindings(payload, baseline)
    for index, row in enumerate(rows):
        if row["family"] == "retarget":
            rows[index] = retarget.analyze(payload, baseline, row["actor"])
        elif row["family"] == "regroup_step":
            # Restore the v1 direction. The v2 distance tradeoff lost in the
            # interrupted study. Keep its geometry as observable diagnostics.
            row["recommended_action"] = row["legacy_action"]
            row["reason"] = "legacy_centroid"
    return rows


def catalog(payload, baseline):
    bindings = enumerate_bindings(payload, baseline)
    effects = {"BASE": {"id": "BASE", "aliases": []}}
    for row in bindings:
        family, actor = row["family"], row["actor"]
        for option in row["options"]:
            action = option["action"]
            patch = v2._patch(payload, baseline, family, actor, action)
            effect = "BASE" if patch is None else patch["effect_key"]
            cid = f"{family}:{actor}:{action}"
            effects.setdefault(effect, {"id": cid, "aliases": []})["aliases"].append(cid)
    return {"bindings": bindings, "effects": effects}


def recommended_candidate(payload, baseline, row):
    patch = v2._patch(payload, baseline, row["family"], row["actor"], row["recommended_action"])
    return deepcopy(patch)
