"""The action under test and the full exploration menu have separate interfaces."""
from skills.local_v1 import library as v1
from . import regroup, retreat, retarget


def _retreat_veto(payload, baseline, actor):
    """The unmodified retreat-to-attack family also belongs in full exploration."""
    unit = payload["state"]["allies"][actor]
    if (not unit.get("alive") or v1._healer(unit)
            or baseline["roles"].get(actor) not in ("recover", "retreat")):
        return None
    base = baseline["actions"][v1._id(actor, "u")]
    targets = [6 + v1._id(target, "e") for target in v1._attack_targets(payload, actor)]
    if not any(action != base for action in targets):
        return None
    return {"family":"retreat_veto", "actor":actor, "base_action":base,
            "recommended_action":base, "uses_frozen_target":True,
            "options":[{"action":a} for a in sorted({base, *targets})]}


KERNELS = {"regroup_step":regroup.analyze, "retreat_step":retreat.analyze,
           "retarget":retarget.analyze, "retreat_veto":_retreat_veto}


def analyze_binding(payload, baseline, family, actor):
    """Describe every legal alternative for a given actor and registered skill."""
    row = KERNELS[family](payload, baseline, actor)
    if row is None:
        return None
    actions = [option["action"] for option in row["options"]]
    base = baseline["actions"][v1._id(actor, "u")]
    if family in ("regroup_step", "retreat_step"):
        legal = v1._menu(payload, actor) & {2,3,4,5}
    else:
        legal = {6 + v1._id(e, "e") for e in v1._attack_targets(payload, actor)}
    if (row["actor"] != actor or row["family"] != family or row["base_action"] != base
            or len(actions) != len(set(actions)) or set(actions) != legal | {base}
            or row["recommended_action"] not in actions):
        raise ValueError(f"Incomplete or inconsistent {family} action menu for {actor}")
    return row


def enumerate_bindings(payload, baseline):
    """Enumerate every eligible family/actor and option, independent of tree scores."""
    rows = []
    for actor in sorted(payload["state"]["allies"], key=lambda key:v1._id(key, "u")):
        for family in KERNELS:
            row = analyze_binding(payload, baseline, family, actor)
            if row is not None:
                rows.append(row)
    return rows


def _patch(payload, baseline, family, actor, action):
    roles = {"regroup_step":"reposition", "retreat_step":"retreat", "retreat_veto":"attack"}
    role = roles.get(family, baseline["roles"][actor])
    return v1._candidate(payload, baseline, actor, action, family, role)


def resolve_action(payload, baseline, family, actor, action):
    """Resolve a registered option now. BASE retains original role and histories."""
    row = analyze_binding(payload, baseline, family, actor)
    if row is None or action not in {item["action"] for item in row["options"]}:
        raise ValueError(f"Unavailable binding: {family}:{actor}:{action}")
    return _patch(payload, baseline, family, actor, action)


def catalog(payload, baseline, frozen_menu):
    """Return unique one-step effects; retain all family/actor/version aliases."""
    bindings = enumerate_bindings(payload, baseline)
    effects = {"BASE":{"id":"BASE", "aliases":[]}}
    for row in bindings:
        family, actor = row["family"], row["actor"]
        for option in row["options"]:
            action = option["action"]
            patch = _patch(payload, baseline, family, actor, action)
            key = "BASE" if patch is None else patch["effect_key"]
            entry = effects.setdefault(key, {"id":f"{family}:{actor}:{action}", "aliases":[]})
            entry["aliases"].append(f"{family}:{actor}:{action}")
    # Every binding available to the frozen v1 selector must be represented.
    for old in frozen_menu:
        if old["effect_key"] not in effects or old["id"] not in effects[old["effect_key"]]["aliases"]:
            raise ValueError("Unrepresented v1 effect: " + old["id"])
    return {"bindings":bindings, "effects":effects,
            "frozen_v1_menu":[p["id"] for p in frozen_menu]}
