"""Fit and apply small win-first corrections to a frozen conditional policy."""
from copy import deepcopy

from decision_tree.local.models import fit_gain, predict, validate_model
from decision_tree.local.runtime import digest
from skills.local_v1 import FEATURE_NAMES
from skills.local_v2.library import _patch


def validate_layers(layers):
    if not isinstance(layers, list) or len(layers) > 8:
        raise ValueError("Expected at most eight frozen correction layers")
    for layer in layers:
        if set(layer) != {"reference_sha256", "gain", "veto"}:
            raise ValueError("Unknown correction layer")
        for kind in ("gain", "veto"):
            models = layer[kind]
            if models is None:
                continue
            if set(models) != {"win", "return"}:
                raise ValueError("Both win and return trees required")
            for model in models.values():
                validate_model(model)
                if model["feature_names"] != list(FEATURE_NAMES) or model["min_groups"] < 4:
                    raise ValueError("Unsupported correction features or support")


def score(models, candidate):
    if models is None:
        return None
    values = [predict(models[key], candidate["family"], candidate["features"]) for key in ("win", "return")]
    if not all(v["supported"] for v in values):
        return None
    wins, reward = (v["value"] for v in values)
    if wins < -1e-9 or (abs(wins) <= 1e-9 and reward <= 1e-9):
        return None
    return wins, reward


def choose(policy, payload, baseline, menu, selected, candidates, *, allowed_actors, exhausted):
    diagnostics = []
    if exhausted or not policy.get("corrections"):
        return selected, diagnostics
    actions = []
    for row in menu["bindings"]:
        if row["actor"] not in allowed_actors:
            continue
        for option in row["options"]:
            candidate = _patch(payload, baseline, row["family"], row["actor"], option["action"])
            if candidate is not None:
                actions.append(candidate)
    for layer in policy["corrections"]:
        choices = []
        for candidate in actions:
            if selected and candidate["effect_key"] == selected["effect_key"]:
                continue
            prediction = score(layer["gain"], candidate)
            if prediction:
                choices.append((prediction, candidate["id"], candidate))
        if selected:
            prediction = score(layer["veto"], selected)
            if prediction:
                choices.append((prediction, "BASE", None))
        if choices:
            # BASE wins a numerical tie, reducing interventions.
            best = sorted(choices, key=lambda x: (-x[0][0], -x[0][1], x[1]))[0]
            diagnostics.append({"before": selected["id"] if selected else "BASE",
                                "after": best[1], "predicted_win_gain": best[0][0],
                                "predicted_return_gain": best[0][1]})
            selected = best[2]
        else:
            diagnostics.append({"before": selected["id"] if selected else "BASE", "after": "retain"})
    return selected, diagnostics


def fit_layer(policy, measured, context_hash):
    result = {"reference_sha256": digest(policy)}
    for kind in ("gain", "veto"):
        selected = [r for r in measured if r["label_kind"] == kind]
        models = None
        if selected:
            models = {}
            for target in ("win", "return"):
                rows = []
                for row in selected:
                    branch, reference = row["branch"], row["incumbent"]
                    delta = (int(branch["outcome"] == "win") - int(reference["outcome"] == "win")
                             if target == "win" else branch["return"] - reference["return"])
                    rows.append({"group_id": row["scene_hash"], "candidate_family": row["family"],
                                 "record_id": row["id"], "features": row["features"], "gain": delta})
                models[target] = fit_gain(rows, list(FEATURE_NAMES), context_hash, max_depth=2, min_groups=4)
        result[kind] = models
    validate_layers([result])
    updated = deepcopy(policy)
    updated["corrections"] = [*updated.get("corrections", []), result]
    return updated
