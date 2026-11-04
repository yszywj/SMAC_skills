"""Closed JSON contracts and dependency-free gain-tree inference.

The regression target is candidate episode return minus base episode return.
No outcomes are read here, and nonpositive predictions retain the entire base
choice (which is not the same instruction as KEEP_HEAL).
"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .candidates import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, validate_candidate
from .validate import validate_tree

POLICY_KIND = "paired_return_selector"


def fingerprint(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False,
                         default=lambda v: sorted(v) if isinstance(v, (set, frozenset)) else str(v))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def runtime_source_hashes():
    """Freeze deployed inference code, excluding training tools and tests."""
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))}


def comparison_contract_hash(base_tree, candidate, runtime_sources=None):
    return fingerprint({"policy_kind": POLICY_KIND, "base_tree": base_tree,
                        "candidate": validate_candidate(candidate),
                        "runtime_sources": runtime_source_hashes() if runtime_sources is None else runtime_sources})


def validate_model(document, contract_hash):
    allowed = {"schema_version", "model_kind", "feature_schema_version", "feature_names",
               "comparison_contract_hash", "root", "training"}
    required = allowed - {"training"}
    if not isinstance(document, dict) or not required <= set(document) or set(document) - allowed:
        raise ValueError("invalid gain model fields")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1 or document["model_kind"] != "gain_regression_tree":
        raise ValueError("unsupported gain model schema")
    if type(document["feature_schema_version"]) is not int or document["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ValueError("unsupported return feature schema")
    if document["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("gain model feature_names do not match runtime schema")
    if document["comparison_contract_hash"] != contract_hash:
        raise ValueError("gain model comparison contract mismatch")
    if "training" in document and not isinstance(document["training"], dict):
        raise ValueError("model training metadata must be an object")
    seen = set()

    def visit(node, depth=0):
        if not isinstance(node, dict) or depth > 32 or len(seen) >= 255:
            raise ValueError("gain tree must have at most 255 nodes and depth 32")
        identifier = node.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("gain node IDs must be nonempty and unique")
        seen.add(identifier)
        if "gain" in node:
            if set(node) != {"id", "gain", "n_samples"}:
                raise ValueError("invalid gain leaf fields")
            if isinstance(node["gain"], bool) or not isinstance(node["gain"], (int, float)) or not math.isfinite(node["gain"]):
                raise ValueError("leaf gain must be finite")
            if type(node["n_samples"]) is not int or node["n_samples"] < 1:
                raise ValueError("leaf n_samples must be a positive integer")
            return
        if set(node) != {"id", "feature", "op", "threshold", "yes", "no"}:
            raise ValueError("invalid gain split fields")
        if node["feature"] not in FEATURE_NAMES or node["op"] != "<=":
            raise ValueError("unsupported gain feature/operator")
        if isinstance(node["threshold"], bool) or not isinstance(node["threshold"], (int, float)) or not math.isfinite(node["threshold"]):
            raise ValueError("gain threshold must be finite")
        visit(node["yes"], depth + 1)
        visit(node["no"], depth + 1)

    visit(document["root"])
    return {"node_count": len(seen)}


def validate_policy(document):
    fields = {"schema_version", "version", "policy_kind", "base_tree", "candidate", "selection"}
    if not isinstance(document, dict) or set(document) != fields:
        raise ValueError("invalid paired return policy fields")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1 or document["policy_kind"] != POLICY_KIND:
        raise ValueError("unsupported paired return policy schema")
    if not isinstance(document["version"], str) or not document["version"]:
        raise ValueError("policy version must be a nonempty string")
    info = validate_tree(document["base_tree"])
    candidate = validate_candidate(document["candidate"])
    selection = document["selection"]
    if not isinstance(selection, dict) or set(selection) - {"mode", "model"} or selection.get("mode") not in ("base", "candidate", "model"):
        raise ValueError("selection mode must be base/candidate/model")
    sources = runtime_source_hashes()
    contract_hash = comparison_contract_hash(document["base_tree"], candidate, sources)
    if selection["mode"] == "model":
        validate_model(selection.get("model"), contract_hash)
    elif "model" in selection:
        raise ValueError("only model selection mode accepts a model")
    return {**info, "candidate": candidate, "comparison_contract_hash": contract_hash,
            "runtime_source_hashes": sources,
            "selection": deepcopy(selection)}


def predict_gain(document, features):
    node, path = document["root"], []
    while "gain" not in node:
        value = features[node["feature"]]
        result = value <= node["threshold"]
        path.append({"node": node["id"], "feature": node["feature"], "value": value,
                     "op": "<=", "threshold": node["threshold"], "result": result})
        node = node["yes"] if result else node["no"]
    path.append({"node": node["id"], "gain": node["gain"], "n_samples": node["n_samples"]})
    return float(node["gain"]), path


def select(selection, features):
    """Return a choice without mutating candidates, memories or reservations."""
    if selection["mode"] == "model":
        gain, path = predict_gain(selection["model"], features)
        return ("CANDIDATE" if gain > 0 else "BASE"), gain, path
    return ("CANDIDATE" if selection["mode"] == "candidate" else "BASE"), None, []
