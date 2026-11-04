"""Frozen execution context shared by dataset, training and model deployment.

The candidate comparison contract freezes the medical algorithm. This additional
context binds return labels to the ground controller, simulator and settings that
produced them. Scenario manifests are deliberately excluded to allow held-out
scenarios under that same execution protocol.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import posixpath
import re


ROOT = Path(__file__).resolve().parents[1]
CONTEXT_FIELDS = (
    "skill_source_hashes", "env_args", "skill_parameters", "versions",
    "observation_protocol", "action_extensions", "method", "skill_version",
    "native_teacher_control", "medivac_policy", "medical_patient_coordination",
)


def context_hash(context):
    return hashlib.sha256(json.dumps(context, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _path_key(value):
    return posixpath.normpath(str(value).replace("\\", "/"))


def _source_hashes(values):
    if not isinstance(values, dict) or not values:
        raise ValueError("execution context needs nonempty source hashes")
    result = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name or not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("invalid execution source hash")
        key = _path_key(name)
        if key in result:
            raise ValueError("duplicate normalized execution source path")
        result[key] = value
    return result


def _is_policy_source(key, declared):
    # Runner writes source keys relative to SMAC_agent, including '..' when
    # needed. Resolve those paths against the same root; never match basename
    # alone, since a distinct JSON dependency must remain bound.
    if key == declared:
        return True
    return _path_key((ROOT / key).resolve()).casefold() == _path_key(Path(declared).resolve()).casefold()


def experiment_context(metadata):
    """Normalize one run.json, excluding only its declared policy JSON source."""
    if not isinstance(metadata, dict):
        raise ValueError("execution metadata must be an object")
    missing = [key for key in CONTEXT_FIELDS if key not in metadata or metadata[key] is None]
    if missing:
        raise ValueError("missing execution context fields: " + ", ".join(missing))
    tree = metadata.get("medivac_tree_metadata")
    declared = tree.get("tree_path") if isinstance(tree, dict) else None
    if not isinstance(declared, str) or not declared.lower().endswith(".json"):
        raise ValueError("execution context needs an explicit policy JSON path")
    declared = _path_key(declared)
    sources = _source_hashes(metadata.get("source_hashes"))
    excluded = [key for key in sources if _is_policy_source(key, declared)]
    if len(excluded) != 1:
        raise ValueError("execution sources must contain exactly one declared policy JSON")
    context = {"schema_version": 1,
               "source_hashes": {key: value for key, value in sources.items() if key not in excluded}}
    context.update({key: deepcopy(metadata[key]) for key in CONTEXT_FIELDS})
    context["skill_source_hashes"] = _source_hashes(metadata["skill_source_hashes"])
    context.update({key: deepcopy(value) for key, value in metadata.items() if key.startswith("unit_value_")})
    if not context["source_hashes"]:
        raise ValueError("execution context contains no non-policy sources")
    context_hash(context)  # Fail closed on non-JSON or nonfinite metadata.
    return context


def validate_context(context, expected_hash):
    if (not isinstance(context, dict) or type(context.get("schema_version")) is not int
            or context.get("schema_version") != 1
            or not set(CONTEXT_FIELDS).issubset(context)):
        raise ValueError("missing or unsupported experiment context")
    _source_hashes(context.get("source_hashes"))
    _source_hashes(context.get("skill_source_hashes"))
    if context_hash(context) != expected_hash:
        raise ValueError("experiment context hash mismatch")
    return context


def validate_deployment(metadata, model):
    """Call before starting SC2, after the runner has assembled its metadata."""
    training = model.get("training", {}) if isinstance(model, dict) else {}
    expected = validate_context(training.get("experiment_context"), training.get("experiment_context_hash"))
    actual = experiment_context(metadata)
    if actual != expected:
        raise ValueError("trained return model execution context differs from this run; freeze the original protocol or collect new paired data")
    return {"experiment_context_hash": context_hash(actual), "matched": True}
