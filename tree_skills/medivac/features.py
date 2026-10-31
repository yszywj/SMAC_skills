"""Feature schema and extraction for the shared medivac pair scorer.

One fixed-length numeric vector per legal (healer, candidate-wounded-ally)
pair. All medivacs share a single tree. Semantic invariants (tested in
tests/test_features.py):

- No raw unit ids/tags, seeds or episode numbers as learning features.
- Missing information is explicitly flagged: no history is NOT zero health
  change, an unknown weapon is NOT "no threat".
- Renumbering allies or permuting candidate order leaves every feature
  unchanged (ties in rank features use average ranks, never id order).
- More legal candidates only means more scored rows, never a wider vector.

Schema v1 is an initial design (~33 features), not a claim of optimality.
"""

HEAL_ORDER_ABILITY = 386

FEATURE_NAMES = [
    "healer_health_ratio",
    "healer_energy_ratio",
    "healer_energy_missing",
    "healer_threatened",
    "healer_known_compatible_threats",
    "healer_threat_info_missing",
    "healer_has_confirmed_heal_order",
    "target_health_ratio",
    "target_missing_health_factor",
    "target_recent_loss_ratio",
    "target_recent_loss_missing",
    "target_is_medivac",
    "target_is_marauder",
    "target_is_marine",
    "target_max_health_norm",
    "target_threatened",
    "target_net_health_change_1",
    "target_history1_missing",
    "target_net_health_change_3",
    "target_history3_missing",
    "pair_distance",
    "pair_distance_norm",
    "pair_has_confirmed_heal_relation",
    "pair_is_previous_assignment",
    "pair_urgency_rank_of_target",
    "pair_blood_rank_of_target",
    "pair_n_legal_candidates",
    "pair_n_legal_healers_for_target",
    "ctx_n_wounded_total",
    "ctx_n_healers_with_candidates",
    "ctx_healer_per_wounded_ratio",
    "ctx_mean_urgency",
    "ctx_urgency_gap",
]

SCHEMA_VERSION = 1


def _ratio(unit):
    return unit["health"] / unit["max_health"] if unit.get("max_health") else 1.0


def _threat_flags(state, actor):
    """(threatened, n_known_compatible, info_missing) for one actor.

    Direction matters: "is this unit inside a VISIBLE ENEMY's weapon range".
    The bridge's relations carry, per enemy e:
      - in_enemy_weapon_range / enemy_weapon_range_status: THIS actor inside
        enemy e's range (enemy shoots us) — the threat direction we want;
      - in_weapon_range / weapon_range_status: OUR weapon vs enemy e —
        must NOT be read as "enemy cannot threaten us": a medivac has no
        weapon, so weapon_range_status is 'no_compatible_weapon' for every
        enemy, which says nothing about the enemy threatening the medivac.
    Unknown relations are counted as missing, never as "no threat".
    """
    relations = state.get("relations", {}).get(actor, {})
    threatened, known, missing = 0, 0, 0
    for relation in relations.values():
        if relation.get("in_enemy_weapon_range") is True:
            threatened, known = 1, known + 1
            continue
        status = relation.get("enemy_weapon_range_status")
        if status == "in_range":
            # explicit in-range but flag False: treat as known, not threatened
            known += 1
        elif status in ("out_of_range", "no_compatible_weapon"):
            known += 1
        elif status is None:
            missing += 1
    return threatened, known, 1 if missing else 0


def _average_rank(values, index):
    """1-based average rank of values[index], ascending; ties share a rank."""
    target = values[index]
    return (1 + sum(1 for v in values if v < target)
            + 0.5 * sum(1 for j, v in enumerate(values) if j != index and v == target))


def _net_change(histories, target, steps):
    """Net health-ratio change over `steps` decision steps, most recent first.

    Returns (change, sufficient). No history is missing information, never 0.
    """
    if histories is None:
        return 0.0, False
    series = histories.get(target)
    if not series or len(series) <= steps:
        return 0.0, False
    return float(series[0] - series[steps]), True


class FeatureSchema:
    def __init__(self, meta):
        if meta.get("version") != SCHEMA_VERSION:
            raise ValueError(
                f"feature schema version {meta.get('version')!r} is not "
                f"supported (expected {SCHEMA_VERSION}); refusing to load model")
        names = meta.get("names")
        if names != FEATURE_NAMES:
            raise ValueError("feature schema names do not match this code; "
                             "refusing to load model")
        self.version = SCHEMA_VERSION

    @classmethod
    def current(cls):
        return cls({"version": SCHEMA_VERSION, "names": list(FEATURE_NAMES)})

    @property
    def size(self):
        return len(FEATURE_NAMES)


def build_pair_features(healer, target_id, context, schema=None):
    """Fixed-length feature row for one legal (healer, target) pair."""
    schema = schema or FeatureSchema.current()
    state = context["state"]
    allies = state["allies"]
    healer_unit = allies[healer]
    target_unit = allies[target_id]
    urgency = context.get("urgency", {})
    target_pool = context.get("target_pool", {})
    histories = context.get("histories")

    energy = healer_unit.get("energy")
    energy_missing = not isinstance(energy, (int, float))
    h_threatened, h_known, h_missing = _threat_flags(state, healer)
    t_threatened, _, _ = _threat_flags(state, target_id)

    healer_orders = healer_unit.get("orders") or []
    confirmed_relation = any(o.get("ability_id") == HEAL_ORDER_ABILITY
                             and o.get("target_unit") == target_id
                             for o in healer_orders)

    # Legal candidate set of this healer: exactly the original edge rows.
    edges = context.get("original_edges", {})
    candidates = [t for _, t in edges.get(healer, [])]
    n_candidates = max(1, len(candidates))
    urgency_values = [urgency.get(t, 0.0) for t in candidates]
    blood_values = [_ratio(allies[t]) for t in candidates]
    idx = candidates.index(target_id)

    max_health = target_unit.get("max_health") or 1
    recent_loss = target_unit.get("health_lost_since_previous_decision")
    recent_missing = not isinstance(recent_loss, (int, float))
    change1, ok1 = _net_change(histories, target_id, 1)
    change3, ok3 = _net_change(histories, target_id, 3)

    healers_with_candidates = ({h for info in target_pool.values() for h in info}
                               or {healer})
    n_wounded = max(1, len(target_pool))
    # THIS healer-target pair's distance, not "the first healer that can
    # reach the target": different healers sit at different ranges.
    distance_info = target_pool.get(target_id, {}).get(healer)
    distance = float(distance_info["distance"]) if distance_info else 0.0

    type_name = str(target_unit.get("type", "")).lower()
    values = [
        _ratio(healer_unit),
        (energy / 200.0) if not energy_missing else 0.0,
        1.0 if energy_missing else 0.0,
        float(h_threatened),
        float(h_known),
        float(h_missing),
        1.0 if any(o.get("ability_id") == HEAL_ORDER_ABILITY for o in healer_orders) else 0.0,
        _ratio(target_unit),
        max(0.0, (max_health - target_unit["health"]) / max_health),
        (min(float(recent_loss) / max_health, 0.5) if not recent_missing else 0.0),
        1.0 if recent_missing else 0.0,
        1.0 if type_name.startswith("medivac") else 0.0,
        1.0 if "marauder" in type_name else 0.0,
        1.0 if "marine" in type_name else 0.0,
        min(max_health / 60.0, 2.0),
        float(t_threatened),
        change1,
        0.0 if ok1 else 1.0,
        change3,
        0.0 if ok3 else 1.0,
        distance,
        min(distance / 20.0, 2.0),
        1.0 if confirmed_relation else 0.0,
        1.0 if context.get("previous_assignments", {}).get(healer) == target_id else 0.0,
        _average_rank(urgency_values, idx) / n_candidates,
        _average_rank(blood_values, idx) / n_candidates,
        float(len(candidates)),
        float(len(target_pool.get(target_id, {})) or 1),
        float(len(target_pool)),
        float(len(healers_with_candidates)),
        len(healers_with_candidates) / n_wounded,
        (sum(urgency.values()) / n_wounded) if urgency else 0.0,
        (max(urgency.values()) - urgency.get(target_id, 0.0)) if urgency else 0.0,
    ]
    if len(values) != schema.size:
        raise AssertionError("feature vector length drifted from schema")
    return values
