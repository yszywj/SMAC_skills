"""Pure attack-timing queries: "can fire" vs "damage has landed".

A weapon's cooldown end is when the unit CAN fire, never proof that the
shot has landed: wind-up and projectile flight times are not present in the
observation contract, so they stay UNKNOWN instead of being invented from
attack periods or a flat constant. This module never mutates state and is
cached per (context, actor, enemy).

Well-founded statements only:
  ready_after_loops         — known: current cooldown (>= 0);
  impact_after_loops_lower  — known: damage cannot land before the source
                              can fire, so cooldown IS a valid lower bound;
  impact_after_loops_upper  — None (no wind-up/projectile data exists in
                              this adapter); never guessed;
  timing_status             — "unknown" (impact spread unknown) or
                              "estimated" (lower bound only, upper None);
  evidence                  — states exactly which catalog fields were used.

Consumers must treat a missing upper bound as "cannot PROVE timely
interception": ordinary shot candidates stay valid, but no hard
"proven-safe intercept" duty may be derived from timing alone.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ImpactTiming:
    ready_after_loops: float
    impact_after_loops_lower: float            # never None: cooldown bound
    impact_after_loops_upper: float = None     # unknown wind-up/flight
    timing_status: str = "unknown"             # unknown | estimated
    evidence: str = "weapon_cooldown_lower_bound;no_windup_or_projectile_data"

    @property
    def conservative_kill_loops(self):
        """Best-supported conservative kill-time, or None when unknown."""
        return self.impact_after_loops_upper


def attack_impact_timing(context, actor, enemy_id):
    """Timing of one source's next shot, relative to the current game_loop."""
    cache = getattr(context, "_impact_timing_cache", None)
    if cache is None:
        cache = context._impact_timing_cache = {}
    key = (actor, enemy_id)
    if key in cache:
        return cache[key]
    unit = context.snapshot.allies.get(actor)
    estimate = context.fire_estimate(actor, enemy_id)
    ready = max(0., float(unit.weapon_cooldown_loops)) if unit else 0.
    timing = ImpactTiming(
        ready_after_loops=ready,
        impact_after_loops_lower=ready,
        impact_after_loops_upper=None,
        timing_status="estimated" if estimate.window_shots >= 1 else "unknown",
        evidence=("weapon_cooldown_lower_bound;window_qualified_shot"
                  if estimate.window_shots >= 1
                  else "weapon_cooldown_lower_bound;no_window_shot"))
    cache[key] = timing
    return timing


def joint_kill_time(context, members, enemy_id, remaining_hp=None):
    """Cumulative kill time for one COMPLETE final action group.

    Members are the actors whose FINAL actions actually attack ``enemy_id``.
    Damage is accumulated in impact-lower-bound order; the kill time is the
    readiness of the member whose cumulative known window damage first
    reaches the target's remaining effective health. Only ONE shot per
    member is counted (no assumed re-fire inside the window), members
    without window damage contribute nothing, and any missing damage data
    downgrades the result to unknown.
    """
    enemy = context.snapshot.enemies.get(enemy_id)
    if enemy is None:
        return {"kill_after_loops": None, "status": "invalid_target",
                "members": tuple(members)}
    remaining = enemy.effective_health if remaining_hp is None \
        else max(0., float(remaining_hp))
    contributions = []
    for actor in members:
        estimate = context.fire_estimate(actor, enemy_id)
        if not context.can_issue_attack(actor, enemy_id) \
                or context.contact_status(actor, enemy_id) != "in_range" \
                or estimate.window_shots < 1:
            continue                      # not an executable contribution
        if estimate.damage_lo is None:
            return {"kill_after_loops": None, "status": "unknown_damage",
                    "members": tuple(members)}
        contributions.append((attack_impact_timing(context, actor, enemy_id)
                              .impact_after_loops_lower, estimate.damage_lo,
                              actor))
    contributions.sort()
    cumulative, kill_loops = 0., None
    for lower, damage, actor in contributions:
        cumulative += max(0., damage)
        kill_loops = lower
        if cumulative >= remaining:
            break
    if cumulative < remaining:
        return {"kill_after_loops": None, "status": "insufficient_fire",
                "members": tuple(members), "cumulative": round(cumulative, 2)}
    return {"kill_after_loops": kill_loops, "status": "estimated_lower_bound",
            "members": tuple(members), "cumulative": round(cumulative, 2),
            "semantics": "lower bound only; landing time unknown — not a "
                         "guarantee that the kill happens before exposure"}
