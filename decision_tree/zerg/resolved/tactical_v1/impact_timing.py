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
    """Cumulative kill assessment for one COMPLETE final action group.

    Returned concepts are strictly separated:
      damage_sufficient        — known window-qualified fire (from the
                                 members' FINAL actions, deduplicated) is
                                 enough for the target's remaining EHP;
      contributors             — the members that actually contributed;
      kill_after_loops_lower   — earliest possible kill time (impact lower
                                 bounds); NEVER a basis for a strong duty;
      kill_after_loops_upper   — conservative completion time supported by
                                 applicable data; currently always None
                                 (no wind-up/projectile data exists), so no
                                 caller may claim a proven-timely kill;
      timing_status            — unknown | estimated_lower_bound |
                                 insufficient_fire | unknown_damage |
                                 invalid_target;
      evidence                 — which catalog fields were used.

    Members come from the final action list: alive, legal, pointing at the
    same valid target, in range with a window shot. One shot per member;
    no assumed re-fire inside the window.
    """
    enemy = context.snapshot.enemies.get(enemy_id)
    if enemy is None:
        return {"damage_sufficient": False, "contributors": (),
                "kill_after_loops_lower": None, "kill_after_loops_upper": None,
                "kill_after_loops": None, "timing_status": "invalid_target",
                "evidence": "target_missing"}
    remaining = enemy.effective_health if remaining_hp is None \
        else max(0., float(remaining_hp))
    contributions = []
    for actor in dict.fromkeys(members):                      # dedup sources
        unit = context.snapshot.allies.get(actor)
        if unit is None or not unit.alive:
            continue
        estimate = context.fire_estimate(actor, enemy_id)
        if not context.can_issue_attack(actor, enemy_id) \
                or context.contact_status(actor, enemy_id) != "in_range" \
                or estimate.window_shots < 1:
            continue                      # not an executable contribution
        if estimate.damage_lo is None:
            return {"damage_sufficient": False,
                    "contributors": (), "kill_after_loops_lower": None,
                    "kill_after_loops_upper": None, "kill_after_loops": None,
                    "timing_status": "unknown_damage",
                    "evidence": "window_shot_without_known_damage"}
        contributions.append((attack_impact_timing(context, actor, enemy_id)
                              .impact_after_loops_lower, estimate.damage_lo,
                              actor))
    contributions.sort()
    cumulative, kill_lower = 0., None
    for lower, damage, actor in contributions:
        cumulative += max(0., damage)
        kill_lower = lower
        if cumulative >= remaining:
            break
    if cumulative < remaining:
        return {"damage_sufficient": False,
                "contributors": tuple(a for _, _, a in contributions),
                "kill_after_loops_lower": None, "kill_after_loops_upper": None,
                "kill_after_loops": None, "timing_status": "insufficient_fire",
                "cumulative": round(cumulative, 2),
                "evidence": "window_fire_below_remaining_ehp"}
    # kill_after_loops is DEFINED as the conservative completion time for
    # strong-timing judgements: without an upper bound it must be None.
    return {"damage_sufficient": True,
            "contributors": tuple(a for _, _, a in contributions),
            "kill_after_loops_lower": kill_lower, "kill_after_loops_upper": None,
            "kill_after_loops": None, "timing_status": "estimated_lower_bound",
            "evidence": "cooldown_lower_bounds;no_windup_or_projectile_data",
            "semantics": "lower bound only: damage may land later; a strong "
                         "timely-kill duty requires kill_after_loops_upper"}
