"""G3 CombatReposition: in-cycle repositioning between attacks.

One skill proposes attack-hold (via G1), sidestep, short retreat and pursuit
candidates from a single comparison — never four separate controllers. The
decision uses the control window (loops until this unit's next meaningful
action), the weapon cooldown, real distance and the opportunity cost of the
damage foregone while moving. Kiting is the RANGED default only when the
window math favours it; melee compares the cost of leaving the contact face.
"""
from typing import Dict, List

from ..context import Context, STOP, distance
from ..contracts import Commitment, EffectClaim, HEURISTIC, ResourceClaims
from .common import best_move_away, best_move_toward, make_plan, move_progress, threats_reaching


class CombatRepositionSkill:
    name = "combat_reposition"

    def __init__(self, config: Dict):
        self.config = config
        params = config.get("combat_reposition", {})
        self.retreat_health_fraction = params.get("retreat_health_fraction", 0.5)
        self.min_window_loops = params.get("min_window_loops", 8)
        self.max_retreat_steps = params.get("max_retreat_steps", 8)

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        for actor, unit in sorted(context.snapshot.allies.items()):
            if not unit.alive:
                continue
            capability = context.capabilities[actor]
            if capability.healer or capability.suicide_attack:
                continue
            attackable = context.attackable_enemies(actor)
            if not attackable:
                continue  # G2 owns out-of-contact units
            telemetry.count(self.name, "applicable")
            window = max(context.snapshot.step_mul, 1)
            cooldown = unit.weapon_cooldown_loops
            threats = context.threats_to(actor)
            # Damage foregone if this unit moves this step.
            best_target = attackable[0]
            damage, evidence, confidence = context.expected_damage(actor, best_target)
            opportunity = damage if cooldown <= window else 0.0
            under_fire = unit.health_lost + unit.shield_lost > 0 and threats
            # Ranged kite / sidestep during cooldown, when the window is long
            # enough that moving does not waste a shot.
            if capability.uses_normal_kiting and threats \
                    and cooldown >= self.min_window_loops \
                    and (under_fire or unit.health_fraction <= self.retreat_health_fraction):
                move = best_move_away(context, actor, threats)
                if move is not None:
                    destinations = context.move_destinations(actor)
                    threat_delta = (len(threats_reaching(context, destinations[move]))
                                    - len(threats))
                    plans.append(make_plan(
                        self.name, {actor: move}, best_target, {actor: move},
                        [EffectClaim("exposure", actor, float(-threat_delta),
                                     window, HEURISTIC, "kite_threat_reduction", 0.6),
                         EffectClaim("position", actor,
                                     move_progress(context, actor, move,
                                                   context.snapshot.enemies[best_target].position),
                                     window, HEURISTIC, "kite_reposition", 0.6)],
                        ResourceClaims(controls=(actor,), space=(f"{actor}@{move}",)),
                        Commitment("repositioning", 0, self.max_retreat_steps,
                                   interrupt_on=("weapon_ready", "danger_change")),
                        {"move_opportunity_cost": opportunity,
                         "raw_bonus": (0.5 if unit.health_fraction
                                       <= self.retreat_health_fraction else 0.0)},
                        snapshot_id, fingerprint,
                        rationale=f"kite during {cooldown:.0f}-loop cooldown"))
            # Melee: leaving the contact face is expensive; only propose a
            # positional shuffle that keeps contact-range distance.
            if capability.combat_class == "melee" and under_fire \
                    and unit.health_fraction <= self.retreat_health_fraction:
                move = best_move_away(context, actor, threats)
                if move is not None:
                    destinations = context.move_destinations(actor)
                    after = distance(destinations[move],
                                     context.snapshot.enemies[best_target].position)
                    if after <= (context.snapshot.relations[actor][best_target]
                                 .native_weapon_range or 2.0) + 1.0:
                        plans.append(make_plan(
                            self.name, {actor: move}, best_target, {actor: move},
                            [EffectClaim("exposure", actor, 0.0, window,
                                         HEURISTIC, "melee_in_contact_shuffle", 0.5)],
                            ResourceClaims(controls=(actor,),
                                           space=(f"{actor}@{move}",)),
                            Commitment("repositioning", 0, 2,
                                       interrupt_on=("illegal", "danger_change")),
                            {"move_opportunity_cost": opportunity},
                            snapshot_id, fingerprint,
                            rationale="melee shuffle keeps contact"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans
