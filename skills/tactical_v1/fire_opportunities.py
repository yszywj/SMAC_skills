"""Final-command budgets for ordinary FUTURE fire, never actual deaths.

Only an explicit legal ATTACK with known current-window damage can reserve
target health. Cooldown is a lower bound, not verified projectile arrival:
without arrival data we apply the reservation only beyond the current H.
Unknown fire cannot consume another actor's opportunity. A depleted expected
budget retains a small contingency opportunity for missed/delayed shots; this
is an engineering credit, NOT an estimated hit probability.
"""
import math
from dataclasses import dataclass


CONTINGENCY_CREDIT = .15


@dataclass(frozen=True)
class ReservedShot:
    actor: str
    target: str
    amount: float
    after_h: float
    arrival_known: bool


class FinalActionBudget:
    def __init__(self, context, actions):
        self.context = context
        self.actions = dict(actions)
        self.shots = []
        H = max(context.H, 1)
        for actor, action in sorted(actions.items()):
            unit = context.snapshot.allies.get(actor)
            if unit is None or not unit.alive:
                continue
            capability = context.capabilities[actor]
            if capability.healer or capability.suicide_attack:
                continue
            semantic = context.decode(actor, action)
            target = semantic.target
            if semantic.kind != 'attack' or target not in context.snapshot.enemies \
                    or not context.can_issue_attack(actor, target):
                continue
            estimate = context.fire_estimate(actor, target)
            lower = estimate.damage_lo
            if estimate.window_shots < 1 or lower is None \
                    or not math.isfinite(lower) or lower <= 0:
                continue
            arrival = estimate.effective_after_loops
            timing_known = bool(estimate.delay_known and arrival is not None)
            # Never invent an exact missile delay or suppress a same-H shot.
            after_h = max(1., float(arrival or 0.) / H)
            self.shots.append(ReservedShot(actor, target, float(lower), after_h, timing_known))
        self.shots = tuple(self.shots)

    def remaining_fraction(self, actor, target, eta_h):
        enemy = self.context.snapshot.enemies.get(target)
        if enemy is None or not math.isfinite(eta_h):
            return 0.
        # Strictly earlier OTHER sources only. Own current attack must never
        # erase itself, and simultaneous/current-window volleys stay legal.
        used = sum(s.amount for s in self.shots
                   if s.target == target and s.actor != actor and s.after_h < eta_h - 1e-6)
        return max(0., min(1., 1. - used / max(enemy.effective_health, 1.)))

    def credit(self, actor, target, eta_h):
        remaining = self.remaining_fraction(actor, target, eta_h)
        return remaining + (1. - remaining) * CONTINGENCY_CREDIT

    def future_targets(self, actor, eta_by_target):
        """Separate usable next targets from backup fire on expected kills.

        ``eta_h`` is PRIMARY-only: backup fire must not fabricate a certain
        next-shot deadline. Callers may retain contingency targets for legal
        follow-up attacks, with their separately bounded credit.
        """
        primary, contingency, credits = {}, {}, {}
        for target, eta in eta_by_target.items():
            if target not in self.context.snapshot.enemies or not math.isfinite(eta):
                continue
            remaining = self.remaining_fraction(actor, target, eta)
            (primary if remaining > 1e-9 else contingency)[target] = eta
            credits[target] = remaining + (1. - remaining) * CONTINGENCY_CREDIT
        return {'primary': primary, 'contingency': contingency, 'credits': credits,
                'eta_h': min(primary.values(), default=float('inf'))}


def final_action_budget(context, actions):
    """Snapshot-local and whole-final-action keyed; no skill/provider labels."""
    cache = getattr(context, '_final_fire_budgets', None)
    if cache is None:
        cache = context._final_fire_budgets = {}
    key = tuple(sorted(actions.items()))
    if key not in cache:
        cache[key] = FinalActionBudget(context, actions)
    return cache[key]
