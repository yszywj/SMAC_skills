"""Observed team objectives and retreat outcomes, independent of skill owners.

This is bounded memory, not a battle rollout. Visibility loss is never a kill;
orders are never credited as damage. Queries do not extend any deadline.
"""
import copy
import math

from .battle_pressure import enemy_pressure
from .context import distance


def contacts(context, actor):
    return tuple(e for e in context.snapshot.enemies
                 if context.can_issue_attack(actor, e)
                 and context.contact_status(actor, e) == "in_range")


def enemy_identity(context, enemy):
    raw = context.snapshot.raw_state.get("visible_enemies", {}).get(enemy, {})
    return raw.get("tag", (enemy, context.snapshot.enemies[enemy].unit_type_id))


def exposure(context, actor):
    unit = context.snapshot.allies[actor]
    risk = context.forecast_risk(actor, unit.position)
    current = set(risk.current_threats)
    # An unknown volley is a threat-count proxy, never zero known damage.
    value = sum((1. if e in current else .5) *
                (max(context.enemy_attack_damage(e, actor), 0.) / 14.
                 if context.enemy_attack_damage(e, actor) is not None else 1.)
                for e in risk.reachable_threats)
    return value + len(risk.effect_weapon_near)


class BattleIntent:
    def __init__(self, config=None):
        params = (config or {}).get("battle_intent", {})
        self.commit_h = float(params.get("commit_h", 4.))
        self.stall_h = float(params.get("stall_h", 4.))
        self.quiet_h = float(params.get("quiet_h", 6.))
        self.failed_retreats = int(params.get("failed_retreats", 2))
        self._loop = None
        self._enemy_health = {}
        self._last_progress = None
        self._last_contact = None
        self._target_until = -1
        self._target_tag = None
        self.pending = {}
        self.retreats = {}
        self.state = {"phase": "search", "engaged": False, "target": None,
                      "targets": (), "stalled_h": 0., "contact_fraction": 0.}

    def __eq__(self, other):
        return isinstance(other, BattleIntent) and self.__dict__ == other.__dict__

    @staticmethod
    def members(context):
        return tuple(a for a, u in sorted(context.snapshot.allies.items())
                     if u.alive and not context.capabilities[a].healer
                     and not context.capabilities[a].suicide_attack)

    @staticmethod
    def eta(context, actor, target):
        unit = context.snapshot.allies[actor]
        raw = context.snapshot.raw_state
        limit = raw.get("relations", {}).get(actor, {}).get(target, {}).get(
            "smac_attack_distance_limit")
        if limit is None:
            limit = raw.get("action_contract", {}).get(
                "candidate_center_distance_by_type", {}).get(unit.type_name)
        if (not isinstance(limit, (int, float)) or not math.isfinite(limit)) \
                and not context.can_issue_attack(actor, target):
            return float("inf")
        opportunity = context.projected_attack_opportunity(
            actor, target, unit.position, horizon=2 * context.H)
        if opportunity.available:
            return 0.
        speed = context.speed(actor)
        if speed is None or not math.isfinite(opportunity.center_gap):
            return float("inf")
        return opportunity.center_gap * 16. / (speed * max(context.H, 1))

    def _choose_target(self, context, members):
        candidates = []
        for target in context.snapshot.enemies:
            times = sorted(self.eta(context, a, target) for a in members)
            reachable = sum(t <= 4. for t in times)
            if reachable == 0:
                continue
            # A pressure objective must be locally attainable. No type/ID
            # preference and no reward for exposing a lone distant pursuer.
            cost = sum(min(t, 8.) for t in times[:min(3, len(times))])
            value = 2. * enemy_pressure(context, target) + \
                .3 * min(reachable, 3) - .1 * cost
            candidates.append((-value, target))
        if not candidates:
            return None
        candidates.sort()
        previous = self.state["target"]
        unit = context.snapshot.enemies.get(previous)
        if unit is not None and enemy_identity(context, previous) == self._target_tag \
                and context.snapshot.game_loop < self._target_until \
                and any(t == previous for _, t in candidates):
            return previous
        return candidates[0][1]

    def observe(self, context):
        snapshot, loop = context.snapshot, context.snapshot.game_loop
        if self._loop is not None and loop <= self._loop:
            return
        self._loop = loop
        members = self.members(context)
        active = {a: contacts(context, a) for a in members}
        in_contact = any(active.values()) or any(
            snapshot.allies[a].health_lost + snapshot.allies[a].shield_lost > 0
            for a in members)
        health = {e: (enemy_identity(context, e), u.effective_health)
                  for e, u in snapshot.enemies.items()}
        # Compare only identities observed in both frames. Shield regeneration
        # offsets damage; an invisible or newly visible enemy changes no tally.
        progress = sum(self._enemy_health[e][1] - ehp
                       for e, (tag, ehp) in health.items()
                       if e in self._enemy_health and self._enemy_health[e][0] == tag)
        self._enemy_health = health
        if self._last_progress is None or progress >= 1.:
            self._last_progress = loop
        if in_contact:
            if self._last_contact is None:
                self._last_progress = loop
            self._last_contact = loop
        engaged = self._last_contact is not None and \
            loop - self._last_contact <= self.quiet_h * max(context.H, 1)
        stalled = (loop - self._last_progress) / max(context.H, 1) if engaged else 0.
        fraction = sum(bool(v) for v in active.values()) / max(len(members), 1)
        healthy = sum(snapshot.allies[a].health_fraction > .35 for a in members)
        phase = "search"
        if engaged:
            phase = "recover" if healthy == 0 else \
                "press" if stalled >= self.stall_h and healthy >= len(members) / 2 \
                else "engage"
        target = self._choose_target(context, members) if engaged and members else None
        if target != self.state["target"] or (target is not None and
                enemy_identity(context, target) != self._target_tag):
            self._target_until = loop + round(self.commit_h * context.H)
            self._target_tag = enemy_identity(context, target) if target else None
        targets = tuple(e for e, unit in sorted(snapshot.enemies.items())
                        if target is not None and distance(
                            unit.position, snapshot.enemies[target].position) <= 6.)
        self.state = {"phase": phase, "engaged": engaged, "target": target,
                      "targets": targets, "stalled_h": stalled,
                      "contact_fraction": fraction, "observed_net_damage": progress}
        for actor in list(self.retreats):
            unit = snapshot.allies.get(actor)
            if unit is None or not unit.alive or unit.tag != self.retreats[actor]["tag"] \
                    or not engaged:
                self.retreats.pop(actor, None)
        for actor, pending in self.pending.items():
            unit = snapshot.allies.get(actor)
            if unit is None or not unit.alive or unit.tag != pending["tag"]:
                continue
            elapsed = loop - pending["loop"]
            if elapsed <= 0 or elapsed > 2 * max(context.H, 1):
                continue
            fired = unit.weapon_cooldown_loops > max(
                0., pending["cooldown"] - elapsed) + .5
            if fired:
                self.retreats.pop(actor, None)
                continue
            if not pending["retreat"]:
                continue
            record = self.retreats.setdefault(actor, {
                "tag": unit.tag, "failures": 0, "attempts": 0,
                "start_loop": pending["loop"]})
            record["attempts"] += 1
            improved = exposure(context, actor) < pending["exposure"] - .05
            record["failures"] = 0 if improved else record["failures"] + 1
            record["last_loop"] = loop
        self.pending = {}

    def commit(self, context, actions, plans):
        by_actor = {a: p for p in plans for a in p.controlled_actors}
        for actor, action in actions.items():
            unit = context.snapshot.allies.get(actor)
            if unit is None or not unit.alive:
                continue
            semantic = context.decode(actor, action)
            plan = by_actor.get(actor)
            retreat = bool(plan and plan.primary_skill == "disengage_reenter"
                           and plan.commitment.phase == "ESCAPE")
            if semantic.kind == "move" and not retreat:
                current = context.risk_at(actor, unit.position)
                enemies = tuple(current.threats) + tuple(current.effect_weapon_near)
                destination = context.predict_position(actor, action).position
                # Preserve evidence across G3/G6/other skill labels too.
                retreat = bool(enemies) and min(distance(destination,
                    context.snapshot.enemies[e].position) for e in enemies) > min(
                    distance(unit.position, context.snapshot.enemies[e].position)
                    for e in enemies) + .1
            self.pending[actor] = {
                "tag": unit.tag, "loop": context.snapshot.game_loop,
                "kind": semantic.kind, "retreat": retreat,
                "exposure": exposure(context, actor),
                "cooldown": unit.weapon_cooldown_loops}

    def target_for(self, context, actor):
        if actor not in self.members(context) or not self.state["engaged"]:
            return None
        actual = contacts(context, actor)
        preferred = self.state["target"]
        if actual:
            return preferred if preferred in actual else min(actual, key=lambda e: (
                -enemy_pressure(context, e), e))
        candidates = [e for e in self.state["targets"] if e in context.snapshot.enemies
                      and self.eta(context, actor, e) <= 6.]
        chosen = min(candidates, key=lambda e: (self.eta(context, actor, e),
                     e != preferred, e)) if candidates else None
        local = [e for e in context.snapshot.enemies
                 if self.eta(context, actor, e) <= 2.
                 and any(e in contacts(context, a) for a in self.members(context)
                         if a != actor)]
        if local:
            nearest = min(local, key=lambda e: (self.eta(context, actor, e), e))
            if chosen is None or self.eta(context, actor, chosen) > \
                    self.eta(context, actor, nearest) + 1.:
                return nearest
        return chosen

    def exhausted(self, context, actor):
        record = self.retreats.get(actor)
        unit = context.snapshot.allies.get(actor)
        return bool(record and unit and unit.alive and unit.tag == record["tag"]
                    and record["failures"] >= self.failed_retreats
                    and context.snapshot.game_loop - record["last_loop"]
                    <= self.quiet_h * max(context.H, 1))

    def summary(self):
        return {**copy.deepcopy(self.state),
                "retreat_failures": {a: r["failures"] for a, r in self.retreats.items()}}
