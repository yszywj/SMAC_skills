"""Observed encounter boundaries and bounded rally completion.

An enemy disappearing for one observation is not a new battle. A rally can
be used once per encounter; another is armed only after real contact/damage
followed by a sustained quiet period. All mutations happen in observation or
commit, never while skills enumerate candidates.
"""
from .context import distance


def combat_members(context):
    return tuple(a for a, unit in sorted(context.snapshot.allies.items())
                 if unit.alive and not context.capabilities[a].flying
                 and not context.capabilities[a].healer
                 and not context.capabilities[a].suicide_attack)


def unsafe_to_rally(context, actors) -> bool:
    for actor in actors:
        unit = context.snapshot.allies.get(actor)
        if unit is None or not unit.alive:
            return True
        if unit.health_lost > 0 or unit.shield_lost > 0:
            return True
        risk = context.risk_at(actor, unit.position)
        if risk.threats or risk.effect_weapon_near:
            return True
        if any(context.can_issue_attack(actor, enemy_id)
               and context.contact_status(actor, enemy_id) == "in_range"
               for enemy_id in context.snapshot.enemies):
            return True
    return False


def rally_readiness(context, members, anchor, ready_radius, positions=None):
    """Geometry of a frozen group/anchor; usable by the final-plan scorer.

    positions is an optional actor->coordinate map, never command endpoints
    claimed as observed facts. A radius band leaves room for unit bodies.
    """
    coords = [(positions or {}).get(a, context.snapshot.allies[a].position)
              for a in members if a in context.snapshot.allies
              and context.snapshot.allies[a].alive]
    if not coords:
        return {"fraction": 0.0, "gap": 0.0, "diameter": 0.0}
    distances = [distance(point, anchor) for point in coords]
    return {
        "fraction": sum(d <= ready_radius for d in distances) / len(coords),
        "gap": sum(max(d - ready_radius, 0.0) for d in distances) / len(coords),
        "diameter": max(distance(a, b) for a in coords for b in coords),
    }


class EncounterTracker:
    def __init__(self, config=None):
        params = (config or {}).get("precombat_rally", {})
        self.rearm_quiet_h = float(params.get("rearm_quiet_h", 4.0))
        self.current_id = 1
        self.closed_ids = {}
        self.had_contact = False
        self.current_hostile = False
        self.quiet_since = None
        self.quiet_budget = None
        self._observed = None
        self.started_loop = None

    def __eq__(self, other):
        return isinstance(other, EncounterTracker) and self.__dict__ == other.__dict__

    def observe(self, context):
        snapshot = context.snapshot
        key = (snapshot.episode, snapshot.snapshot_id, snapshot.game_loop)
        if key == self._observed:
            return
        self._observed = key
        if self.started_loop is None:
            self.started_loop = snapshot.game_loop
        self.current_hostile = unsafe_to_rally(context, combat_members(context))
        if self.current_hostile:
            self.had_contact = True
            self.quiet_since = None
            self.quiet_budget = None
        elif self.had_contact:
            if self.quiet_since is None:
                self.quiet_since = snapshot.game_loop
                self.quiet_budget = max(1, int(round(self.rearm_quiet_h * context.H)))
            elif snapshot.game_loop - self.quiet_since >= self.quiet_budget:
                self.current_id += 1
                self.started_loop = snapshot.game_loop
                self.had_contact = False
                self.quiet_since = None
                self.quiet_budget = None

    def can_rally(self):
        # Threat starts covered assembly; it does not veto organization.
        return self.current_id not in self.closed_ids

    def close(self, encounter_id, reason):
        if encounter_id is not None:
            self.closed_ids.setdefault(encounter_id, reason)

    def task_outcome(self, context, instance):
        """Observation-phase outcome, before total/phase deadline handling.

        Partial engagement no longer terminates the organization: engaged
        members keep fighting in their own skills while the remaining task
        continues as COVERED_JOIN. The task ends only when EVERY ordinary
        member is engaged, when executable readiness is met, or on the
        existing no-progress/identity rules.
        """
        if instance.skill != "precombat_rally":
            return None
        data = instance.extra
        if data.get("encounter_id") != self.current_id:
            return "encounter_changed"
        if any(a not in context.snapshot.allies
               or not context.snapshot.allies[a].alive for a in instance.actors):
            return "member_lost"
        from .deployment import ENGAGE, deployment_target
        ordinary = tuple(a for a in instance.actors
                         if not context.capabilities[a].healer
                         and not context.capabilities[a].suicide_attack)
        group = {'id': data.get('squad_id', instance.instance_id),
                 'members': ordinary, 'ordinary': ordinary}
        target = deployment_target(context, group, instance)
        if target['phase'] == ENGAGE:
            # Geometry releases opening control; this is NOT a claim that
            # every rear weapon is already firing. G2/G6 own that advance.
            return 'rally_ready'
        engaged = target['engaged']
        pending = tuple(a for a in ordinary if a not in engaged)
        anchor = target['anchor']
        if anchor is None:
            return 'no_safe_anchor'
        changed = (tuple(data['anchor']) != tuple(anchor)
                   or data.get('deployment_phase') != target['phase']
                   or data.get('front_actor') != target['front_actor'])
        data.update(anchor=list(anchor), waypoint=list(anchor),
                    deployment_phase=target['phase'], front_actor=target['front_actor'])
        quality = rally_readiness(context, pending or ordinary, anchor,
                                  data['ready_radius'])
        loop = context.snapshot.game_loop
        if changed:
            # New target coordinates are not observed movement progress and
            # must not renew either the stall clock or the original deadline.
            data['best_gap'] = quality['gap']
        elif quality["gap"] < data.get("best_gap", float("inf")) - 0.05:
            data["best_gap"] = quality["gap"]
            data["rally_last_progress_loop"] = loop
            instance.last_progress_loop = loop
        last_progress = data.get("rally_last_progress_loop", instance.start_loop)
        if loop - last_progress >= data["no_progress_loops"]:
            return "no_progress"
        return None
