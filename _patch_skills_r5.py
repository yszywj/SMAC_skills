"""Round-5 skill patches: objective declarations. Run then delete."""
import io


def patch(path, pairs):
    s = io.open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert old in s, f"{path}: MISSING\n{old[:140]}"
        s = s.replace(old, new)
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)
    print(f"patched {path}")


# ---- G2: contact_point for melee slots, range_band for ranged -------------
patch("skills/tactical_v1/skills/coordinated_engage.py", [
    ('''                    goal_kind="enemy",
                    goal_data={"target_id": target_id, "waypoint": goal},
                    memory_delta=({"kind": "create", "skill": self.name,''',
     '''                    goal_kind="enemy",
                    goal_data={"target_id": target_id, "waypoint": goal,
                               "objective": _objective(context, actor,
                                                       target_id, goal)},
                    memory_delta=({"kind": "create", "skill": self.name,'''),
    ('''    def _range_band(self, context: Context, actor: str, target_id: str):
        relation = context.snapshot.relations[actor].get(target_id)
        if relation is not None and relation.native_weapon_range:
            return max(relation.native_weapon_range - 0.5, 0.5)
        return None''',
     '''    def _range_band(self, context: Context, actor: str, target_id: str):
        relation = context.snapshot.relations[actor].get(target_id)
        if relation is not None and relation.native_weapon_range:
            return max(relation.native_weapon_range - 0.5, 0.5)
        return None


def _objective(context, actor, target_id, goal):
    """2.2: melee evaluate the CONTACT SLOT; ranged evaluate the RANGE BAND
    around the target — never the enemy centre for everyone."""
    capability = context.capabilities[actor]
    if capability.combat_class == "melee":
        return {"kind": "contact_point", "position": list(goal)}
    relation = context.snapshot.relations[actor].get(target_id)
    r_max = (relation.native_weapon_range if relation is not None else None) \\
        or 5.0
    return {"kind": "range_band", "target": target_id,
            "r_min": max(r_max - 1.5, 0.0), "r_max": r_max}'''),
])

# ---- G3: range_band objective; keep ALL four move directions ---------------
patch("skills/tactical_v1/skills/combat_reposition.py", [
    ('''            provider = self.providers.get("splash_fire")
            budget = h_loops(context, 8.0)
            for label, move in sorted(by_class.items()):''',
     '''            provider = self.providers.get("splash_fire")
            budget = h_loops(context, 8.0)
            relation = context.snapshot.relations[actor][target_id]
            r_max = relation.native_weapon_range or 5.0
            # keep one candidate per class, but never fewer than the plain
            # four directions for the unified evaluator to compare (2.3)
            seen_actions = set()
            ordered = sorted(by_class.items())
            for label, move in ordered:
                seen_actions.add(move)
            for action in sorted(context.move_destinations(actor)):
                if action not in seen_actions and len(ordered) < \\
                        self.max_candidates:
                    ordered.append((by_class.get(action, "sidestep"), action))
                    seen_actions.add(action)
            for label, move in ordered:'''),
    ('''                    # the lost window shot is priced by physics only (D3)
                    {"progress": coverage * 0.3},
                    snapshot_id, fingerprint,''',
     '''                    # the lost window shot is priced by physics only
                    {"switch_cost": 0.0},
                    snapshot_id, fingerprint,'''),
    ('''                    goal_kind="enemy",
                    goal_data={"target_id": target_id},''',
     '''                    goal_kind="enemy",
                    goal_data={"target_id": target_id,
                               "objective": {
                                   "kind": "range_band", "target": target_id,
                                   "r_min": max(r_max - 1.5, 0.0)
                                   if not melee else 0.0,
                                   "r_max": r_max if not melee else 2.0}},'''),
])

# ---- G4: escape objective; reenter uses a range band -----------------------
patch("skills/tactical_v1/skills/disengage_reenter.py", [
    ('''                        Commitment(ESCAPE, 0, max_loops=total_budget),
                        {}, snapshot_id, fingerprint,
                        instance_id=instance.instance_id if instance else None,
                        memory_delta=self._delta(instance, ESCAPE, actor, context=context),
                        rationale="escape: risk improvement available"))''',
     '''                        Commitment(ESCAPE, 0, max_loops=total_budget),
                        {}, snapshot_id, fingerprint,
                        instance_id=instance.instance_id if instance else None,
                        memory_delta=self._delta(instance, ESCAPE, actor, context=context),
                        goal_kind="internal",
                        goal_data={"objective": {"kind": "escape"}},
                        rationale="escape: risk improvement available"))'''),
    ('''                    Commitment(REENTER, 0, max_loops=h_loops(
                        context, self.reenter_max_h)),
                    {"progress": move_progress(
                        context, actor, move,
                        context.snapshot.enemies[target].position)},
                    snapshot_id, fingerprint,''',
     '''                    Commitment(REENTER, 0, max_loops=h_loops(
                        context, self.reenter_max_h)),
                    {}, snapshot_id, fingerprint,'''),
    ('''                    memory_delta=self._delta(instance, REENTER, actor,
                                             goal=target,
                                             context=context),
                    rationale=f"re-enter toward {target}"))''',
     '''                    memory_delta=self._delta(instance, REENTER, actor,
                                             goal=target,
                                             context=context),
                    goal_kind="enemy",
                    goal_data={"target_id": target,
                               "objective": {"kind": "range_band",
                                             "target": target,
                                             "r_min": 0.0, "r_max": 5.0}},
                    rationale=f"re-enter toward {target}"))'''),
])

# ---- G5: merge options carry the REAL target group (2.5) -------------------
patch("skills/tactical_v1/skills/regroup.py", [
    ('''        if fighting:
            main = max(fighting, key=lambda g: (len(g), g))
            positions = [context.snapshot.allies[a].position for a in main]
            options.append(min(positions,
                               key=lambda p: (distance(unit.position, p), p)))
        else:
            group = min(pool, key=lambda g: min(
                distance(unit.position, context.snapshot.allies[a].position)
                for a in g))
            positions = [context.snapshot.allies[a].position for a in group]
            options.append(min(positions,
                               key=lambda p: (distance(unit.position, p), p)))
        closest = min(pool, key=lambda g: min(
            distance(unit.position, context.snapshot.allies[a].position)
            for a in g))
        own_centre = _centroid(context, own)
        other_centre = _centroid(context, closest)
        options.append(((own_centre[0] + other_centre[0]) / 2,
                        (own_centre[1] + other_centre[1]) / 2))
        return [o for o in options
                if distance(unit.position, o) > self.support_radius * 0.5]''',
     '''        # each option carries its REAL target group — the declared
        # group_b is exactly the group the waypoint serves (2.5)
        if fighting:
            main = max(fighting, key=lambda g: (len(g), g))
            positions = [context.snapshot.allies[a].position for a in main]
            options.append((main, min(positions,
                                      key=lambda p: (distance(unit.position,
                                                              p), p))))
        else:
            group = min(pool, key=lambda g: min(
                distance(unit.position, context.snapshot.allies[a].position)
                for a in g))
            positions = [context.snapshot.allies[a].position for a in group]
            options.append((group, min(positions,
                                       key=lambda p: (distance(unit.position,
                                                               p), p))))
        closest = min(pool, key=lambda g: min(
            distance(unit.position, context.snapshot.allies[a].position)
            for a in g))
        own_centre = _centroid(context, own)
        other_centre = _centroid(context, closest)
        options.append((closest, ((own_centre[0] + other_centre[0]) / 2,
                                  (own_centre[1] + other_centre[1]) / 2)))
        return [(g, o) for g, o in options
                if distance(unit.position, o) > self.support_radius * 0.5]'''),
    ('''            own, pairs = self._link_pairs(context, actor)
            for goal in options[:2]:          # comparable alternatives
                move = best_move_toward(context, actor, goal)
                if move is None:
                    continue
                # the REAL evaluator goal: which support link this closes
                other = min((p[1] for p in pairs),
                            key=lambda g: min(
                                distance(unit.position,
                                         context.snapshot.allies[a].position)
                                for a in g if a in context.snapshot.allies),
                            default=None)
                plans.append(make_plan(
                    self.name, {actor: move}, "merge", {actor: move},''',
     '''            own, _pairs = self._link_pairs(context, actor)
            for other, goal in options[:2]:   # real target group + waypoint
                move = best_move_toward(context, actor, goal)
                if move is None:
                    continue
                plans.append(make_plan(
                    self.name, {actor: move}, "merge", {actor: move},'''),
    ('''                    goal_kind="group_link",
                    goal_data={
                        "group_a": sorted(own),
                        "group_b": sorted(other) if other else [],
                        "support_radius": self.support_radius,
                        "scale": 4.0, "position": list(goal)},
                    task_key="|".join(("regroup", ",".join(sorted(own)),
                                       ",".join(sorted(other or [])))),''',
     '''                    goal_kind="group_link",
                    goal_data={
                        "group_a": sorted(own),
                        "group_b": sorted(other) if other else [],
                        "support_radius": self.support_radius,
                        "scale": 4.0, "position": list(goal),
                        "objective": {"kind": "support_link"}},
                    task_key="|".join((
                        "regroup",
                        ",".join(sorted(own)),
                        *sorted((",".join(sorted(other or [])),)))),'''),
])

# ---- G6: spacing for decongestion, protected_position for the colossus ----
patch("skills/tactical_v1/skills/local_formation.py", [
    ('''                    goal_kind=(
                        "role" if context.capabilities[actor].specialization == "splash_fire"
                        and guards else "position"),
                    goal_data=(
                        {"role": "colossus_protected", "actor": actor,
                         "guards": guards}
                        if context.capabilities[actor].specialization == "splash_fire"
                        and guards else {"position": list(role_goal or
                                                          unit.position)}),''',
     '''                    is_core = \\
                        context.capabilities[actor].specialization == "splash_fire" \\
                        and guards
                    goal_kind="role" if is_core else "internal",
                    goal_data=(
                        {"role": "colossus_protected", "actor": actor,
                         "guards": guards,
                         "objective": {"kind": "protected_position"}}
                        if is_core else
                        # 2.4: plain decongestion NEVER declares a point
                        # target — the physics congestion term prices it.
                        {"objective": {"kind": "spacing"}}),'''),
])

# ---- G7: waypoint objective -------------------------------------------------
patch("skills/tactical_v1/skills/cohesive_search.py", [
    ('''                    goal_kind="position",
                    goal_data={"waypoint": list(goal)},
                    instance_id=instance.instance_id if instance else None,''',
     '''                    goal_kind="position",
                    goal_data={"waypoint": list(goal),
                               "objective": {"kind": "waypoint",
                                             "position": list(goal)}},
                    instance_id=instance.instance_id if instance else None,'''),
])

# ---- S2: meeting/escort objectives; heal stays none ------------------------
patch("skills/tactical_v1/skills/heal_support.py", [
    ('''                        goal_kind="meeting",
                        goal_data={"healer": actor, "patient": patient,
                                   "position": list(meet)},''',
     '''                        goal_kind="meeting",
                        goal_data={"healer": actor, "patient": patient,
                                   "position": list(meet),
                                   "objective": {"kind": "meeting"}},'''),
    ('''                                goal_kind="position",
                                goal_data={"waypoint": list(goal)},
                                rationale="follow effective friendly squad"))''',
     '''                                goal_kind="position",
                                goal_data={"waypoint": list(goal),
                                           "objective": {
                                               "kind": "escort_band",
                                               "position": list(goal),
                                               "r_min": 1.0,
                                               "r_max": self.follow_distance}},
                                rationale="follow effective friendly squad"))'''),
])
print("objectives wired")
