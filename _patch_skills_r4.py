"""Round-4 skill patch: goal metadata wiring + E-section gaps."""
import io, re

def patch(path, pairs):
    s = io.open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert old in s, f"{path}: MISSING\n{old[:120]}"
        s = s.replace(old, new)
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)
    print(f"patched {path}")

# ---- G1 focus_fire: enemy goal metadata -----------------------------------
patch("skills/tactical_v1/skills/focus_fire.py", [
    ('''                    snapshot_id, fingerprint, provider=provider,
                    instance_id=instance.instance_id if instance else None,
                    memory_delta=delta,
                    rationale=f"attack {enemy_id}"))''',
     '''                    snapshot_id, fingerprint, provider=provider,
                    instance_id=instance.instance_id if instance else None,
                    memory_delta=delta,
                    goal_kind="enemy",
                    goal_data={"target_id": enemy_id},
                    rationale=f"attack {enemy_id}"))'''),
    ('''                            {"urgency": self.urgency_value},
                            snapshot_id, fingerprint, atomic=True,''',
     '''                            {"urgency": self.urgency_value},
                            snapshot_id, fingerprint, atomic=True,
                            goal_kind="enemy",
                            goal_data={"target_id": enemy_id},'''),
])

# ---- G2 coordinated_engage: enemy goal + waypoint; waiting gets a budget --
patch("skills/tactical_v1/skills/coordinated_engage.py", [
    ('''                    instance_id=instance.instance_id if instance else None,
                    dependencies=(Dependency("target_valid", (target_id,)),),
                    memory_delta=({"kind": "create", "skill": self.name,''',
     '''                    instance_id=instance.instance_id if instance else None,
                    dependencies=(Dependency("target_valid", (target_id,)),),
                    goal_kind="enemy",
                    goal_data={"target_id": target_id, "waypoint": goal},
                    memory_delta=({"kind": "create", "skill": self.name,'''),
    ('''                    {"hold_value": 0.3}, snapshot_id, fingerprint,
                    rationale="bounded wait for squad contact"))''',
     '''                    {"hold_value": 0.3}, snapshot_id, fingerprint,
                    goal_kind="enemy",
                    goal_data={"target_id": target_id, "waypoint": goal},
                    memory_delta={"kind": "create", "skill": self.name,
                                  "actors": [actor], "goal": target_id,
                                  "goal_kind": "enemy", "phase": "waiting",
                                  "max_loops": h_loops(context,
                                                       self.wait_max_h)},
                    rationale="bounded wait for squad contact"))'''),
])

# ---- G3 combat_reposition: enemy goal --------------------------------------
patch("skills/tactical_v1/skills/combat_reposition.py", [
    ('''                    memory_delta=({"kind": "create", "skill": self.name,
                                   "actors": [actor], "goal": target_id,
                                   "goal_kind": "enemy",
                                   "phase": "repositioning",
                                   "max_loops": budget}
                                  if instance is None else None),
                    rationale=f"{label} during weapon cycle"))''',
     '''                    memory_delta=({"kind": "create", "skill": self.name,
                                   "actors": [actor], "goal": target_id,
                                   "goal_kind": "enemy",
                                   "phase": "repositioning",
                                   "max_loops": budget}
                                  if instance is None else None),
                    goal_kind="enemy",
                    goal_data={"target_id": target_id},
                    rationale=f"{label} during weapon cycle"))'''),
])

# ---- G5 regroup: group_link goal with the FIXED group pair (B3) ------------
patch("skills/tactical_v1/skills/regroup.py", [
    ("""    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        loop = context.snapshot.game_loop
        budget = h_loops(context, self.max_h)""",
     """    def _link_pairs(self, context: Context, actor: str):
        \"\"\"The comparable (own group, other group) pairs for this actor,
        FIXED for the whole search from the current visible state (B3).\"\"\"
        groups = context.engagement_groups(self.support_radius)
        own = next((g for g in groups if actor in g), [actor])
        pairs = []
        for other in groups:
            if other == own:
                continue
            pairs.append((own, other))
        return own, pairs

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        loop = context.snapshot.game_loop
        budget = h_loops(context, self.max_h)"""),
    ('''            for goal in options[:2]:          # comparable alternatives
                move = best_move_toward(context, actor, goal)
                if move is None:
                    continue
                plans.append(make_plan(
                    self.name, {actor: move}, "merge", {actor: move},''',
     '''            own, pairs = self._link_pairs(context, actor)
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
                    self.name, {actor: move}, "merge", {actor: move},'''),
    ('''                    memory_delta=({"kind": "create", "skill": self.name,
                                   "actors": [actor], "goal": "merge",
                                   "goal_kind": "area", "phase": "rallying",
                                   "max_loops": budget,
                                   "extra": {"waypoint": goal}}
                                  if instance is None else None),
                    rationale="rally toward merge goal"))''',
     '''                    memory_delta=({"kind": "create", "skill": self.name,
                                   "actors": [actor], "goal": "merge",
                                   "goal_kind": "area", "phase": "rallying",
                                   "max_loops": budget,
                                   "extra": {"waypoint": goal}}
                                  if instance is None else None),
                    goal_kind="group_link",
                    goal_data={
                        "group_a": sorted(own),
                        "group_b": sorted(other) if other else [],
                        "support_radius": self.support_radius,
                        "scale": 4.0, "position": list(goal)},
                    task_key="|".join(("regroup", ",".join(sorted(own)),
                                       ",".join(sorted(other or [])))),
                    rationale="rally toward merge goal"))'''),
])

# ---- G6 local_formation: role goals for colossus (B4) ----------------------
patch("skills/tactical_v1/skills/local_formation.py", [
    ('''            if best_action is not None:
                plans.append(make_plan(
                    self.name, {actor: best_action}, "formation",
                    {actor: best_action},''',
     '''            guards = [a for a, u in sorted(context.snapshot.allies.items())
                      if a != actor and u.alive and not u.is_flying
                      and distance(u.position, unit.position) <= 6.0]
            if best_action is not None:
                plans.append(make_plan(
                    self.name, {actor: best_action}, "formation",
                    {actor: best_action},'''),
    ('''                    {"switch_cost": 0.1 if attackable else 0.0},
                    snapshot_id, fingerprint,
                    rationale=("spread from splash danger" if under_splash
                               else "role/quality position adjustment")))''',
     '''                    {"switch_cost": 0.1 if attackable else 0.0},
                    snapshot_id, fingerprint,
                    goal_kind=(
                        "role" if capability.specialization == "splash_fire"
                        and guards else "position"),
                    goal_data=(
                        {"role": "colossus_protected", "actor": actor,
                         "guards": guards}
                        if capability.specialization == "splash_fire"
                        and guards else {"position": list(role_goal or
                                                          unit.position)}),
                    rationale=("spread from splash danger" if under_splash
                               else "role/quality position adjustment")))'''),
])

# ---- G7 cohesive_search: waypoint travels with the goal --------------------
patch("skills/tactical_v1/skills/cohesive_search.py", [
    ('''                    {"progress": move_progress(context, actor, move, goal)},
                    snapshot_id, fingerprint,''',
     '''                    {"progress": move_progress(context, actor, move, goal)},
                    snapshot_id, fingerprint,
                    goal_kind="position",
                    goal_data={"waypoint": list(goal)},'''),
])

# ---- G4: goal metadata + retry key fix + H0 budget -------------------------
patch("skills/tactical_v1/skills/disengage_reenter.py", [
    ("            if memory.retry_blocked(self.name, [actor], \"escape\", loop):",
     "            if memory.retry_blocked(self.name, [actor], \"safe\", loop):"),
    ('''    def _delta(self, instance, phase, actor, goal=None):
        total_budget = int(self.total_max_h * 8)''',
     '''    def _delta(self, instance, phase, actor, goal=None):
        total_budget = int(self.total_max_h * 8)   # replaced at propose time'''),
])

# ---- S2 heal_support: ally goal + meeting goal_data for joint meets --------
patch("skills/tactical_v1/skills/heal_support.py", [
    ('''                    instance_id=instance.instance_id if instance else None,
                    rationale=f"heal {patient}" +''',
     '''                    instance_id=instance.instance_id if instance else None,
                    goal_kind="ally",
                    goal_data={"target_id": patient},
                    rationale=f"heal {patient}" +'''),
    ('''                        Commitment("meet_patient", 0, max_loops=6 * context.H),
                        {"switch_cost": 0.3 if patient_attack else 0.0},
                        snapshot_id, fingerprint, provider="heal_support",
                        atomic=True,''',
     '''                        Commitment("meet_patient", 0, max_loops=6 * context.H),
                        {"switch_cost": 0.3 if patient_attack else 0.0},
                        snapshot_id, fingerprint, provider="heal_support",
                        atomic=True,
                        goal_kind="meeting",
                        goal_data={"healer": actor, "patient": patient,
                                   "position": list(meet)},'''),
    ('''                                Commitment("escort", 0, max_loops=6 * context.H),
                                {}, snapshot_id, fingerprint,
                                provider="heal_support",
                                rationale="follow effective friendly squad"))''',
     '''                                Commitment("escort", 0, max_loops=6 * context.H),
                                {}, snapshot_id, fingerprint,
                                provider="heal_support",
                                goal_kind="position",
                                goal_data={"waypoint": list(goal)},
                                rationale="follow effective friendly squad"))'''),
])

# ---- S3 burst_engage: enemy goal + keep TWO attack attempts (E4) -----------
patch("skills/tactical_v1/skills/burst_engage.py", [
    ('''            best = next((a for a in attempts if a[0] >= threshold), None)
            if best is not None:
                value, enemy_id, contact = best
                action = context.attack_action(actor, enemy_id)''',
     '''            viable = [a for a in attempts if a[0] >= threshold][:2]
            for value, enemy_id, contact in viable or []:
                action = context.attack_action(actor, enemy_id)'''),
    ('''                    rationale=f"burst attempt at {enemy_id} "
                              f"(contact={contact}, proxy value {value:.1f})"))
                continue''',
     '''                    rationale=f"burst attempt at {enemy_id} "
                              f"(contact={contact}, proxy value {value:.1f})"))
                if len(viable) <= 1:
                    break
            if viable:
                continue'''),
    ('''                    memory_delta=({"kind": "create", "skill": self.name,
                                       "actors": [actor], "goal": enemy_id,
                                       "goal_kind": "enemy",
                                       "phase": "approaching",
                                       "max_loops": 40 * context.H,
                                       "extra": {"waypoint": position}}
                                      if instance is None else None),
                        rationale=f"approach burst region {enemy_id}"))''',
     '''                    memory_delta=({"kind": "create", "skill": self.name,
                                       "actors": [actor], "goal": enemy_id,
                                       "goal_kind": "enemy",
                                       "phase": "approaching",
                                       "max_loops": 40 * context.H,
                                       "extra": {"waypoint": position}}
                                      if instance is None else None),
                        goal_kind="enemy",
                        goal_data={"target_id": enemy_id,
                                   "waypoint": list(position)},
                        rationale=f"approach burst region {enemy_id}"))'''),
])
print("all skills patched")
