"""tactical_v1 manager facade (runner contract unchanged).

step(payload, n_agents) -> (actions, telemetry); commit_final_actions(actions)
commits memory ONLY for plans whose final actions survived untouched. The
step cache is keyed by the decision-relevant content (state + legal menus +
n_agents + config); repeated identical calls return safe copies and never
double-count observation history.
"""
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve()
_WORKSPACE = _HERE.parents[2]
if str(_WORKSPACE) not in sys.path:  # reuse of skills/current pure helpers
    sys.path.insert(0, str(_WORKSPACE))

from . import config as config_module
from .context import build_snapshot, Context
from .coordinator import Coordinator
from .evaluator import Evaluator
from .memory import TacticalMemory
from .registry import build_skills, GENERIC, SPECIALIZED
from .telemetry import Telemetry


def _digest(payload: Dict, n_agents: int, fingerprint: str) -> str:
    material = json.dumps(
        {"state": payload.get("state", {}),
         "questions": payload.get("questions", {}),
         "n_agents": n_agents, "fingerprint": fingerprint},
        sort_keys=True, default=str)
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def config_fingerprint(config: Dict[str, Any]) -> str:
    """Whole package source (recursive, tests excluded) + the actually
    reused shared modules + the effective configuration."""
    parts = []
    for path in sorted(p for p in _HERE.parent.rglob("*")
                       if p.suffix in (".py", ".json")):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        parts.append(path.read_bytes())
    shared = _WORKSPACE / "skills/current/skills"
    for name in ("damage_model.py", "combat_timing.py", "group_geometry.py"):
        target = shared / name
        if target.is_file():
            parts.append(target.read_bytes())
    # The reused legacy object-value evaluator changes behaviour with us.
    legacy_values = _WORKSPACE / "skills/current/unit_value.py"
    if legacy_values.is_file():
        parts.append(legacy_values.read_bytes())
    parts.append(json.dumps(config, sort_keys=True).encode())
    return hashlib.sha256(b"".join(parts)).hexdigest()[:16]



def _local_micro_target(ctx, summary, plans, actor, row, cmd):
    plan = next((p for p in plans if p.proposal_id == row['plan'] and p.primary_skill == 'local_engagement'
                 and actor in p.controlled_actors and p.current_actions.get(actor) == cmd), None)
    if not plan:
        return None
    data = plan.goal_data
    change = summary['local_coord']['changes'].get(actor, {})
    target = data['actor_target'].get(actor)
    tag = ctx.snapshot.raw_state['visible_enemies'].get(target, {}).get('tag')
    if (data['roles'].get(actor) != 'approach_attack' or data.get('request_id') or tag is None
            or str(tag) != data['target_tags'].get(target) or change.get('target_tag') != str(tag)
            or change.get('after') != cmd or change.get('before') == cmd or change.get('pack_id') != data['pack_id']):
        return None
    return target


def _micro_intents(ctx, summary, plans, actions):
    from .context import distance

    snap = ctx.snapshot
    intents = []
    if snap.step_mul != 8:
        return intents
    duties = summary['execution_duties'].get('actors', {})
    for actor, row in summary['units'].items():
        unit = snap.allies[actor]
        cmd = actions[unit.index]
        local = row['skill'] == 'local_engagement'
        target = _local_micro_target(ctx, summary, plans, actor, row, cmd) if local else row['goal']
        enemy = snap.enemies.get(target)
        duty = duties.get(actor, {})
        if (not unit.alive or unit.type_name != 'hydralisk' or row['action'] != cmd
                or cmd not in ctx.move_destinations(actor) or (not local and row['goal_kind'] != 'enemy')
                or not enemy or enemy.health <= 0 or duty.get('kind') != 'entry'
                or not duty.get('satisfied') or duty.get('selected_action') != cmd
                or duty.get('joint_rescue') or duty.get('branch_reason')
                or actor in summary['joint_rescue_substitutions'] or ctx.forced_hydra_targets(actor)):
            continue

        end = ctx.predict_position(actor, cmd).position
        role = row['skill'], row['phase']
        if local:
            if duty.get('reason') != 'local_contact' or duty.get('target') != target:
                continue
        elif role == ('ranged_fire_formation', 'exclusive_control'):
            # This phase also owns search, alignment, escapes and stationary
            # fallback. Exclude moves that leave predicted coverage, even if
            # the same step incidentally closes on the shared focus.
            model = getattr(ctx, '_ranged_formation_forecast', None)
            if (row['plan'] is not None or ctx.formation_orders.get(actor) != cmd
                    or target != ctx.hydra_formation_target or duty['reason'] != 'exclusive_hydra_formation'
                    or cmd == getattr(ctx, '_debug_stationary', {}).get(actor, cmd)
                    or model is None or model.threats_at(actor, unit.position) - model.threats_at(actor, end)):
                continue
        elif role == ('coordinated_engage', 'approaching'):
            # Do not treat synchronizing/supporting/guarding (even with an
            # enemy goal), or a deployment/follow duty as an attack approach.
            plan = next((p for p in plans if p.proposal_id == row['plan']
                         and p.current_actions.get(actor) == cmd), None)
            if (plan is None or plan.goal != target or plan.goal_data.get('target_id') != target
                    or plan.goal_data.get('objective', {}).get('kind') != 'range_band'
                    or plan.goal_data['objective'].get('target') != target
                    or duty.get('target') != target
                    or duty.get('reason') not in ('short_route_commit', 'join_active_front_now',
                                                  'uncommitted_approach', 'replace_failed_or_overdue_route')):
                continue
        else:
            continue

        # An unmasked public menu can include out-of-range enemies. Use the
        # geometric mask fact at L, not a prospective shot or nearest enemy.
        rel = snap.raw_state['relations'].get(actor, {}).get(target, {})
        if rel.get('smac_center_distance_allowed') is not False:
            continue
        reach = ctx.effective_attack_reach(actor, target)
        before = set(ctx.risk_at(actor, unit.position).threats)
        if (not reach or reach <= 0 or before - set(ctx.risk_at(actor, end).threats)
                or distance(end, enemy.position) + .05 >= distance(unit.position, enemy.position)):
            continue
        raw_enemy = snap.raw_state['visible_enemies'][target]
        tags = [str(snap.raw_state['allies'][actor].get('tag')), str(raw_enemy.get('tag'))]
        if (raw_enemy.get('alive') is False or not all(t.isascii() and t.isdecimal()
                and 0 < int(t) < 2**64 for t in tags)):
            continue
        intents.append(dict(actor=actor, actor_tag=tags[0], target=target, target_tag=tags[1],
                            issued_loop=snap.game_loop, macro_action_id=cmd,
                            kind='approach_then_attack', skill=row['skill'], phase=row['phase']))
    return intents


"""One-window permissions from the winning attack route, never target selection."""
from math import dist, isfinite

from .actions import attack_action_id


def _native_tag(raw):
    text = str(raw.get('tag'))
    if (len(text) <= 20 and text.isascii() and text.isdecimal()
            and not text.startswith('0') and int(text) < 2**64):
        return text
    return None


def _named_approach(ctx, actor, cmd, plan, duty, local):
    """Read the provider's own target; a duty or a closing vector alone is not a plan."""
    data = plan.goal_data
    target = data.get('target_id')
    phase = plan.commitment.phase
    reason = duty.get('reason')
    unit = ctx.snapshot.allies[actor]
    semantic = ctx.decode(actor, cmd)
    if plan.primary_skill == 'local_engagement':
        # In a joint plan the actor's enemy need not be the group's first enemy.
        target = data.get('actor_target', {}).get(actor)
        tag = _native_tag(ctx.snapshot.raw_state['visible_enemies'].get(target, {}))
        change = local.get('changes', {}).get(actor, {})
        if (semantic.kind == 'move' and phase == 'support' and reason == 'local_contact'
                and duty.get('kind') == 'entry' and duty.get('target') == target
                and not data.get('request_id') and data.get('roles', {}).get(actor) == 'approach_attack'
                and data.get('actor_tags', {}).get(actor) == str(unit.tag)
                and tag and data.get('target_tags', {}).get(target) == change.get('target_tag') == tag
                and change.get('skill') == plan.primary_skill and data.get('pack_id')
                and change.get('pack_id') == data['pack_id'] and change.get('after') == cmd
                and isinstance(change.get('before'), int) and change['before'] != cmd
                and change['before'] == data.get('reference_actions', {}).get(actor)):
            return target
        return None

    if target != plan.goal or plan.controlled_actors != (actor,):
        return None
    if plan.primary_skill == 'burst_engage':
        # A fixed AoE region has an anchor enemy but is not a command to
        # attack that enemy. Only the direct-contact branch names a victim.
        if (unit.type_name == 'baneling' and phase == 'approaching' and semantic.kind == 'move'
                and duty.get('kind') == 'burst' and reason == 'cashout_direct_contact'
                and duty.get('target') == target and cmd in duty.get('actions', ())
                and (data.get('execution_duty') or {}).get('reason') == reason):
            return target
        return None

    if duty.get('kind') != 'entry':
        return None
    objective = data.get('objective', {})
    melee = unit.type_name in ('zergling', 'zealot')
    g2 = (plan.primary_skill, phase) == ('coordinated_engage', 'approaching')
    if semantic.kind == 'attack':
        # The public mask can open before a melee weapon makes contact.
        # G1 can also own the same attack while the FINAL duty is exactly
        # native melee approach. Ordinary fire, including on cooldown, cannot.
        rel = ctx.snapshot.relations.get(actor, {}).get(target)
        if (melee and semantic.target == target and duty.get('required') and cmd in duty.get('actions', ())
                and reason == 'native_contact_commit' and duty.get('target') == target
                and rel is not None and rel.in_weapon_range is False
                and ((plan.primary_skill, phase) == ('focus_fire', 'attacking') or (
                    g2 and data.get('contact_execution') == reason and objective.get('kind') == 'range_band'
                    and objective.get('target') == target))):
            return target
        return None
    if not g2 or semantic.kind != 'move':
        return None
    if not ((reason in ('uncommitted_approach', 'short_route_commit', 'join_active_front_now',
                        'replace_failed_or_overdue_route') and duty.get('target') == target)
            or (reason == 'no_visible_contact_duty' and not duty.get('required') and not duty.get('target'))):
        return None
    if melee:
        # G2 reserves a contact slot, not a ranged unit's range_band.
        if objective.get('kind') == 'contact_point' and any(
                c.actor == actor and (c.slot_id or '').startswith(f'slot:{target}:') for c in plan.claims.space):
            return target
    elif objective.get('kind') == 'range_band' and objective.get('target') == target:
        return target
    return None


def extra_approaches(ctx, summary, plans, actions):
    """Capture stable identities at decision time; publish only after a matching commit."""
    snap = ctx.snapshot
    if snap.step_mul != 8:
        return []
    raw = snap.raw_state
    duties = summary['execution_duties']['actors']
    by_id = {p.proposal_id: p for p in plans if p.snapshot_id == snap.snapshot_id}
    grants = []
    for actor, row in summary['units'].items():
        unit = snap.allies[actor]
        cmd = actions[unit.index]
        duty = duties.get(actor, {})
        # Legacy hydra authorizations pass through unchanged. No new permissions
        # for a healer, an unknown type, or a branch satisfying another duty.
        if (unit.type_name not in ('zergling', 'baneling', 'zealot', 'stalker', 'colossus', 'marine', 'marauder')
                or not unit.alive or unit.health <= 0 or row['action'] != cmd
                or cmd not in ctx.legal_actions(actor) or duty.get('selected_action') != cmd
                or not duty.get('satisfied') or duty.get('branch_reason') or duty.get('joint_rescue')
                or duty.get('isolated_shooter_support') or actor in summary['joint_rescue_substitutions']):
            continue
        plan = by_id.get(row['plan'])
        if plan is None:
            # The live all-Zerg profile completes Banelings without CandidatePlans.
            # Require BOTH the actual directed fallback and the direct-contact
            # duty; an unlabelled baseline or a region/wait duty is not enough.
            target = duty.get('target')
            roster = getattr(ctx, 'zerg_policy', None)
            if not (unit.type_name == 'baneling' and row['plan'] is None and row['skill'] == 'baseline'
                    and roster and roster.named_moves.get(actor) == (cmd, target)
                    and ctx.decode(actor, cmd).kind == 'move' and duty.get('kind') == 'burst'
                    and duty.get('reason') == 'cashout_direct_contact' and cmd in duty.get('actions', ())):
                continue
            skill, phase = 'baseline', 'cashout_direct_contact'
        elif (row['skill'] != plan.primary_skill or row['phase'] != plan.commitment.phase
                or row['goal'] != plan.goal or row['goal_kind'] != 'enemy' or plan.goal_kind != 'enemy'
                or actor not in plan.controlled_actors or plan.current_actions.get(actor) != cmd):
            continue
        else:
            skill, phase = row['skill'], row['phase']
            target = _named_approach(ctx, actor, cmd, plan, duty, summary.get('local_coord', {}))
        enemy = snap.enemies.get(target)
        if not enemy or enemy.health <= 0:
            continue
        actor_tag = _native_tag(raw['allies'][actor])
        raw_enemy = raw['visible_enemies'][target]
        target_tag = _native_tag(raw_enemy)
        if (not actor_tag or not target_tag or actor_tag == target_tag or actor_tag != str(unit.tag)
                or raw_enemy.get('alive') is False):
            continue
        if unit.type_name == 'baneling':
            # Effect-based ground contact, no native catalog weapon. Never use
            # splash radius or issue Detonate. Execution rechecks body contact.
            if enemy.is_flying or unit.unit_type_id != 9:
                continue
            reach = unit.radius + enemy.radius
        else:
            reach = ctx.effective_attack_reach(actor, target)
        before = dist(unit.position, enemy.position)
        if not reach or not isfinite(reach) or reach <= 0 or not isfinite(before) or before <= reach:
            continue
        if ctx.decode(actor, cmd).kind == 'move':
            if cmd not in ctx.move_destinations(actor):
                continue
            end = ctx.predict_position(actor, cmd).position
            if not dist(end, enemy.position) + .05 < before:
                continue
        grants.append(dict(actor=actor, actor_tag=actor_tag, target=target, target_tag=target_tag,
                           issued_loop=snap.game_loop, macro_action_id=cmd, kind='approach_then_attack',
                           skill=skill, phase=phase))
    return grants


def merge_approaches(original, extra):
    actors, tags, merged = set(), set(), []
    for r in (*original, *extra):
        if r['actor'] not in actors and r['actor_tag'] not in tags:
            merged.append(r)
            actors.add(r['actor'])
            tags.add(r['actor_tag'])
    return merged


def record_approach(memory, intents, actor, action):
    """Runner calls only for an accepted interior attack. No plan/timer commit."""
    if type(action) is int and any(
            r['actor'] == actor and memory.last_committed_actions.get(actor) == r['macro_action_id']
            and r['macro_action_id'] != action and attack_action_id(int(r['target'][1:])) == action
            for r in intents):
        memory.last_committed_actions[actor] = action


class SkillManager:
    micro_all_attackers = False

    def __init__(self, max_steps: int = 4, posture: str = "balanced",
                 tactical_config: Optional[str] = None, **compat):
        # Explicit compatibility surface: unknown kwargs are recorded, never
        # silently behaviour-changing. The arg resolver rejects the legacy
        # flag surface for tactical_v1 before we ever get here.
        self.compatibility_notes = {k: v for k, v in compat.items()}
        self.config = config_module.load(tactical_config)
        self.config["_fingerprint"] = config_fingerprint(self.config)
        self.posture = posture
        self.skills = build_skills(self.config)
        enabled = self.config.get("enabled", {})
        # real mechanism switches reach the FINAL effect/scoring path (B3)
        self.evaluator = Evaluator(
            self.config.get("weights"),
            mechanisms={"splash": bool(enabled.get("splash_fire", True)),
                        "burst": bool(enabled.get("burst_engage", True))},
            splash_radius=self.config.get("splash_fire", {}).get(
                "coverage_radius", 3.0),
            splash_weight=self.config.get("splash_fire", {}).get(
                "coverage_weight", 0.5),
            burst_radius=self.config.get("burst_engage", {}).get(
                "proxy_radius", 2.5),
            burst_weight=0.5,
            burst_self_cost=self.config.get("burst_engage", {}).get(
                "self_cost", 1.0),
            forecast_config=self.config.get("forecasting"),
            urgency_value=self.config.get("focus_fire", {}).get(
                "urgency_value", 0.5),
            target_planning_enabled=self.config.get("focus_fire", {}).get(
                "target_planning_enabled", True),
            common_combat=self.config.get("common_combat"),
            splash_reliability=self.config.get("splash_fire", {}).get(
                "coverage_reliability", 0.5))
        self.coordinator = Coordinator(self.config, self.evaluator)
        lifecycle = self.config.get("lifecycle", {})
        self.memory = TacticalMemory(
            observation_ttl_loops=int(lifecycle.get("observation_ttl_loops", 240)),
            search_revisit_cooldown_loops=int(
                lifecycle.get("search_revisit_cooldown_loops", 160)),
            retry_cooldown_h=lifecycle.get("retry_cooldown_h", 2.0),
            no_progress_window_h=lifecycle.get("no_progress_window_h", 3.0),
            encounter_config=self.config)
        self.telemetry = Telemetry(top_k_candidates=int(
            self.config.get("telemetry", {}).get("top_k_candidates", 3)))
        self._snapshot_counter = 0
        self._cache = None           # (digest, actions, summary, plans, snapshot)
        self._pending_commit = None
        self.stats = {"steps": 0, "episodes": 0, "search_budget_events": 0,
                      "repairs": 0, "baseline_units": 0}
        self.current_skill = "tactical_v1"
        self.reset()  # construction resets like the current manager does

    # -- runner interface ---------------------------------------------------
    @property
    def active(self) -> bool:
        return True  # the joint controller is always the acting policy

    def reset(self) -> None:
        self.last_micro_intents = []
        self.memory = TacticalMemory(
            observation_ttl_loops=self.memory.observation_ttl_loops,
            search_revisit_cooldown_loops=self.memory.search_revisit_cooldown_loops,
            retry_ttl_loops=self.memory.retry_ttl_loops,
            retry_cooldown_h=self.memory.retry_cooldown_h,
            no_progress_window_h=self.memory.no_progress_window_h,
            encounter_config=self.config)
        self.telemetry = Telemetry(top_k_candidates=self.telemetry.top_k)
        self._snapshot_counter = 0
        self._cache = None
        self._pending_commit = None

    def select(self, skill: Optional[str] = None, priority_target=None,
               posture: Optional[str] = None) -> None:
        """Accepted for interface compatibility. The joint coordinator never
        hands exclusive control to one skill; posture is recorded."""
        if posture:
            self.posture = posture

    def step(self, payload: Dict, n_agents: int):
        self.last_micro_intents = []
        digest = _digest(payload, n_agents, self.config["_fingerprint"])
        cached = self._cache
        if cached is not None and cached[0] == digest:
            # Same decision content: idempotent re-read, safe copies, no
            # observation-history appends, no budget advance.
            return list(cached[1]), copy.deepcopy(cached[2])
        snapshot = build_snapshot(payload, self._snapshot_counter + 1)
        self._snapshot_counter += 1
        self.memory.begin_step(snapshot)
        context = Context(snapshot)
        context.deployment_memory = self.memory
        # Shared per-snapshot object-value configuration (unit_value_view);
        # queries read it lazily and never mutate tasks.
        context.unit_value_parameters = dict(self.config.get("unit_value", {})) \
            or {"enabled": False}
        context.intercept_config = dict(self.config.get("intercept", {}))
        context.common_config = dict(self.config.get("common_combat", {}))
        context.burst_config = dict(self.config.get("burst_engage", {}))
        risk_now = {a: len(context.risk_at(a, snapshot.allies[a].position).threats)
                    for a in snapshot.allies if snapshot.allies[a].alive}
        # observation phase FIRST (single ordered entry, 3.1): record the
        # observation, then judge identity invalid -> progress -> completion
        # -> deadline. No earlier retire call may delete timed-out tasks.
        self.memory.observe(snapshot, risk_now)
        for event in self.memory.update_task_facts(snapshot, context):
            self.telemetry.log(event)
        previous_budget_events = self.telemetry.search_budget_events
        actions, summary, plans = self.coordinator.decide(
            context, self.skills, self.memory, self.telemetry, n_agents)
        summary["battle_intent"] = self.memory.battle_state(context)
        provider = self.skills.get("splash_fire")
        if provider is not None and hasattr(provider, "coverage_model"):
            summary["splash_model"] = provider.coverage_model()
        self.stats["steps"] += 1
        self.stats["episodes"] += self._snapshot_counter == 1
        # Telemetry resets per episode; run statistics accumulate deltas.
        self.stats["search_budget_events"] += (
            self.telemetry.search_budget_events - previous_budget_events)
        self.stats["repairs"] += len(summary.get("repairs", ()))
        self.stats["baseline_units"] += sum(
            row["skill"] == "baseline" and snapshot.allies[actor].alive
            for actor, row in summary.get("units", {}).items())
        self._cache = (digest, list(actions), copy.deepcopy(summary), plans,
                       snapshot)
        self._pending_commit = (snapshot, plans, list(actions), risk_now, context)
        if self.micro_all_attackers is True:
            context.micro_approaches = extra_approaches(context, summary, plans, actions)
        return list(actions), copy.deepcopy(summary)

    def commit_final_actions(self, final_actions: List[int]) -> None:
        """Commit memory for exactly the plans whose actions survived."""
        pending = self._pending_commit
        self._pending_commit = None
        self.last_micro_intents = []
        if pending is None:
            return
        snapshot, plans, actions, risk_now, context = pending
        actions_by_actor = {f"u{i}": a for i, a in enumerate(final_actions)}
        intents = (_micro_intents(context, self._cache[2], plans, actions)
                   if final_actions == actions else [])
        if self.micro_all_attackers is True and final_actions == actions:
            intents = merge_approaches(intents, getattr(context, 'micro_approaches', ()))
        self.memory.commit(snapshot, actions_by_actor, plans, risk_now,
                           context=context)

        from .local_engagement import commit_local
        commit_local(context, self.memory, self._cache[2]["local_coord"])
        self.last_micro_intents = intents

    def record_micro_action(self, actor: str, action: int) -> None:
        """Runner acknowledgement; do not re-commit tasks, plans or timers."""
        record_approach(self.memory, self.last_micro_intents, actor, action)
