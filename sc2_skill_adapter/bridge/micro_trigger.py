"""Execution-layer 8-loop micro trigger for authorized approach-then-attack intents.

One macro step (step_mul=8) is still ONE policy step. After the team's final
actions are submitted at loop L, the window interior (L+1..L+7) only re-checks
authorizations registered by the macro policy. Nothing here runs a tactical
policy: no target selection, no STOP/no-op/move fillers, no re-sends of team
orders. The single permitted operation is one non-queued Attack(A) per
authorized attacker, and only while A is alive, currently visible, and legal
under BOTH current SMACv2 menus. In micro8-all the additional unit types
must also be in native weapon range of their authorized target.

micro8-off and micro8-hydra share this state machine verbatim; the only
difference is `execute`. The off mode records when every condition would have
been satisfied (first_legal_loop / trigger states) without ever issuing a
command, so paired runs answer "when would the trigger have fired first?".
"""
import math

from s2clientprotocol import raw_pb2, sc2api_pb2 as sc_pb

from .policy import weapon_range

# The SMAC environment resolves native vs. _RL allied type IDs at reset.
_ADDITIONAL_ATTACKERS = ('marine', 'marauder', 'stalker', 'zealot', 'colossus',
                         'zergling', 'baneling')

# SC2 Attack and its general ability ID (PySC2 Attack_unit).
_ATTACK_IDS = {23, 3674}

# Final per-authorization states (submitted means the command left the
# process; it is never evidence of firing or damage).
SUBMITTED = "submitted"                # Attack(A) submitted this window
ALREADY_ORDERED = "already_ordered"    # effective order was already Attack(A)
TARGET_LOST = "target_lost"            # A died or left team vision; revoked
NEVER_LEGAL = "never_legal"            # window ended without menu legality
LEGAL_CONTROL_OFF = "legal_control_off"  # legal, control mode does not send
ACTOR_DEAD = "actor_dead"              # authorized actor died or was replaced


class MicroAttackWindow:
    """Authorizations and per-loop checks for exactly one macro window.

    `intents` come from the macro policy after final-action arbitration
    (skill_manager.last_micro_intents): {actor, actor_tag, target,
    target_tag, issued_loop, kind, skill, phase}. Tags are stable SC2 tags;
    enemy slot ids (e{i}) are never used as target identity.
    """

    def __init__(self, env, intents, macro_step, issued_loop, macro_loops,
                 execute, emit=None, episode=None, *, all_attackers=False, static_data=None):
        if macro_loops != 8:
            raise ValueError("micro trigger supports macro_loops=8 only")
        self.env, self.macro_step, self.issued_loop = env, macro_step, issued_loop
        self.expires_loop = issued_loop + macro_loops - 1
        self.execute = bool(execute)
        self.all_attackers, self.static_data = all_attackers, static_data
        self.emit, self.episode = emit, episode
        self.interior_checks = 0
        self._last_loop = issued_loop
        self._closed = False
        self.records = []
        self._by_tag = {}
        if intents and self.env._obs.observation.game_loop == issued_loop:
            visible = self._visible_slots()
            for intent in intents:
                self._register(intent, visible)

    def _register(self, intent, visible):
        if (intent.get("kind") != "approach_then_attack"
                or intent.get("issued_loop") != self.issued_loop):
            return
        # Bind the policy's decision-time slots to native tags. Slots are
        # never identities: a later replacement cannot inherit this grant.
        actor_tag, target_tag = str(intent.get('actor_tag')), str(intent.get('target_tag'))
        unit = next((u for i, u in self.env.agents.items() if 0 <= i < self.env.n_agents
                     and f'u{i}' == intent.get('actor') and u.health > 0 and u.tag
                     and str(u.tag) == actor_tag), None)
        if unit is None or target_tag not in visible:
            return
        enemy_id, target = visible[target_tag]
        if f'e{enemy_id}' != intent.get('target'):
            return
        actor_type = self._actor_type(unit)
        if actor_type is None or (actor_type != 'hydralisk'
                and self._native_reach(actor_type, unit, target) is None):
            return
        record = {
            "macro_step": self.macro_step, "episode": self.episode,
            "actor": intent.get("actor"), "actor_tag": str(intent["actor_tag"]),
            "actor_type": actor_type, "actor_type_id": int(unit.unit_type),
            "target": intent.get("target"), "target_tag": str(intent["target_tag"]),
            "kind": intent["kind"], "skill": intent.get("skill"),
            "phase": intent.get("phase"),
            "issued_loop": int(self.issued_loop), "macro_action_id": intent.get('macro_action_id'),
            "expires_loop": int(self.expires_loop),
            "first_legal_loop": None, "submitted_loop": None,
            "submitted_action_id": None,
            "status": None, "checked_loops": 0,
        }
        self.records.append(record)
        if record["actor_tag"] in self._by_tag:
            raise ValueError(f"duplicate micro authorization for {record['actor']}")
        self._by_tag[record["actor_tag"]] = record
        self._emit_state(record, event="micro_authorization_registered")

    # ------------------------------------------------------------- helpers
    def _emit(self, event, **fields):
        if self.emit is not None:
            self.emit(event, episode=self.episode, macro_step=self.macro_step, **fields)

    def _emit_state(self, record, event, **fields):
        self._emit(event, actor=record["actor"], actor_tag=record["actor_tag"],
                   actor_type=record['actor_type'],
                   target=record["target"], target_tag=record["target_tag"],
                   loop=self.env._obs.observation.game_loop,
                   status=record["status"], **fields)

    def _finish(self, record, status, loop=None):
        if record["status"] is not None:
            return
        record["status"] = status
        if loop is not None and status == SUBMITTED:
            record["submitted_loop"] = int(loop)
        self._emit_state(record, event="micro_authorization_state")

    def _actor_type(self, unit):
        if unit.unit_type == self.env.hydralisk_id:
            return 'hydralisk'
        if self.all_attackers:
            return next((name for name in _ADDITIONAL_ATTACKERS
                         if unit.unit_type == getattr(self.env, name + '_id', None)), None)
        return None

    def _native_reach(self, actor_type, unit, target):
        """Center distance limit for a compatible native attack; None if unknown."""
        stat = self.static_data.unit_stats.get(unit.unit_type) if self.static_data is not None else None
        if stat is None:
            return None
        reach = weapon_range(self.static_data, unit, target)
        if actor_type == 'baneling' and not stat.weapons and not target.is_flying:
            # The catalog omits the Baneling's ground contact effect. Allow
            # body contact only; neither SMAC's min range nor the blast radius.
            reach = 0.0
        if reach is None or not all(math.isfinite(x) and x >= 0
                                    for x in (reach, unit.radius, target.radius)):
            return None
        return reach + unit.radius + target.radius

    def _order_targets(self, unit, target_tag):
        """True when the unit's current effective order already attacks A."""
        orders = list(getattr(unit, "orders", ()) or ())
        return bool(orders) and orders[0].ability_id in _ATTACK_IDS \
            and str(orders[0].target_unit_tag) == target_tag

    def _visible_slots(self):
        from .environment import visible_enemies
        return {str(u.tag): (i, u) for i, u in visible_enemies(self.env).items()
                if u.tag and u.display_type == raw_pb2.Visible and u.health > 0}

    def _submit_attack(self, actor_id, action, record):
        """One non-queued Attack(A) through the upstream command builder.

        get_agent_action re-asserts the CURRENT menu mask and rebuilds the
        exact SMACv2 attack command; it never queues. The submission bypasses
        the macro feedback collection window, so it cannot pollute the next
        macro decision's last_action_result, and it is never counted as a
        second tactical decision.
        """
        command = self.env.get_agent_action(actor_id, action)
        if command is None:
            raise RuntimeError('True-menu attack produced no native command')
        order = command.action_raw.unit_command
        if (order.ability_id not in _ATTACK_IDS or order.queue_command
                or list(order.unit_tags) != [int(record['actor_tag'])]
                or str(order.target_unit_tag) != record['target_tag']):
            raise RuntimeError('SMAC command does not match micro authorization')
        response = self.env.submit_micro_commands(sc_pb.RequestAction(actions=[command]))
        record['submitted_action_id'] = int(action)
        record["sc2_response"] = [int(code) for code in response.result]

    # ---------------------------------------------------------------- loop
    def on_interior_loop(self, offset):
        """Re-check every pending authorization after one engine loop.

        Checks run only at L+1..L+7 on the fresh observation. L+8 closes the
        window without acting on it: the next command comes from the next
        tactical decision, never from the old authorization.
        """
        loop = self.env._obs.observation.game_loop
        if self._closed:
            return
        if loop > self.expires_loop:
            self.close(loop)
            return
        if loop <= self._last_loop or loop != self.issued_loop + offset:
            return
        self._last_loop = loop
        self.interior_checks += 1
        pending = [r for r in self.records if r["status"] is None]
        if not pending:
            return
        visible = self._visible_slots()
        submitted = 0
        for record in pending:
            record["checked_loops"] += 1
            actor_id = int(record["actor"][1:])
            unit = self.env.agents.get(actor_id)
            if unit is None or unit.health <= 0 or unit.unit_type != record['actor_type_id'] \
                    or str(unit.tag) != record["actor_tag"]:
                self._finish(record, ACTOR_DEAD, loop)
                continue
            if record['target_tag'] not in visible:
                self._finish(record, TARGET_LOST, loop)
                continue
            enemy_id, target = visible[record['target_tag']]
            action = self.env.n_actions_no_attack + enemy_id
            legal = (self.env.get_true_avail_agent_actions(actor_id)[action] == 1
                     and self.env.get_avail_agent_actions(actor_id)[action] == 1)
            if legal and record['actor_type'] != 'hydralisk':
                reach = self._native_reach(record['actor_type'], unit, target)
                legal = reach is not None and math.hypot(unit.pos.x - target.pos.x,
                                                         unit.pos.y - target.pos.y) <= reach
            if legal and record["first_legal_loop"] is None:
                record["first_legal_loop"] = int(loop)
                self._emit_state(record, event="micro_authorization_legal")
            if not legal:
                continue
            if self._order_targets(unit, record["target_tag"]):
                self._finish(record, ALREADY_ORDERED, loop)
                continue
            if not self.execute:
                self._finish(record, LEGAL_CONTROL_OFF, loop)
                continue
            self._submit_attack(actor_id, action, record)
            submitted += 1
            self._finish(record, SUBMITTED, loop)
        self._emit(event="micro_loop_check", loop=int(loop), offset=int(offset),
                   authorized=len(pending), submitted=submitted,
                   mode=('all' if self.all_attackers else 'hydra') if self.execute else 'off')

    def close(self, closed_loop):
        """Window boundary at L+8: no more actions; finalize open states."""
        if self._closed:
            return self.records
        self._closed = True
        for record in self.records:
            if record["status"] is None:
                self._finish(record, NEVER_LEGAL, closed_loop)
        summary = {}
        for record in self.records:
            summary[record["status"]] = summary.get(record["status"], 0) + 1
        self._emit(event="micro_window_closed", issued_loop=int(self.issued_loop),
                   closed_loop=int(closed_loop), interior_checks=self.interior_checks,
                   authorizations=len(self.records), states=summary,
                   records=[dict(r) for r in self.records])
        return self.records
