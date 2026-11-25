"""Opt-in R8 arbitration. The guard and the distill parent run once per beat."""
import hashlib
import importlib
import copy
from pathlib import Path
from types import SimpleNamespace

from .zerg_distill import Distill, require, vector
from .zerg_squad import SquadRuntime, check_options as check_guard
from .zerg_whole_ac import WholeAC, action_map

SOURCE = 'decision_tree/zerg/combat_relay_r8.py'
GUARD = '0eb07f856ba8792a2754de250cc30d809e296cd819a7d188d6403d2685688652'


def check_options(env, method):
    check_guard({k:v for k,v in env.items() if k not in ('ZERG_RELAY_MODE', 'ZERG_RELAY_SHA256')}, method)
    require(env.get('ZERG_SQUAD_MODE') == 'guard' and env.get('ZERG_SQUAD_CROSS_SHA256') == GUARD
            and env.get('ZERG_RELAY_MODE') in ('menu', 'full') and env.get('ZERG_RELAY_SHA256'),
            'relay requires frozen guard, mode and source SHA')


class RelayRuntime(SquadRuntime):
    def configure_relay(self, env, sources):
        from decision_tree.zerg import combat_relay_r8
        path = Path(combat_relay_r8.__file__).resolve()
        require(path == Path(__file__).resolve().parents[2] / SOURCE and hashlib.sha256(path.read_bytes()).hexdigest()
                == sources[SOURCE] == env['ZERG_RELAY_SHA256'], 'relay controller SHA mismatch')
        self.relay_factory = combat_relay_r8.CombatRelayR8
        self.mode = env['ZERG_RELAY_MODE']
        self.totals['relay_menu'] = 0
        return dict(kind='combat_relay_r8', mode=self.mode, controller_sha256=sources[SOURCE],
                    parent_mode='guard', parent_controller_sha256=GUARD, trial_id=self.contract['trial_id'])

    def start(self, *args):
        super().start(*args)
        self.skill = self.relay_factory(mode=self.mode)
        self.previous_owner = None
        self.counts = dict.fromkeys(('opening_steps','r6_steps','menu_steps','custom_steps','handoffs_to_r6'), 0)

    def relay_select(self, payload, masks, menu, v):
        masks = [list(map(int, m)) for m in masks]
        guard = self.squad.propose(payload, masks, list(v))
        r6, handoff = guard['actions'], guard['handoff']
        require(type(handoff) is bool and (not self.handoff or handoff) and (not handoff or tuple(r6) == v)
                and len(r6) == len(masks) and vector(action_map(r6), masks) is not None, 'invalid R6 opening')
        self.row = dict(v2_actions=list(v), actions=r6, handoff=handoff, avail_actions=masks,
                        phase=guard['phase'], reason=guard['reason'], skill_audit=guard)
        candidates = {card['candidate_id']:list(vec) for vec,(card,_) in menu.items()} if handoff else {}
        p = self.skill.propose(payload, masks, list(r6), {k:list(a) for k,a in candidates.items()})
        final = p['actions']
        require(isinstance(final, list) and len(final) == len(masks) and vector(action_map(final), masks) is not None,
                'illegal relay action')
        changed = [f'u{i}' for i,(a,b) in enumerate(zip(r6,final)) if a != b]
        owner = ('opening' if not handoff else 'r6' if not changed else 'menu' if tuple(final) in menu else 'custom')
        require((not changed or handoff) and (owner != 'custom' or self.mode == 'full'), 'unavailable relay action')
        self.relay_row = dict(p, mode=self.mode, owner=owner, candidates=candidates, avail_actions=masks,
                              changed_actors=changed, r6_actions=r6, opening_handoff=handoff and not self.handoff,
                              r6_candidate_id=menu[v][0]['candidate_id'])
        self.owner, self.handoff = owner, handoff
        self.native = owner in ('menu', 'r6')
        return tuple(final) if owner == 'menu' else v

    def step(self, step, loop, controller, payload, env):
        m, shadow = controller.manager, controller.runtime
        n = self.receipt['steps']
        require(step == n + self.squad_steps + self.counts['custom_steps'], 'missing relay receipt')
        events = []
        require('log_event' not in vars(shadow), 'overlapping relay preparation')
        shadow.log_event = lambda kind, **kw: events.append((kind,kw))
        try:
            prepared, summary = Distill.step(self, n, loop, controller, payload, env)
        finally:
            del shadow.log_event
        final = self.relay_row['actions']
        self.plans = []
        self.dropped_grants = []
        snap, plans, _, risk, ctx = self.pending
        if self.owner in ('custom', 'menu'):
            mod = importlib.import_module(type(m).__module__)
            grants = mod.merge_approaches(mod._micro_intents(ctx, summary, plans, prepared), ctx.micro_approaches)
        if self.native:
            require(final == prepared, 'native relay selection mismatch')
            self.decision.update(step=step, execution_owner=self.owner, r6_candidate_id=self.relay_row['r6_candidate_id'])
            if self.owner == 'menu':
                # The BASE entry can also be selected *back* from a v2 switch.
                # Recheck it too, against the original native ordered plans.
                check = importlib.import_module(f'{mod.__package__}.validator').complete_and_validate
                context, memory, evaluator = copy.deepcopy((ctx, m.memory, m.evaluator))
                cmds = action_map(final)
                require(not check(plans, context, cmds, report=evaluator.effects(context, cmds), memory=memory),
                        'relay menu failed complete native validation')
                self.expected_grants = grants
                self.decision.update(mode='relay_menu', r6_selection_reason=self.decision['selection_reason'],
                                     selection_reason=self.relay_row['reason'])
            self.plans = self.decision['plan_ids']
            for kind, fields in events:
                shadow.log_event(kind, **fields)
        else:
            self.discard_for_swarm(shadow)
            require(m._pending_commit is self.pending and not m.memory.committed, 'stale relay proposal')
            m.last_micro_intents = []
            if self.owner == 'custom':
                # Generate the exact native ordered grants without committing its
                # plans. Exclude actors in an atomic plan broken by the override.
                blocked = set(self.relay_row['changed_actors'])
                for plan in plans:
                    if blocked.intersection(plan.controlled_actors):
                        blocked.update(plan.controlled_actors)
                m.last_micro_intents = [g for g in grants if g['actor'] not in blocked]
                self.dropped_grants = [g for g in grants if g['actor'] in blocked]
                m.memory.commit(snap, action_map(final), [], risk, context=None)
                require(m.memory.committed and m.memory.last_committed_actions == action_map(final), 'custom memory mismatch')
            else:
                self.external = (snap, risk, action_map(final))
            m._pending_commit = m._cache = self.pending = self.decision = None
            controller.last_meta = controller._pending_actions = None
        return final, summary

    def committed(self, manager, actions):
        super().committed(manager, actions)
        if self.owner == 'menu':
            require(manager.last_micro_intents == self.expected_grants, 'native ordered micro grants changed')

    def after_engine(self, manager, feedback, step, loop):
        if self.owner == 'opening':
            self.acknowledge_squad(manager, feedback)
        elif self.owner == 'custom':
            # Use the same strict macro/micro authorization checks as native
            # decisions, without attributing an external beat to distillation.
            final = self.relay_row['actions']
            receipt = dict(steps=0, engine_unconfirmed=0, episode=self.receipt['episode'])
            checker = SimpleNamespace(receipt=receipt, grants={g['actor']:g for g in manager.last_micro_intents},
                micro_clock=self.micro_clock, decision=dict(step=step, game_loop=loop,
                final_actions=action_map(final), committed=True))
            WholeAC.engine(checker, feedback)
            require(not receipt['engine_unconfirmed'], 'unconfirmed custom window')
        else:
            self.engine(feedback)
            require(self.decision['macro_receipt'] == 'accepted' and self.decision['micro_receipt'] != 'unknown',
                    'unconfirmed relay native window')
        handoff = self.previous_owner in ('custom', 'menu') and self.owner == 'r6'
        self.counts[self.owner+'_steps'] += 1
        self.counts['handoffs_to_r6'] += handoff
        self.previous_owner = self.owner
        return dict(owner=self.owner, actions=self.relay_row['actions'], plan_ids=self.plans,
                    micro_grants=manager.last_micro_intents, micro_submissions=feedback['micro_submissions'],
                    dropped_r6_micro_grants=self.dropped_grants, handoff_to_r6=handoff)

    def finish(self, steps):
        c = self.counts
        n = self.receipt['steps']
        require(self.external is None and n == c['r6_steps'] + c['menu_steps'] and self.squad_steps == c['opening_steps']
                and steps == n + self.squad_steps + c['custom_steps'], 'incomplete relay episode')
        self.receipt['scope'] = 'executed_native_relay_steps_only'
        self.receipt['combat_relay'] = dict(c)
        if n:
            Distill.finish(self, n)
            self.receipt['stats']['relay_menu'] = c['menu_steps']
            self.totals['relay_menu'] += c['menu_steps']
            self.save()
        else:
            require(not self.receipt['decisions'], 'unexpected native receipt')
            self.receipt.update(status='complete', stats=dict.fromkeys(self.totals, 0))
            self.save()
