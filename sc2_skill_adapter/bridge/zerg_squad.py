"""Opt-in macro-only Squad prefix followed by an ordinary frozen-v2 suffix."""
import hashlib
from pathlib import Path

from .zerg_distill import Distill, require
from .zerg_whole_ac import action_map

SOURCE = 'decision_tree/zerg/squad_rendezvous.py'
CROSS_SOURCE = 'decision_tree/zerg/squad_cross_support.py'
CROSS_MODES = ('guard', 'front')
MODES = ('adaptive', 'fixed') + CROSS_MODES
# Parent implementation in the completed adaptive old-900 design.
ADAPTIVE_SHA256 = '7324aa4bc69b2df33e61c999e74a106372eb629d83b54edb2f0da2068af73ac4'
FROZEN = dict(policy_version='v2',
    model_sha256='1243710f83f14a171799855befb804122591acd150b0477dfd759b07641eb578',
    selector_sha256='0782035b9200bd8162fc6908152cec0a11129df345bb0549636a29ce0f968b2b',
    opening_sha256='e02d7f9d730e29c6082b5005ab0e04730cc6e8cb1fafdd9a7532fe867d2562b8',
    source_hash='2f135a09b7374512731583c2b511425066b4fa78a64493b842676a1115d302a7',
    execution_context_hash='1cbf9c466265f041b1fa13d1e90a2b49878a5593a044fb39cf0b7ea37c191ee4')


def check_options(environ, method):
    allowed = {'ZERG_SQUAD_MODE', 'ZERG_SQUAD_SHA256', 'ZERG_DISTILL_MODEL', 'ZERG_DISTILL_MODEL_SHA256',
               'ZERG_DISTILL_SELECTOR_SHA256', 'ZERG_DISTILL_OPENING_SHA256', 'ZERG_DISTILL_VERSION',
               'ZERG_DISTILL_TRIAL_ID', 'ZERG_SHADOW_DIR', 'ZERG_SHADOW_ROSTER'}
    if environ.get('ZERG_SQUAD_MODE') in CROSS_MODES:
        allowed.add('ZERG_SQUAD_CROSS_SHA256')
    require(method == 'adaptive-skill' and environ.get('ZERG_SQUAD_MODE') in MODES
            and environ.get('ZERG_DISTILL_VERSION') == 'v2' and all(environ.get(k) for k in allowed)
            and not any(k.startswith('ZERG_') and k not in allowed for k in environ),
            'Squad requires an explicit mode, controller hash and exclusive full v2 parent')


def controller_class(mode):
    if mode in CROSS_MODES:
        from decision_tree.zerg.squad_cross_support import CrossSupportRendezvous
        return CrossSupportRendezvous
    require(mode in MODES, 'unknown squad mode')
    from decision_tree.zerg.squad_rendezvous import SquadRendezvous
    return SquadRendezvous


def squad_contract(mode, sources, trial_id):
    if mode in CROSS_MODES:
        require(sources[SOURCE] == ADAPTIVE_SHA256, 'changed adaptive parent')
        return dict(kind='squad_cross_support_v1', mode=mode, controller_sha256=sources[CROSS_SOURCE],
                    parent_mode='adaptive', parent_controller_sha256=sources[SOURCE], trial_id=trial_id)
    return dict(kind='squad_rendezvous_v1', mode=mode, controller_sha256=sources[SOURCE], trial_id=trial_id)


class SquadRuntime(Distill):
    def configure(self, environ, sources):
        from decision_tree.zerg import squad_rendezvous
        src = Path(squad_rendezvous.__file__).resolve()
        require(src == Path(__file__).resolve().parents[2] / SOURCE and hashlib.sha256(src.read_bytes()).hexdigest()
                == sources[SOURCE] == environ['ZERG_SQUAD_SHA256'], 'Squad controller SHA mismatch')
        require(all(self.contract.get(k) == v for k, v in FROZEN.items()), 'not frozen v2')
        mode = environ['ZERG_SQUAD_MODE']
        if mode in CROSS_MODES:
            from decision_tree.zerg import squad_cross_support
            src = Path(squad_cross_support.__file__).resolve()
            require(src == Path(__file__).resolve().parents[2] / CROSS_SOURCE and hashlib.sha256(src.read_bytes()).hexdigest()
                    == sources[CROSS_SOURCE] == environ['ZERG_SQUAD_CROSS_SHA256'], 'cross-support controller SHA mismatch')
        self.factory = controller_class(mode)
        self.squad_contract = squad_contract(mode, sources, self.contract['trial_id'])
        return self.squad_contract

    def start(self, *args):
        super().start(*args)
        self.squad = self.factory(mode=self.squad_contract['mode'])
        self.squad_steps = 0
        self.handoff = False
        self.external = None

    def step(self, step, loop, controller, payload, env):
        shadow = controller.runtime
        n = self.receipt['steps']
        require(step == n + self.squad_steps and self.external is None, 'missing Squad macro receipt')
        # Shadow logs its proposal as submitted inside prepare. Only publish those
        # events if this really is a native beat, including the handoff itself.
        events = []
        publish = shadow.log_event
        shadow.log_event = lambda kind, **fields: events.append((kind, fields))
        try:
            base, summary = super().step(n, loop, controller, payload, env)
        finally:
            del shadow.log_event
        masks = [list(map(int, env.get_avail_agent_actions(i))) for i in range(env.n_agents)]
        p = self.squad.propose(payload, masks, list(base))
        actions, handoff = p['actions'], p['handoff']
        require(type(handoff) is bool and (not self.handoff or handoff) and (not handoff or actions == base)
                and all(isinstance(p[k], str) and p[k] for k in ('phase', 'reason')), 'invalid Squad handoff')
        require(isinstance(actions, list) and len(actions) == len(masks) and all(type(a) is int and
                0 <= a < len(m) and m[a] for a, m in zip(actions, masks)), 'invalid Squad macro')
        self.handoff = handoff
        self.row = dict(v2_actions=list(base), avail_actions=masks, actions=list(actions), handoff=handoff,
                        phase=p['phase'], reason=p['reason'])
        if self.squad_contract['mode'] in CROSS_MODES:
            self.row['skill_audit'] = p
        if handoff:
            self.decision['step'] = step  # Micro grants and receipts use the real episode clock.
            for kind, fields in events:
                publish(kind, **fields)
        else:
            m = controller.manager
            require(m._pending_commit is self.pending and not m.memory.committed, 'stale v2 proposal')
            self.discard_for_swarm(shadow)
            snap, _, _, risk, _ = self.pending
            self.external = (snap, risk, action_map(actions))
            m._pending_commit = m._cache = self.pending = self.decision = None
            m.last_micro_intents = []
            controller.last_meta = controller._pending_actions = None
        return actions, summary

    def acknowledge_squad(self, manager, feedback):
        snap, risk, actions = self.external
        actors = feedback['actors']
        require(not feedback['micro_submissions'] and feedback['response_mapping'] == 'exact' and
                feedback['from_game_loop'] == snap.game_loop and actors.keys() == actions.keys() and
                all(r == 1 for r in feedback['raw_response_results']), 'unconfirmed Squad window')
        for u, a in actions.items():
            r = actors[u]
            require(r['requested_action_id'] == str(a) and ((a == 0 and r['submission'] == 'dead_noop' and
                    r['command'] is None and r['response'] is None) or (a != 0 and r['submission'] == 'accepted'
                    and r['command'] and r['response']['code'] == 1)), 'unconfirmed Squad macro')
        # No native plans OR tactical context: context could infer a deployment
        # task from a coincidentally matching move. Persist actual command facts only.
        manager.memory.commit(snap, actions, [], risk, context=None)
        self.external = None
        self.squad_steps += 1

    def finish(self, steps):
        n = self.receipt['steps']
        require(self.external is None and steps == n + self.squad_steps, 'incomplete Squad run')
        self.receipt['scope'] = 'executed_v2_suffix_only'
        if n:
            super().finish(n)
        else:
            require(not self.receipt['decisions'], 'unexecuted v2 decisions')
            self.receipt.update(status='complete', stats=dict.fromkeys(self.totals, 0))
            self.save()
