"""Exploratory whole-episode selector. No anchor, model, reward or outcome input."""
import copy
import hashlib
import importlib
import json
from pathlib import Path

from decision_tree.zerg import rules_v1 as rule
from decision_tree.zerg.candidates import joint_candidates_from_summary


SELECTOR = 'exploratory_prefer_certified_novel_c_v1'


def require(ok, message):
    if not ok:
        raise ValueError(f'Zerg whole A/C: {message}')


def action_map(actions):
    return {f'u{i}': a for i, a in enumerate(actions)}


def select(parent, offered):
    """Only a certified extension may join the parent's candidate menu."""
    return offered if offered is not None else parent


def parent_step(controller, payload, n):
    # Native _summary mutates telemetry. Save its pre-selection state so the
    # winning entry can use the same renderer without running search twice.
    coord = controller.manager.coordinator
    native, render = coord._summary, None

    def capture(final, ctx, telemetry, *args):
        nonlocal render
        private = copy.deepcopy(telemetry)
        render = lambda e: (native(e, ctx, private, *args), private)
        return native(final, ctx, telemetry, *args)

    require('_summary' not in vars(coord), 'unexpected coordinator override')
    coord._summary = capture
    try:
        actions, summary = controller.step(payload, n)
    finally:
        del coord._summary
    return actions, summary, render


def certify(manager, entry, base, record, env):
    if entry is None:
        return None
    cache = manager.coordinator._cache
    snap, _, _, _, ctx = manager._pending_commit
    mod = importlib.import_module(type(manager).__module__)
    coord = importlib.import_module(type(manager.coordinator).__module__)
    require(isinstance(entry, coord._CompletePlan) and not base['local_coord']['selected'],
            'not a complete independent native branch')
    actions, plans = entry.actions, entry.plans
    require(set(actions) == {f'u{i}' for i in range(env.n_agents)} and all(type(a) is int for a in actions.values()),
            'incomplete or noninteger candidate')
    if (actions == base['actions'] or any(e.actions == actions for e in cache.values())
            or any(c['proposed_actions'] == actions for c in record['candidates'])):
        return None
    for i in range(env.n_agents):
        mask, a = env.get_avail_agent_actions(i), actions[f'u{i}']
        if not 0 <= a < len(mask) or not mask[a]:
            return None
    require(plans and all(isinstance(p, coord.CandidatePlan) and p.snapshot_id == snap.snapshot_id
            and p.primary_skill not in ('local_engagement', 'reserve_dispatch') and p.controlled_actors
            and set(p.controlled_actors) == set(p.current_actions)
            and all(actions.get(u) == a for u, a in p.current_actions.items()) for p in plans),
            'invalid native C plans')
    ledger = coord.Ledger()
    for p in plans:
        require(p.proposal_id not in ledger.plans and not ledger.conflicts(p), 'conflicting native C plans')
        ledger.reserve(p)
    # Native effects/duty checks may populate lazy caches even on failure.
    ctx, memory, evaluator = copy.deepcopy((ctx, manager.memory, manager.evaluator))
    validate = importlib.import_module(f'{mod.__package__}.validator').complete_and_validate
    problems = validate(plans, ctx, actions, report=evaluator.effects(ctx, actions), memory=memory)
    return None if problems else entry


def install(controller, entry, base, record, render):
    manager, rt = controller.manager, controller.runtime
    snap, _, _, risk, ctx = manager._pending_commit
    mod = importlib.import_module(type(manager).__module__)
    summary, telemetry = render(entry)
    actions, plans = [entry.actions[f'u{i}'] for i in range(len(entry.actions))], list(entry.plans)
    require(summary['actions'] == entry.actions == {u: r['action'] for u, r in summary['units'].items()}
            and [p.proposal_id for p in plans] == [p['id'] for p in summary['chosen_plans']], 'branch render mismatch')
    for key in ('battle_intent', 'splash_model', 'local_coord'):
        if key in base:
            summary[key] = base[key]
    summary['plan_evaluations_used'] += summary['local_coord']['local_evaluations']
    summary['whole_ac'] = dict(selector=SELECTOR, selected_candidate=rule.RULE_ID)
    candidate, = joint_candidates_from_summary({**summary, 'top_alternates': []}, snap, ctx)
    require(candidate.legal and all(c['candidate_id'] != rule.RULE_ID for c in record['candidates']),
            'invalid shadow candidate')
    record['candidates'].append(dict(candidate.to_record(), candidate_id=rule.RULE_ID, source_plan_id=rule.RULE_ID,
                                     origin='whole_ac_C', rule_hash=rule.rule_hash()))
    record.update(selected_candidate=rule.RULE_ID, final_actions=dict(entry.actions))
    controller.last_meta['selected_candidate'] = rule.RULE_ID
    if record.get('skill_proposals'):
        record['skill_proposals'] = rt._enriched_skill_proposals(snap, controller.last_meta, record['final_actions'],
                                                                 record['execution_status'])
    rt.log_event('submitted', episode_id=record['episode_id'], snapshot_id=snap.snapshot_id, game_loop=snap.game_loop,
                 reason=SELECTOR, selected_candidate=rule.RULE_ID, final_actions=entry.actions)
    manager.telemetry = telemetry
    manager.stats['baseline_units'] += sum((r['skill'] == 'baseline') - (base['units'][u]['skill'] == 'baseline')
                                            for u, r in summary['units'].items() if snap.allies[u].alive)
    ctx.micro_approaches = mod.extra_approaches(ctx, summary, plans, actions) if manager.micro_all_attackers else []
    manager.last_micro_intents = []
    manager._cache = (manager._cache[0], list(actions), copy.deepcopy(summary), plans, snap)
    manager._pending_commit = (snap, plans, list(actions), risk, ctx)
    return actions, summary


class WholeAC:
    def __init__(self, output, arm, shadow):
        require(arm in ('A', 'C'), 'unknown arm')
        self.output, self.arm, self.receipt = output, arm, None
        self.contract = dict(kind='zerg_whole_ac_v1', arm=arm, selector=SELECTOR, exploratory_only=True,
                             selector_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             rule_id=rule.RULE_ID, rule_hash=rule.rule_hash(), roster=shadow['scene_config_id'],
                             source_hash=shadow['source_fingerprint'], execution_context_hash=shadow['execution_context_hash'])

    def start(self, episode, scenario, initial, version):
        self.receipt = dict(self.contract, episode=episode, scenario_sha256=scenario['sha256'], seed=scenario['sc2_seed'],
                            initial_state_sha256=initial, game_version=version, steps=0, proposals=0, offered=0,
                            applied=0, engine_unconfirmed=0, decisions=[], status='pending')
        self.save()

    def save(self):
        body = json.dumps(self.receipt, indent=2, allow_nan=False) + '\n'
        for path in (self.output / 'whole_ac' / f"episode-{self.receipt['episode']:04d}.json", self.output / 'whole_ac.json'):
            path.parent.mkdir(exist_ok=True)
            tmp = path.with_suffix('.tmp')
            tmp.write_text(body, encoding='utf-8')
            tmp.replace(path)

    def step(self, step, loop, controller, payload, env):
        r, m, rt = self.receipt, controller.manager, controller.runtime
        require(step == len(r['decisions']) == r['steps'], 'overlapping or missing step')
        actions, base, render = parent_step(controller, payload, env.n_agents)
        pending = m._pending_commit
        snap = pending[0]
        require(not m.memory.committed and snap is m._cache[4] is pending[4].snapshot and
                snap.snapshot_id == base['snapshot_id'] and snap.game_loop == loop == base['snapshot_game_loop'] and
                base['actions'] == action_map(actions) == action_map(pending[2]), 'stale parent')
        row = rt._buffer[(rt.run_id, str(r['episode']), snap.snapshot_id, loop)]
        require(row['teacher_choice'] == row['selected_candidate'] and row['final_actions'] == base['actions'],
                'shadow parent mismatch')
        entry = certify(m, rule.make_entry(m, base, payload), base, row, env)
        offered = entry if self.arm == 'C' else None
        selected = select(None, offered)  # None denotes keeping the native decision, untouched.
        result = install(controller, selected, base, row, render) if selected is not None else (actions, base)
        self.pending = m._pending_commit
        self.decision = dict(step=step, game_loop=loop, snapshot_id=snap.snapshot_id, proposed=entry is not None,
                             offered=offered is not None, applied=False, parent_actions=dict(base['actions']),
                             proposal_actions=dict(entry.actions) if entry is not None else None,
                             final_actions=action_map(result[0]), plan_ids=[p['id'] for p in result[1]['chosen_plans']],
                             committed=False)
        r['decisions'].append(self.decision)
        r['proposals'] += self.decision['proposed']
        r['offered'] += self.decision['offered']
        return result

    def before_commit(self, manager, actions):
        d = self.decision
        snap, plans, planned, _, ctx = self.pending
        require(manager._pending_commit is self.pending and not manager.memory.committed and
                snap is ctx.snapshot is manager._cache[4] and plans is manager._cache[3] and
                planned == actions == manager._cache[1] and action_map(actions) == d['final_actions'] and
                [p.proposal_id for p in plans] == d['plan_ids'] and all(p.snapshot_id == snap.snapshot_id and
                all(d['final_actions'][u] == p.current_actions[u] for u in p.controlled_actors) for p in plans),
                'COMMIT pending actions/plans differ from selection')

    def committed(self, manager, actions):
        d, mem = self.decision, manager.memory
        require(not d['committed'] and manager._pending_commit is None and mem.committed and
                mem.last_committed_loop == d['game_loop'] and mem.last_committed_actions == action_map(actions)
                == d['final_actions'], 'native COMMIT not confirmed')
        d.update(committed=True, applied=d['offered'])
        self.receipt['applied'] += d['applied']
        self.grants = {g['actor']: g for g in manager.last_micro_intents}
        self.pending = None

    @staticmethod
    def micro_clock(sub):
        return sub['issued_loop'] < sub['submitted_loop'] < sub['expires_loop'] == sub['issued_loop'] + 8

    def engine(self, feedback):
        d, final = self.decision, self.decision['final_actions']
        require(d['committed'] and 'macro_receipt' not in d, 'receipt without fresh commit')
        start, actors = feedback.get('from_game_loop'), feedback.get('actors') or {}
        require(start in (None, d['game_loop']), 'stale macro receipt')
        raw = feedback.get('raw_response_results', [])
        rejected = any(c != 1 for c in raw)
        known = start is not None and feedback.get('response_mapping') == 'exact' and actors.keys() == final.keys()
        for u, a in final.items():
            row = actors.get(u, {})
            requested, code = row.get('requested_action_id'), (row.get('response') or {}).get('code')
            require(requested is None or str(requested) == str(a), 'macro requested action mismatch')
            rejected |= row.get('submission') == 'rejected' or code not in (None, 1)
            known &= requested is not None and ((row.get('submission') == 'dead_noop' and a == 0 and
                      row.get('command') is None and code is None) or (row.get('submission') == 'accepted' and
                      bool(row.get('command')) and code == 1))
        d['macro_receipt'] = 'rejected' if rejected else 'accepted' if known and any(final.values()) else 'unknown'
        micro = feedback.get('micro_submissions', [])
        d['micro_receipt'] = 'not_submitted' if not micro else 'accepted'
        seen = set()
        for sub in micro:
            u = sub['actor']
            grant = self.grants.get(u)
            require(grant and u not in seen and sub['status'] == 'submitted' and sub['macro_step'] == d['step'] and
                    sub['episode'] == self.receipt['episode'] and sub['issued_loop'] == d['game_loop'] and
                    self.micro_clock(sub) and
                    sub['macro_action_id'] == final[u] and all(sub[k] == grant[k] for k in
                    ('actor_tag', 'target', 'target_tag', 'issued_loop', 'macro_action_id')) and
                    sub['target'] == f"e{sub['submitted_action_id'] - 6}", 'micro command not authorized by selected plans')
            seen.add(u)
            codes = sub.get('sc2_response') or []
            if any(c != 1 for c in codes):
                d['micro_receipt'] = 'rejected'
            elif codes != [1] and d['micro_receipt'] != 'rejected':
                d['micro_receipt'] = 'unknown'
        require('rejected' not in (d['micro_receipt'], d['macro_receipt']), 'engine rejected submitted command')
        self.receipt['engine_unconfirmed'] += 'unknown' in (d['micro_receipt'], d['macro_receipt'])
        self.receipt['steps'] += 1

    def finish(self, steps):
        require(steps > 0 and steps == self.receipt['steps'] == len(self.receipt['decisions']), 'missing steps')
        self.receipt['status'] = 'complete'
        self.save()

    def fail(self, error):
        if self.receipt is not None and self.receipt['status'] != 'complete':
            self.receipt.update(status='failed', error=error)
            self.save()
