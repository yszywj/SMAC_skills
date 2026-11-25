"""Per-macro joint policy distillation on the original reported Shadow menu."""
from collections import Counter
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path

from .zerg_fork import install_pending
from .zerg_whole_ac import WholeAC, action_map, parent_step


def require(ok, message):
    if not ok:
        raise ValueError(f'Zerg distill: {message}')


def check_protocol(metadata):
    from decision_tree.scripts.linux.collect_zerg_shadow import TEMPLATE, historical_config
    config, versions = historical_config()
    parent = json.loads((TEMPLATE / 'run.json').read_text())
    for k in ('method', 'skill_version', 'skill_parameters', 'observation_protocol', 'macro_loops', 'engine_step_loops',
              'execution_mode', 'damage_by_type_enabled'):
        require(metadata[k] == parent[k], f'historical protocol mismatch: {k}')
    require(metadata['env_args'] == config and metadata['versions'] == versions, 'historical environment mismatch')
    require(hashlib.sha256(Path(metadata['map_file']).read_bytes()).hexdigest() ==
            parent['source_hashes']['sc2_skill_adapter/damage_maps/10gen_zerg.SC2Map'], 'historical map content mismatch')


def vector(actions, masks):
    if set(actions) != {f'u{i}' for i in range(len(masks))}:
        return None
    v = tuple(actions[f'u{i}'] for i in range(len(masks)))
    return v if all(type(a) is int and 0 <= a < len(mask) and mask[a] for a, mask in zip(v, masks)) else None


def branch(manager, base, alt, render):
    if base['local_coord']['selected'] or any(p['skill'] in ('local_engagement', 'reserve_dispatch') for p in alt['plans']):
        return None, 'private_local'
    if not alt['plans']:
        return None, 'empty_plan'
    coord = manager.coordinator
    matches = [e for e in coord._cache.values() if e.actions == alt['actions']]
    if not matches or render is None:
        return None, 'no_native_branch'
    entry = min(matches, key=lambda e: coord._rank(e, manager.memory))
    native = importlib.import_module(type(coord).__module__)
    snap, _, _, _, ctx = manager._pending_commit
    if (not isinstance(entry, native._CompletePlan) or [dict(id=p.proposal_id, skill=p.primary_skill,
            actors=list(p.controlled_actors), goal=p.goal) for p in entry.plans] != alt['plans']):
        return None, 'plan_identity'
    ledger = native.Ledger()
    for p in entry.plans:
        if (not isinstance(p, native.CandidatePlan) or p.snapshot_id != snap.snapshot_id or not p.controlled_actors
                or set(p.current_actions) != set(p.controlled_actors) or p.proposal_id in ledger.plans
                or ledger.conflicts(p) or any(entry.actions.get(u) != a for u, a in p.current_actions.items())):
            return None, 'plan_conflict'
        ledger.reserve(p)
    if any(entry.actions.get(u) != a for u, a in getattr(ctx, 'formation_orders', {}).items()
           if u not in {u for p in entry.plans for u in p.controlled_actors}):
        return None, 'formation_order'
    validator = importlib.import_module(f'{native.__package__}.validator')
    ctx, mem, evaluator = copy.deepcopy((ctx, manager.memory, manager.evaluator))
    if validator.complete_and_validate(entry.plans, ctx, entry.actions, report=evaluator.effects(ctx, entry.actions), memory=mem):
        return None, 'validator'
    return entry, None


class Distill(WholeAC):
    """Reuse the whole-runner's pending/commit and macro/micro receipt checks."""
    def __init__(self, output, path, shadow):
        from decision_tree.zerg import policy_distill
        path = Path(path).resolve(strict=True)
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        self.model = policy_distill.load_model(path)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == h == os.environ.get('ZERG_DISTILL_MODEL_SHA256', h),
                'model file hash mismatch')
        require(self.model['source_hash'] == shadow['source_fingerprint'] and
                self.model['ctx_hash'] == shadow['execution_context_hash'], 'model source/execution context mismatch')
        self.choose = policy_distill.choose
        self.version = os.environ.get('ZERG_DISTILL_VERSION', 'v1')
        self.routed = 'ZERG_ROUTE_IDENTITY' in os.environ
        require(self.version in (('v2', 'alternative', 'route') if self.routed else ('v1','v2','v3')), 'unknown selector version')
        selector = 'policy_distill'
        if self.version == 'v2' or self.routed:
            from decision_tree.zerg.policy_repair_v2 import v2_choose
            self.choose = v2_choose
            selector = 'policy_repair_v2'
            if self.version in ('alternative', 'route'):
                selector = 'policy_route_alt'
        elif self.version == 'v3':
            from decision_tree.zerg.policy_repair_v3 import v3_choose
            self.choose = v3_choose
            selector = 'policy_repair_v3'
        source = Path(__file__).resolve().parents[2] / f'decision_tree/zerg/{selector}.py'
        selector_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        require(selector_sha == os.environ.get('ZERG_DISTILL_SELECTOR_SHA256', selector_sha), 'selector SHA mismatch')
        self.output, self.receipt = output, None
        self.contract = dict(kind=f'zerg_distill_{self.version}', policy_version=self.version,
            selector_sha256=selector_sha, model_sha256=h, source_hash=self.model['source_hash'],
            execution_context_hash=self.model['ctx_hash'], projection_version=self.model['schema_hash'],
            split_hash=self.model['split_hash'], roster=shadow['scene_config_id'])
        self.generator = None
        opening_sha = os.environ.get('ZERG_DISTILL_OPENING_SHA256')
        if self.version == 'v3' or 'ZERG_DISTILL_TRIAL_ID' in os.environ:
            require(opening_sha and os.environ.get('ZERG_DISTILL_TRIAL_ID'), 'full trial requires opening and trial ID')
            self.contract['trial_id'] = os.environ['ZERG_DISTILL_TRIAL_ID']
        if opening_sha:
            from decision_tree.zerg import opening_support_v2
            require((self.routed or self.version in ('v2', 'v3')) and hashlib.sha256(Path(opening_support_v2.__file__).read_bytes()).hexdigest() == opening_sha,
                    'opening generator version mismatch')
            self.generator = opening_support_v2
            self.contract['opening_sha256'] = opening_sha
        if self.routed:
            from decision_tree.zerg import policy_route_alt, policy_route_gate
            self.alt_choose = policy_route_alt.choose
            self.gate = policy_route_gate
            self.identity = json.loads(os.environ['ZERG_ROUTE_IDENTITY'])
            require(opening_sha and self.contract.get('trial_id') and all(self.identity[k] == val for k,val in {
                'opening':opening_sha, 'model':h, 'parent':self.model['source_hash'],
                'v2':hashlib.sha256(source.with_name('policy_repair_v2.py').read_bytes()).hexdigest(),
                'alternative':hashlib.sha256(Path(policy_route_alt.__file__).read_bytes()).hexdigest(),
            }.items()), 'route policy identity mismatch')
            self.contract['policy_sha256'] = self.identity
            if self.version == 'route':
                raw = Path(os.environ['ZERG_ROUTE_MODEL']).read_bytes()
                self.gate_model = json.loads(raw)
                self.contract['gate_sha256'] = hashlib.sha256(raw).hexdigest()
                require(self.contract['gate_sha256'] == os.environ['ZERG_ROUTE_MODEL_SHA256'] and
                        self.gate_model['schema'] == 'zerg_route_gate_v1' and self.gate_model['policy_sha256'] == self.identity,
                        'invalid gate model')
        self.totals = dict.fromkeys(('steps', 'model_base', 'model_switch', 'fallback', 'filtered_candidates', 'engine_unconfirmed'), 0)

    def save(self):
        path = self.output / 'distill' / f"episode-{self.receipt['episode']:04d}.json"
        path.parent.mkdir(exist_ok=True)
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.receipt, allow_nan=False) + '\n', encoding='utf-8')
        tmp.replace(path)

    def start(self, episode, scenario, initial, version):
        require(version == '4.10.0.75689', 'historical SC2 build mismatch')
        self.route_choice = None
        self.receipt = dict(self.contract, episode=episode, scenario_sha256=scenario['sha256'], seed=scenario['sc2_seed'],
            initial_state_sha256=initial, game_version=version, decisions=[], steps=0, engine_unconfirmed=0, applied=0,
            status='pending')
        self.save()

    def step(self, step, loop, controller, payload, env):
        m, rt, r = controller.manager, controller.runtime, self.receipt
        require(step == r['steps'] == len(r['decisions']), 'missing decision')
        actions, base, render = parent_step(controller, payload, env.n_agents)
        snap, plans, planned, _, ctx = m._pending_commit
        require(snap is ctx.snapshot is m._cache[4] and snap.game_loop == base['snapshot_game_loop'] == loop
                and base['snapshot_id'] == snap.snapshot_id and m.micro_all_attackers and not m.memory.committed
                and actions == planned and action_map(actions) == base['actions'], 'stale parent')
        row = rt._buffer[(rt.run_id, str(r['episode']), snap.snapshot_id, loop)]
        require(row['source_hash'] == self.contract['source_hash'] and row['execution_context_hash'] == self.contract['execution_context_hash']
                and row['teacher_choice'] == row['selected_candidate'] and row['final_actions'] == base['actions'], 'shadow mismatch')
        masks = [env.get_avail_agent_actions(i) for i in range(env.n_agents)]
        basevec = vector(base['actions'], masks)
        cards = row['candidates']
        require(basevec is not None and cards and cards[0]['candidate_id'] == row['teacher_choice'] and
                cards[0]['proposed_actions'] == base['actions'] and cards[0]['legal'] is True, 'invalid BASE')
        alts = {a['id']: a for a in base['top_alternates']}
        require(all(c['candidate_family'] == 'joint_plan' for c in cards) and
                len(alts) == len(cards)-1 and [a['id'] for a in base['top_alternates']] ==
                [c['source_plan_id'] for c in cards[1:]], 'not original joint summary menu')
        menu, filtered = {basevec: (cards[0], None)}, []
        for c in cards[1:]:
            alt = alts[c['source_plan_id']]
            require(c['proposed_actions'] == alt['actions'], 'candidate actions differ')
            v = vector(alt['actions'], masks)
            if v is None or c['legal'] is not True:
                entry, reason = None, 'illegal'
            elif v in menu:
                entry, reason = None, 'duplicate'
            else:
                entry, reason = branch(m, base, alt, render)
            if reason:
                filtered.append(dict(candidate_id=c['candidate_id'], reason=reason))
            else:
                menu[v] = (c, entry)
        # Use the pre-observation's actual previous commands, including accepted
        # interior micro orders, exactly as in the historical fit.
        candidates = [list(v) for v in menu]
        if self.version == 'v3':
            result = self.choose(self.model, payload, candidates, previous_actions=None)
            audit = result['audit']
            v2_vec = tuple(audit['v2_actions'])
            require(v2_vec in menu and type(audit['v3_triggered']) is bool and
                    (audit['v3_triggered'] or list(v2_vec) == result['actions']), 'invalid v3 reference')
            if audit['v2_reason'] == 'no_existing_formation_fix':
                require(result['reason'] == audit['v2_reason'] and result['actions'] == list(v2_vec),
                        'v3 suppressed shared opening branch')
        else:
            result = self.choose(self.model, payload, candidates)
        reason = result['reason'] if self.version != 'v1' else 'v1_cart_argmax'
        generated = False
        if getattr(self, 'generator', None) and reason == 'no_existing_formation_fix' and render is not None and not base['local_coord']['selected']:
            from .zerg_whole_ac import certify
            reference_vec = tuple(result['actions'])
            ref = menu[reference_vec][1]
            alternative = self.generator.make_entry(m, base, payload, reference=ref)
            entry = certify(m, alternative, base, row, env)
            if entry is not None:
                generated = True
                vec = vector(entry.actions, masks)
                require(vec is not None and vec not in menu, 'invalid extra opening proposal')
                cid = self.generator.RULE_ID
                require(all(c['candidate_id'] != cid for c in cards), 'duplicate opening proposal identifier')
                menu[vec] = ({'candidate_id':cid},entry)
                # Priority branch, not an estimated reward or an additional CART prediction.
                priority = max(result['scores'])+2.
                result = dict(result, actions=list(vec), scores=[*result['scores'],priority], score=priority,
                              top_ties=1, unique_count=len(menu), audit=dict(result['audit'],
                              before_opening_actions=list(reference_vec), priority_branch=True))
                reason = 'generated_opening_support'
        if self.routed:
            # Resolve the full v2 opening once, before considering an alternative.
            alt = result if generated else self.alt_choose(self.model, payload, candidates, previous_actions=None)
            def plan(res):
                vec = tuple(res['actions'])
                require(vec in menu, 'route returned unavailable plan')
                card, entry = menu[vec]
                return dict(actions=list(vec), candidate_id=card['candidate_id'],
                            plan_ids=[p.proposal_id for p in (entry.plans if entry is not None else plans)])
            parent, other = plan(result), plan(alt)
            anchor = None
            if self.route_choice is None and parent != other:
                require(parent['actions'] != other['actions'], 'same actions with different plan semantics')
                serialized = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
                features = self.gate.anchor_features(payload, parent, other)
                anchor = dict(step=step, game_loop=loop, features=features, parent_plan=parent, alt_plan=other,
                              payload_sha256=hashlib.sha256(serialized).hexdigest(), policy_sha256=self.identity)
                self.route_choice = self.gate.choose_route(self.gate_model, features) if self.version == 'route' else self.version
                require(self.route_choice in ('v2','alternative'), 'invalid route decision')
            if self.route_choice == 'alternative':
                result = alt
                if not generated:
                    reason = result['reason']
        require(isinstance(reason, str) and reason.strip(), 'missing selector reason')
        v = tuple(result['actions'])
        scores = result['scores']
        require(v in menu and len(scores) == len(menu) == result['unique_count'] and all(math.isfinite(x) for x in scores)
                and scores[list(menu).index(v)] == result['score'] and abs(max(scores) - result['score']) <= 1e-12,
                'invalid model result')
        # Explicit R8 only: expose the certified, deduplicated native menu,
        # after resolving the frozen selector (including its opening branch).
        if hasattr(self, 'relay_select'):
            selected = self.relay_select(payload, masks, menu, v)
            if selected != v:
                generated = False
            v = selected
        card, entry = menu[v]
        mode = 'fallback' if len(menu) == 1 else 'model_base' if v == basevec else 'model_switch'
        summary = base
        if entry is not None:
            summary, telemetry = render(entry)
            actions, plans = list(v), list(entry.plans)
            require(summary['actions'] == entry.actions == {u: x['action'] for u, x in summary['units'].items()} and
                     [p.proposal_id for p in plans] == [p['id'] for p in summary['chosen_plans']], 'native render mismatch')
            if generated:
                from decision_tree.zerg.candidates import joint_candidates_from_summary
                new, = joint_candidates_from_summary({**summary,'top_alternates':[]},snap,ctx,payload=payload)
                require(new.legal and new.proposed_actions == entry.actions,'invalid generated shadow candidate')
                row['candidates'].append(dict(new.to_record(),candidate_id=card['candidate_id'],
                    source_plan_id=card['candidate_id'],origin='distill_opening_support',rule_hash=self.contract['opening_sha256']))
            for k in ('battle_intent', 'splash_model', 'local_coord'):
                if k in base:
                    summary[k] = base[k]
            summary['plan_evaluations_used'] += summary['local_coord']['local_evaluations']
            install_pending(m, base, summary, telemetry, plans, actions)
            row.update(selected_candidate=card['candidate_id'], final_actions=dict(entry.actions))
            controller.last_meta['selected_candidate'] = card['candidate_id']
            if row.get('skill_proposals'):
                row['skill_proposals'] = rt._enriched_skill_proposals(snap, controller.last_meta, row['final_actions'], row['execution_status'])
            rt.log_event('submitted', episode_id=row['episode_id'], snapshot_id=snap.snapshot_id, game_loop=loop,
                         reason='distill_switch', selected_candidate=card['candidate_id'], final_actions=entry.actions)
        self.pending = m._pending_commit
        self.decision = dict(step=step, game_loop=loop, snapshot_id=snap.snapshot_id, mode=mode, filtered=filtered,
            policy_version=self.version, selector_sha256=self.contract['selector_sha256'], selection_reason=reason,
            selector_audit=result.get('audit', {}),
            fallback_reason=('all_alternates_filtered' if filtered else 'singleton') if mode == 'fallback' else None,
            scores={c['candidate_id']: s for (c, _), s in zip(menu.values(), scores)}, top_ties=result['top_ties'],
            parent_actions=base['actions'], final_actions=action_map(actions), teacher_choice=row['teacher_choice'],
            selected_candidate=row['selected_candidate'], plan_ids=[p['id'] for p in summary['chosen_plans']],
            offered=entry is not None, committed=False)
        if 'trial_id' in self.contract:
            self.decision.update(trial_id=self.contract['trial_id'], opening_sha256=self.contract['opening_sha256'])
        if self.routed:
            self.decision.update(parent_plan=parent, alt_plan=other, route_anchor=anchor, route_choice=self.route_choice)
        if self.version == 'v3':
            reference = v if generated else v2_vec
            self.decision.update(v2_reference_actions=action_map(reference), v3_triggered=audit['v3_triggered'],
                                 v3_actions_changed=v != reference)
        r['decisions'].append(self.decision)
        return actions, summary

    @staticmethod
    def micro_clock(sub):
        # MicroAttackWindow records the last legal interior loop (inclusive).
        return sub['issued_loop'] < sub['submitted_loop'] <= sub['expires_loop'] == sub['issued_loop'] + 7

    def discard_for_swarm(self, shadow):
        """Separate an anchor proposal from the prefix of executed v2 decisions."""
        d = self.decision
        require(self.receipt['decisions'][-1] is d and not d['committed'] and
                self.receipt['steps'] == d['step'], 'cannot discard a committed or stale decision')
        key = (shadow.run_id, str(self.receipt['episode']), d['snapshot_id'], d['game_loop'])
        proposal = shadow._buffer.pop(key)
        require(proposal['final_actions'] == d['final_actions'], 'discarded wrong shadow proposal')
        shadow.stats['decisions_seen'] -= 1
        return self.receipt['decisions'].pop()

    def finish_swarm_prefix(self, steps):
        self.receipt['scope'] = 'executed_v2_prefix_only'
        if steps:
            self.finish(steps)
        else:
            require(not self.receipt['steps'] and not self.receipt['decisions'], 'nonempty prefix')
            self.receipt.update(status='complete', stats=dict.fromkeys(self.totals, 0))
            self.save()

    def finish(self, steps):
        r = self.receipt
        counts = Counter(d['mode'] for d in r['decisions'])
        stats = dict(steps=steps, model_base=counts['model_base'], model_switch=counts['model_switch'], fallback=counts['fallback'],
            engine_unconfirmed=r['engine_unconfirmed'], filtered_candidates=sum(len(d['filtered']) for d in r['decisions']))
        require(steps == sum(counts.values()), 'incomplete stats')
        r['stats'] = stats
        super().finish(steps)
        for k in stats:
            self.totals[k] += stats[k]
