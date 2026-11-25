"""One predeclared, matched-observation decision on the ordinary Shadow runner."""
import copy
import importlib
import json
from pathlib import Path
import re

from .config import WORKSPACE, digest


def require(ok, message):
    if not ok:
        raise ValueError(f'Zerg fork: {message}')


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def one(items, name):
    items = list(items)
    require(len(items) == 1, f'expected exactly one {name}, found {len(items)}')
    return items[0]


def action_map(actions):
    return {f'u{i}': a for i, a in enumerate(actions)}


def install_pending(manager, base, summary, telemetry, plans, actions):
    """Install a rendered native branch; the runner remains the only committer."""
    snap, _, _, risk, ctx = manager._pending_commit
    mod = importlib.import_module(type(manager).__module__)
    manager.telemetry = telemetry
    manager.stats['baseline_units'] += sum((v['skill'] == 'baseline') - (base['units'][k]['skill'] == 'baseline')
                                           for k, v in summary['units'].items() if snap.allies[k].alive)
    ctx.micro_approaches = mod.extra_approaches(ctx, summary, plans, actions) if manager.micro_all_attackers else []
    manager.last_micro_intents = []
    manager._cache = (manager._cache[0], list(actions), copy.deepcopy(summary), plans, snap)
    manager._pending_commit = (snap, plans, list(actions), risk, ctx)


def check_candidate(row, arm):
    c = one((c for c in row['candidates'] if c['candidate_id'] == arm['candidate_id']), 'candidate')
    require(c['candidate_family'] == 'joint_plan' and c['legal'] is True
            and c['source_plan_id'] == arm['source_plan_id'] and c['proposed_actions'] == arm['actions'],
            'candidate not the declared legal joint plan')


class Fork:
    def __init__(self, receipt_path):
        self.path = receipt_path
        self.receipt = dict(kind='zerg_one_shot_fork_receipt_v1', prefix_verified=False, anchor_commits=0,
                            final_actions=None, engine_receipt='unknown', error=None)
        self.checked_engine = False
        self.next_step = 0
        self.save(status='pending')

    def save(self, **fields):
        self.receipt.update(fields)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        tmp.replace(self.path)

    def load(self, spec_path, manifest):
        spec = json.loads(Path(spec_path).read_text(encoding='utf-8'))
        p, arm = spec['pair'], spec['arm']
        self.pair, self.arm = p, arm
        require(arm in ('A', 'B', 'C'), 'unknown arm')
        self.receipt.update({k: p[k] for k in ('episode_group', 'seed', 'roster', 'snapshot_id', 'game_loop')},
                            arm=arm, source_hash=spec['source_hash'], collection_context=spec['collection_context'],
                            candidate_id=p[arm]['candidate_id'], source_plan_id=p[arm]['source_plan_id'])
        if arm == 'C':
            h = spec.get('rule_hash')
            require(isinstance(h, str) and re.fullmatch('[0-9a-f]{64}', h), 'invalid C rule_hash')
            self.receipt['rule_hash'] = h
        source = Path(p['manifest'])
        require(json.loads(source.read_text()) == manifest and len(manifest['episodes']) == 1
                and p['episode_id'] == '1' and p['seed'] == manifest['episodes'][0]['sc2_seed']
                and p['episode_group'] == f"{p['run_id']}:1", 'wrong source scenario/episode')
        self.anchor = p['snapshot_id'] - 1
        require(type(p['snapshot_id']) is int and self.anchor >= 0, 'invalid snapshot')
        worker = source.parent
        self.original_run = json.loads((worker / 'runner/run.json').read_text())
        self.original = r = one((r for r in rows(worker / 'shadow/shadow.jsonl')
                                 if r['snapshot_id'] == p['snapshot_id'] and r['episode_id'] == p['episode_id']),
                                'archived anchor')
        self.check_shadow(r, p['run_id'])
        base, alt = p['A'], p['B']
        for candidate in (base, alt):
            a = candidate['actions']
            require(set(a) == {f'u{i}' for i in range(len(a))} and all(type(v) is int for v in a.values()),
                    'invalid per-unit actions')
            check_candidate(r, candidate)
        require(base['actions'].keys() == alt['actions'].keys() and base['actions'] != alt['actions'],
                'arms not distinct complete alternatives')
        if arm == 'C':
            c = p['C']['actions']
            require(isinstance(c, dict) and c.keys() == base['actions'].keys() and
                    all(type(a) is int for a in c.values()) and c not in (base['actions'], alt['actions']),
                    'invalid or non-novel declared C actions')
        require(p['changed'] == {u: dict(unit=r['global_features']['unit_types'].get(u), A=a, B=alt['actions'][u])
                                 for u, a in base['actions'].items() if alt['actions'][u] != a}, 'changed actors differ')

        self.observations, self.actions = {}, {}
        starts, summaries = [], []
        for event in rows(worker / 'runner/events.jsonl'):
            if event.get('episode') != 1:
                continue
            kind, step = event['event'], event.get('step')
            if kind == 'episode_started':
                starts.append(event)
            elif kind == 'skill_step' and step == self.anchor:
                summaries.append(event)
            elif kind == 'skill_observation' and step <= self.anchor:
                require(step not in self.observations, 'duplicate original observation')
                self.observations[step] = digest(event['payload'])
            elif kind == 'step' and step <= self.anchor + 1:
                require(step - 1 not in self.actions, 'duplicate original action')
                self.actions[step - 1] = (event['game_loop'], event['actions'])
        self.started = one(starts, 'original episode start')
        self.summary = one(summaries, 'original anchor skill_step')
        require(set(self.observations) == set(self.actions) == set(range(self.anchor + 1))
                and self.summary['snapshot_id'] == p['snapshot_id'] and self.summary['game_loop'] == p['game_loop']
                and self.summary['actions'] == action_map(self.actions[self.anchor][1]) == base['actions'],
                'inconsistent archived prefix')

    def check_shadow(self, r, run_id):
        p = self.pair
        require((r['run_id'], r['episode_id'], r['seed'], r['scene_config_id'], r['snapshot_id'], r['game_loop'],
                 r['source_hash'], r['execution_context_hash']) ==
                (run_id, p['episode_id'], p['seed'], p['roster'], p['snapshot_id'], p['game_loop'],
                 self.receipt['source_hash'], self.receipt['collection_context']), 'shadow identity/context mismatch')
        require(r['teacher_choice'] == r['selected_candidate'] == p['A']['candidate_id']
                and r['final_actions'] == p['A']['actions'], 'BASE was not the rule choice')

    def check_context(self, metadata, runtime):
        require(runtime is not None, 'requires ZERG_SHADOW_DIR and ZERG_SHADOW_ROSTER')
        old = self.original_run
        for key in ('method', 'skill_version', 'execution_mode', 'macro_loops', 'engine_step_loops', 'skill_parameters',
                    'manifest_sha256', 'env_args', 'map_file', 'damage_by_type_enabled', 'skill_source_hashes', 'versions'):
            require(old[key] == metadata[key], f'run context changed: {key}')
        map_key = one((k for k in metadata['source_hashes'] if k.endswith('.SC2Map') and
                       (WORKSPACE / k).resolve() == Path(metadata['map_file'])), 'map content hash')
        require(metadata['source_hashes'][map_key] == old['source_hashes'][map_key], 'map content changed')
        for meta in (old, metadata):
            shadow = meta['zerg_shadow']
            require((shadow['source_fingerprint'], shadow['execution_context_hash'], shadow['scene_config_id']) ==
                    (self.receipt['source_hash'], self.receipt['collection_context'], self.pair['roster']),
                    'source/context/roster differs')
        require(old['zerg_shadow']['run_id'] == self.pair['run_id'], 'archive run_id mismatch')

    def start(self, scenario, initial_digest, version, loop):
        e = self.started
        self.receipt.update(scenario_sha256=scenario['sha256'], initial_state_sha256=initial_digest)
        require((scenario['sha256'], initial_digest, version, loop) ==
                (e['scenario_sha256'], e['initial_state_sha256'], e['game_version'], e['game_loop']),
                'initial state/scenario/game build mismatch')

    def step(self, step, loop, controller, payload, env):
        if step > self.anchor:
            self.check_reached()
            return controller.step(payload, env.n_agents)
        require(step == self.next_step and digest(payload) == self.observations[step]
                and loop == self.actions[step][0], f'observed prefix mismatch at step {step}')
        self.next_step += 1
        manager = controller.manager
        render = None
        if step == self.anchor and self.arm in ('B', 'C'):
            # Retain the native summary inputs before BASE updates the usage counters.
            # Re-render the selected complete entry without rerunning BASE search.
            coord = manager.coordinator
            native = coord._summary
            def capture(final, ctx, telemetry, *args):
                nonlocal render
                branch_telemetry = copy.deepcopy(telemetry)
                render = lambda entry: (native(entry, ctx, branch_telemetry, *args), branch_telemetry)
                return native(final, ctx, telemetry, *args)
            require('_summary' not in vars(coord), 'unexpected coordinator override')
            coord._summary = capture
            try:
                actions, summary = controller.step(payload, env.n_agents)
            finally:
                del coord._summary
        else:
            actions, summary = controller.step(payload, env.n_agents)
        require(actions == self.actions[step][1], f'rule actions differ at step {step}')
        if step != self.anchor:
            return actions, summary

        runtime = controller.runtime
        p, arm = self.pair, self.pair[self.arm]
        pending, cache = manager._pending_commit, manager._cache
        require(pending and cache and pending[0] is cache[4] is pending[4].snapshot
                and pending[0].snapshot_id == summary['snapshot_id'] == p['snapshot_id']
                and pending[0].game_loop == summary['snapshot_game_loop'] == p['game_loop']
                and not manager.memory.committed, 'no current uncommitted snapshot')
        record = runtime._buffer[(runtime.run_id, p['episode_id'], p['snapshot_id'], p['game_loop'])]
        self.check_shadow(record, runtime.run_id)
        # Shadow persists nonfinite/missing feature values as null.
        from decision_tree.zerg.schema import dumps
        require(dumps(record['global_features']) == dumps(self.original['global_features']), 'anchor snapshot features differ')
        for candidate in (p['A'], p['B'] if self.arm == 'C' else arm):
            check_candidate(record, candidate)
        require(summary['chosen_plans'] == self.summary['chosen_plans'] and
                [plan.proposal_id for plan in pending[1]] == [s['id'] for s in summary['chosen_plans']] and
                all(plan.snapshot_id == p['snapshot_id'] for plan in pending[1]), 'rule plan identity changed')
        require(set(arm['actions']) == set(action_map(actions)) and len(actions) == env.n_agents,
                'arm does not cover the whole team')
        for i in range(env.n_agents):
            action = arm['actions'][f'u{i}']
            mask = env.get_avail_agent_actions(i)
            require(0 <= action < len(mask) and mask[action], f'illegal live action for u{i}')
        if self.arm == 'B':
            actions, summary = self.replace(controller, summary, record, render)
        elif self.arm == 'C':
            entry, card = self._propose_c(controller, summary, payload, env, self.receipt['rule_hash'])
            require(entry is not None, 'C rule abstained')
            require(all(arm[k] == v for k, v in card.items()), 'live C differs from frozen declaration')
            require(render is not None, 'native C render unavailable')
            actions, summary = self.install(controller, summary, record, render, entry)
        self.receipt['prefix_verified'] = True
        return actions, summary

    def peek_c(self, controller, base_summary, payload, env, rule_hash):
        """On a disposable A replay, call after the verified anchor step, before commit.

        Returns the frozen pair['C'] fields or None for abstention. This does not
        install/commit C or change candidate membership in the replay record.
        """
        require(self.arm == 'A' and self.receipt['prefix_verified'] and not self.receipt['anchor_commits'],
                'C peek requires a verified uncommitted A anchor')
        return self._propose_c(controller, base_summary, payload, env, rule_hash)[1]

    def _propose_c(self, controller, base, payload, env, expected_hash):
        from decision_tree.zerg import rules_v1 as rule

        require(isinstance(expected_hash, str) and re.fullmatch('[0-9a-f]{64}', expected_hash) and
                rule.rule_hash() == expected_hash, 'C rule_hash mismatch')
        rule_id = rule.RULE_ID
        manager, rt = controller.manager, controller.runtime
        pending, cache = manager._pending_commit, manager.coordinator._cache
        require(pending and cache and not manager.memory.committed and
                pending[0] is manager._cache[4] is pending[4].snapshot and
                (pending[0].snapshot_id, pending[0].game_loop) == (self.pair['snapshot_id'], self.pair['game_loop']) and
                base['actions'] == action_map(pending[2]) == self.pair['A']['actions'] and
                digest(payload) == self.observations[self.anchor], 'C needs the verified live BASE')
        record = rt._buffer[(rt.run_id, self.pair['episode_id'], pending[0].snapshot_id, pending[0].game_loop)]
        # Freeze the original search space BEFORE calling the extension. The
        # summary is top-k and cannot by itself establish action novelty.
        def cache_state():
            return {k: (id(e), tuple(sorted(e.actions.items()))) for k, e in cache.items()}
        before = cache_state()
        existing = {v for _, v in before.values()}
        for summary in (base, self.summary):
            existing.add(tuple(sorted(summary['actions'].items())))
            existing.update(tuple(sorted(a['actions'].items())) for a in summary['top_alternates'])
        for row in (self.original, record):
            for c in row['candidates']:
                require(rule_id not in (c['candidate_id'], c['source_plan_id']), 'C id already in original candidates')
                existing.add(tuple(sorted(c['proposed_actions'].items())))
        require(not base['local_coord']['selected'], 'branch-private local arbitration context unavailable')
        entry = rule.make_entry(manager, base, payload)
        require(manager.coordinator._cache is cache and before == cache_state() and
                manager._pending_commit is pending and not manager.memory.committed,
                'C rule changed original cache or pending commit')
        if entry is None:
            return None, None
        require(type(entry) is type(next(iter(cache.values()))), 'C lacks a native complete entry')
        actions, plans = entry.actions, entry.plans
        require(isinstance(actions, dict) and set(actions) == {f'u{i}' for i in range(env.n_agents)} and
                all(type(v) is int for v in actions.values()), 'C must cover the whole team with integer actions')
        require(tuple(sorted(actions.items())) not in existing, 'C actions already in original candidates/cache')
        for i in range(env.n_agents):
            mask, a = env.get_avail_agent_actions(i), actions[f'u{i}']
            require(0 <= a < len(mask) and mask[a], f'illegal live C action for u{i}')
        coord_module = importlib.import_module(type(manager.coordinator).__module__)
        require(plans and all(isinstance(p, coord_module.CandidatePlan) for p in plans) and
                len({p.proposal_id for p in plans}) == len(plans) and
                all(p.snapshot_id == pending[0].snapshot_id and p.controlled_actors and
                    p.primary_skill not in ('local_engagement', 'reserve_dispatch') and
                    set(p.current_actions) == set(p.controlled_actors) and
                    all(actions.get(a) == cmd for a, cmd in p.current_actions.items()) for p in plans),
                'C has empty, stale, local or mismatched plans')
        ledger = coord_module.Ledger()
        for p in plans:
            require(not ledger.conflicts(p), 'C plans have conflicting resource claims')
            ledger.reserve(p)
        self.validate_entry(manager, entry)
        return entry, dict(candidate_id=rule_id, source_plan_id=rule_id, actions=dict(actions))

    def replace(self, controller, base, record, render):
        manager = controller.manager
        arm = self.pair['B']
        def alternate(summary):
            return one((a for a in summary['top_alternates'] if a['id'] == arm['source_plan_id']), 'ranked alternative')
        alt = alternate(base)
        require(alt == alternate(self.summary) and alt['actions'] == arm['actions'], 'live alternative differs')
        require(alt['plans'], 'alternative lacks real plan objects')
        require(not base['local_coord']['selected'] and all(p['skill'] not in ('local_engagement', 'reserve_dispatch')
                for p in alt['plans']), 'branch-private local arbitration context unavailable')
        # Same action vector can belong to several different cached plan sets.
        # Use the same ordering as the coordinator's top_alternates, then check
        # the entire ordered plan descriptor, not a synthetic joint_alt_N id.
        coord = manager.coordinator
        matches = [e for e in coord._cache.values() if e.actions == alt['actions']]
        require(matches and render is not None, 'complete live branch unavailable')
        entry = min(matches, key=lambda e: coord._rank(e, manager.memory))
        plans = list(entry.plans)
        snap = manager._pending_commit[0]
        require([dict(id=p.proposal_id, skill=p.primary_skill, actors=list(p.controlled_actors), goal=p.goal)
                 for p in plans] == alt['plans'] and all(p.snapshot_id == snap.snapshot_id and
                     all(entry.actions[a] == p.current_actions[a] for a in p.controlled_actors) for p in plans),
                'cached plan objects do not match the current branch')
        self.validate_entry(manager, entry)
        return self.install(controller, base, record, render, entry)

    @staticmethod
    def validate_entry(manager, entry):
        ctx = manager._pending_commit[-1]
        mod = importlib.import_module(type(manager).__module__)
        validator = importlib.import_module(f'{mod.__package__}.validator')
        problems = validator.complete_and_validate(list(entry.plans), ctx, entry.actions,
                    report=manager.evaluator.effects(ctx, entry.actions), memory=manager.memory)
        require(not problems, f'complete branch validation failed: {problems}')

    def install(self, controller, base, record, render, entry):
        manager, runtime = controller.manager, controller.runtime
        mod = importlib.import_module(type(manager).__module__)
        snap, _, _, risk, ctx = manager._pending_commit
        arm, plans = self.pair[self.arm], list(entry.plans)
        summary, telemetry = render(entry)
        actions = [entry.actions[f'u{i}'] for i in range(len(entry.actions))]
        require(summary['actions'] == arm['actions'], 'native branch render mismatch')
        for key in ('battle_intent', 'splash_model', 'local_coord'):
            if key in base:
                summary[key] = base[key]
        summary['plan_evaluations_used'] += summary['local_coord']['local_evaluations']
        summary['zerg_fork'] = dict(arm=self.arm, candidate_id=arm['candidate_id'], rule_actions=base['actions'])
        if self.arm == 'C':
            from decision_tree.zerg.candidates import joint_candidates_from_summary
            require([p.proposal_id for p in plans] == [p['id'] for p in summary['chosen_plans']] and
                    {u: row['action'] for u, row in summary['units'].items()} == entry.actions,
                    'native C plan/unit render mismatch')
            h = self.receipt['rule_hash']
            summary['zerg_fork']['rule_hash'] = h
            # Only the replay gets this new candidate, never the archived menu
            # or coordinator search cache. Keep the original teacher choice.
            c, = joint_candidates_from_summary({**summary, 'top_alternates': []}, snap, ctx)
            record['candidates'].append({**c.to_record(), 'candidate_id': arm['candidate_id'],
                                         'source_plan_id': arm['source_plan_id'],
                                         'origin': 'one_shot_fork_C', 'rule_hash': h})
        install_pending(manager, base, summary, telemetry, plans, actions)
        record.update(selected_candidate=arm['candidate_id'], final_actions=dict(entry.actions))
        if record.get('skill_proposals'):
            record['skill_proposals'] = runtime._enriched_skill_proposals(
                snap, controller.last_meta, record['final_actions'], record['execution_status'])
        controller.last_meta['selected_candidate'] = arm['candidate_id']
        runtime.log_event('submitted', episode_id=record['episode_id'], snapshot_id=snap.snapshot_id,
                          game_loop=snap.game_loop, reason='one_shot_fork_supersedes_rule_proposal',
                          selected_candidate=arm['candidate_id'], final_actions=entry.actions)
        self.receipt['validated_plan_ids'] = [p.proposal_id for p in plans]
        return actions, summary

    def committed(self, manager, actions):
        final = action_map(actions)
        require(self.receipt['anchor_commits'] == 0 and self.receipt['prefix_verified'] and
                manager._pending_commit is None and manager.memory.committed and
                manager.memory.last_committed_loop == self.pair['game_loop'] and
                manager.memory.last_committed_actions == final == self.pair[self.arm]['actions'],
                'anchor memory commit not confirmed')
        self.save(anchor_commits=1, final_actions=final)

    def check_engine(self, feedback):
        actors = feedback.get('actors') or {}
        mapped = feedback.get('response_mapping') == 'exact'
        final = self.receipt['final_actions']
        start = feedback.get('from_game_loop')
        require(start is None or start == self.pair['game_loop'], 'stale macro receipt')
        if any(code != 1 for code in feedback.get('raw_response_results', [])):
            self.receipt['engine_receipt'] = 'rejected'
            raise ValueError('Zerg fork: engine rejected a command in the macro batch')
        known = mapped and start is not None and set(actors) == set(final)
        for actor, action in final.items():
            row = actors.get(actor, {})
            requested = row.get('requested_action_id')
            require(requested is None or str(requested) == str(action), 'engine requested action differs')
            code = (row.get('response') or {}).get('code')
            if row.get('submission') == 'rejected' or code not in (None, 1):
                self.receipt['engine_receipt'] = 'rejected'
                raise ValueError(f'Zerg fork: engine rejected {actor}')
            if requested is None or not ((row.get('submission') == 'dead_noop' and action == 0
                                         and row.get('command') is None and code is None) or
                                        (row.get('submission') == 'accepted' and row.get('command') and code == 1)):
                known = False
        self.checked_engine = True
        self.save(engine_receipt='accepted' if known and any(final.values()) else 'unknown')

    def check_reached(self):
        require(self.receipt['anchor_commits'] == 1 and self.checked_engine, 'anchor never completed')

    def finish(self):
        self.check_reached()
        self.save(status='complete')
