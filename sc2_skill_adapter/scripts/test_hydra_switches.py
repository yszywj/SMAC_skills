"""Offline checks for the hydra switch experiment (no SC2, no engine).

1. Patch a throwaway copy of the frozen tactical_v1 + bridge runner and
   compile the result; verify every injection anchor took effect.
2. Import the patched package and exercise the switch logic on synthetic
   battlefields covering each A/B criterion, veto, exemption, the A+B
   interplay and the flags-off default.
3. Load a second copy with both switches OFF and verify apply() is inert.

Run:  python scripts/test_hydra_switches.py
"""
import copy
import importlib
import json
import py_compile
import shutil
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ADAPTER = HERE.parent
sys.path.insert(0, str(ADAPTER))

from hydra_control_switches import apply_variant  # noqa: E402

SOURCE = (ADAPTER.parent / 'runs/contact_response_micro_20261121_153552_909880'
          / 'source_snapshot')
sys.path.insert(0, str(SOURCE))
FAILURES = []


def check(name, condition, detail=''):
    status = 'PASS' if condition else 'FAIL'
    if not condition:
        FAILURES.append(f'{name}: {detail}')
    print(f'[{status}] {name}' + (f' — {detail}' if detail and not condition else ''))


def load_package(tactical_dir, name):
    package = types.ModuleType(name)
    package.__path__ = [str(tactical_dir)]
    sys.modules[name] = package
    return package


def make_world(pkg, allies, enemies, actions, movement, legal, window=7):
    """allies/enemies: dicts of views built by the caller; actions maps actor
    to the current final action id."""
    contracts = importlib.import_module(pkg.__name__ + '.contracts')
    relations = {}
    for actor, unit in allies.items():
        row = {}
        for enemy_id, enemy in enemies.items():
            dist = ((unit.position[0] - enemy.position[0]) ** 2 +
                    (unit.position[1] - enemy.position[1]) ** 2) ** 0.5
            row[enemy_id] = contracts.Relation(
                distance=dist, in_weapon_range=dist <= 5.0,
                weapon_range_status='in_range' if dist <= 5.0 else 'out_of_range',
                native_weapon_range=5.0, in_enemy_weapon_range=dist <= 2.0,
                enemy_weapon_range_status='in_range' if dist <= 2.0 else 'out_of_range',
                smac_attack_available=dist <= 5.0)
        relations[actor] = row
    snapshot = contracts.Snapshot(
        snapshot_id=window, game_loop=8 * (window - 1), step_mul=8,
        steps_remaining=100, map_name='10gen_zerg', episode=1,
        search_location=(16.0, 16.0), allies=allies, enemies=enemies,
        relations=relations, movement_candidates=movement, heal_candidates={},
        legal={a: frozenset(legal[a]) for a in legal}, keep_heal_order={},
        previous_action_ids={}, raw_state={
            'allies': {a: {'tag': f'{a}-tag', 'orders': []} for a in allies},
            'visible_enemies': {e: {'tag': f'{e}-tag', 'alive': True}
                                for e in enemies}},
        n_allies=len(allies), n_enemies=max(e.index for e in enemies.values()) + 1, keep_heal_action_id=None)
    context = importlib.import_module(pkg.__name__ + '.context').Context(snapshot)
    return context


class Estimate:
    def __init__(self, shots=1, damage=12.0):
        self.window_shots = shots
        self.damage_est = damage
        self.damage_lo = damage


THREAT_R = 2.0
ATTACK_R = 5.0


def install_stubs(context, killable=(), threat_r=THREAT_R):
    """Deterministic geometry: threats within threat_r, attacks within 5."""
    contracts = importlib.import_module(_pkg_name(context) + '.context')

    def risk_at(actor, position, horizon=None):
        threats = tuple(e for e, enemy in context.snapshot.enemies.items()
                        if ((position[0] - enemy.position[0]) ** 2 +
                            (position[1] - enemy.position[1]) ** 2) ** 0.5 < threat_r)
        return contracts.RiskReport(threats=threats, heuristic=False,
                                    effect_weapon_near=())

    def fire_estimate(actor, enemy_id, *args, **kwargs):
        if enemy_id in killable:
            return Estimate(damage=999.0)
        return Estimate()

    def reach(actor, enemy_id):
        return 5.0

    def contact_status(actor, enemy_id):
        relation = context.snapshot.relations[actor][enemy_id]
        return 'in_range' if relation.distance <= ATTACK_R else 'out_of_range'

    def attackable(actor):
        return sorted((e for e in context.snapshot.enemies
                       if context.snapshot.relations[actor][e].distance
                       <= ATTACK_R and context.attack_action(actor, e) is not None),
                      key=lambda e: (context.snapshot.relations[actor][e].distance, e))

    context.risk_at = risk_at
    context.fire_estimate = fire_estimate
    context.effective_attack_reach = reach
    context.contact_status = contact_status
    context.attackable_enemies = attackable
    context.speed = lambda actor: 2.25


def _pkg_name(context):
    return type(context).__module__.rsplit('.', 1)[0]


def ally_view(pkg, actor, index, position, hp=24.0, max_hp=80.0, lost=10.0,
              cooldown=5.0, ready=False):
    contracts = importlib.import_module(pkg.__name__ + '.contracts')
    return contracts.UnitView(
        actor=actor, index=index, tag=f'{actor}-tag', type_name='hydralisk',
        unit_type_id=107, position=position, radius=0.5, is_flying=False,
        alive=True, health=hp, max_health=max_hp, shield=0.0, max_shield=0.0,
        weapon_cooldown_loops=cooldown, weapon_ready=ready, energy=None,
        health_lost=lost, shield_lost=0.0)


def enemy_view(pkg, enemy_id, index, position, hp=100.0, kind='zergling'):
    contracts = importlib.import_module(pkg.__name__ + '.contracts')
    return contracts.EnemyView(
        enemy_id=enemy_id, index=index, type_name=kind, unit_type_id=105,
        position=position, radius=0.35, is_flying=False, health=hp,
        max_health=hp, shield=0.0, max_shield=0.0)


def plan_view(pkg, actor, skill, phase, rationale, goal=None, goal_kind='internal',
              goal_data=None):
    contracts = importlib.import_module(pkg.__name__ + '.contracts')
    return contracts.CandidatePlan(
        proposal_id=f'{actor}-plan', snapshot_id=1, config_fingerprint='x',
        primary_skill=skill, specialization_provider=None, instance_id=None,
        controlled_actors=(actor,), beneficiaries=(actor,), goal=goal,
        current_actions={actor: 5 if phase == 'ESCAPE' else 6}, expected_effects=[], claims=None, dependencies=(),
        commitment=contracts.Commitment(phase), features={},
        proposed_memory_delta={}, rationale=rationale, goal_kind=goal_kind,
        goal_data=goal_data or {})


class FakeMemory:
    def battle_target(self, context, actor):
        return None


class FakeManager:
    def __init__(self):
        self.memory = FakeMemory()
        self.evaluator = types.SimpleNamespace(effects=lambda ctx, joint: None)
        self._hydra_switch_state = None


def base_actions(n=10):
    return [1] * n


def moves_at(position):
    x, y = position
    return {2: (x, y + 1), 3: (x, y - 1), 4: (x + 1, y), 5: (x - 1, y)}


def scenario_setup(pkg, allies_spec, enemies_spec, legal_by_actor, window=2):
    allies, enemies = {}, {}
    for actor, index, position, kw in allies_spec:
        allies[actor] = ally_view(pkg, actor, index, position, **kw)
    for enemy_id, index, position, kw in enemies_spec:
        enemies[enemy_id] = enemy_view(pkg, enemy_id, index, position, **kw)
    movement, legal = {}, {}
    for actor, (position,) in ((a, (allies[a].position,)) for a in allies):
        movement[actor] = {a: {'destination': list(dest)}
                           for a, dest in moves_at(position).items()}
        legal[actor] = sorted(set([1, *movement[actor]] + [
            6 + e.index for e in enemies.values()]))
    from dataclasses import replace
    allies['u0'] = replace(ally_view(pkg, 'u0', 0, (0., 0.)), alive=False, health=0.)
    legal['u0'], movement['u0'] = [0], {}
    actions = base_actions(len(allies))
    for actor in allies:
        actions[allies[actor].index] = legal_by_actor.get(actor, 0)
    return allies, enemies, movement, legal, actions


def rows_for(summary):
    return summary.get('hydra_switch_decisions', [])


def replay_baseline(pkg):
    """Exact action/authorization comparison, including frozen memory commits."""
    manager = importlib.import_module(pkg.__name__ + '.skill_manager').SkillManager()
    shard = SOURCE.parent / 'micro8-hydra/zerg_3zergling_0baneling_7hydralisk_00'
    windows = 0
    with (shard / 'events.jsonl').open() as stream:
        for line in stream:
            event = json.loads(line)
            kind = event['event']
            if kind == 'skill_observation':
                actions, summary = manager.step(event['payload'], 10)
                manager.commit_final_actions(actions)
            elif kind == 'micro_window_closed':
                keys = ('actor', 'actor_tag', 'target', 'target_tag', 'macro_action_id')
                project = lambda rows: sorted(tuple(r[k] for k in keys) for r in rows)
                assert project(manager.last_micro_intents) == project(event['records'])
            elif kind == 'step':
                assert actions == event['actions']
                assert summary['actions'] == {f'u{i}': n for i, n in enumerate(actions)}
                for r in event['feedback']['micro_submissions']:
                    if r.get('sc2_response') == [1]:
                        manager.record_micro_action(r['actor'], 6 + int(r['target'][1:]))
                windows += 1
                if windows == 8:
                    break
    check('baseline replays 80 historical actions and original authorizations', windows == 8)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / 'work'
        (work / 'skills').mkdir(parents=True)
        shutil.copytree(SOURCE / 'skills' / 'tactical_v1',
                        work / 'skills' / 'tactical_v1',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copytree(SOURCE / 'sc2_skill_adapter' / 'bridge',
                        work / 'sc2_skill_adapter' / 'bridge',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        flags = apply_variant(work, escape=True, shot_guard=True)
        check('patch flags recorded', flags == {'isolated_escape': True,
                                                'protect_next_shot': True,
                                                'hydra_switches_module': str(
                                                    work / 'skills/tactical_v1/hydra_switches.py')})
        tactical = work / 'skills' / 'tactical_v1'
        for name in ('hydra_switches.py', 'skill_manager.py'):
            py_compile.compile(str(tactical / name), doraise=True)
        py_compile.compile(str(work / 'sc2_skill_adapter' / 'bridge' / 'runner.py'),
                           doraise=True)
        check('patched files compile', True)
        manager_text = (tactical / 'skill_manager.py').read_text(encoding='utf-8')
        runner_text = (work / 'sc2_skill_adapter' / 'bridge' / 'runner.py').read_text(
            encoding='utf-8')
        check('skill_manager hook injected',
              '_hydra_switches.apply(self, context, plans, summary, actions)'
              in manager_text)
        for anchor in ('switch_rows = []', "telemetry.get('hydra_switch_decisions')",
                       "record['hydra_switches']"):
            check(f'runner hook present: {anchor[:28]}', anchor in runner_text)
        flag_text = (tactical / 'hydra_switches.json').read_text(encoding='utf-8')
        check('flags file written', '"isolated_escape": true' in flag_text)

        pkg = load_package(str(tactical), '_hydra_switch_test_on')
        switches = importlib.import_module(pkg.__name__ + '.hydra_switches')
        check('switches enabled in on-copy', switches.ENABLED is True)

        # ---- A: isolated + pincer + bleeding -> applied escape move ----
        allies_spec = [('u1', 1, (10.0, 10.0), {}),
                       ('u2', 2, (25.0, 25.0), {'lost': 0.0, 'cooldown': 0.0})]
        enemies_spec = [('e0', 0, (8.8, 10.0), {'kind': 'zergling'}),
                        ('e1', 1, (11.8, 10.0), {'kind': 'hydralisk'})]
        allies, enemies, movement, legal, actions = scenario_setup(
            pkg, allies_spec, enemies_spec, {'u1': 6, 'u2': 1}, window=2)
        context = make_world(pkg, allies, enemies, actions, movement, legal, window=2)
        install_stubs(context)
        manager = FakeManager()
        summary = {}
        plans = [plan_view(pkg, 'u1', 'focus_fire', 'attacking', 'volley')]
        out = switches.apply(manager, context, plans, summary, actions)
        rows = rows_for(summary)
        row_a = next(r for r in rows if r['actor'] == 'u1' and r['switch'] == 'A')
        check('A pincer escape applied', row_a['result'] == 'applied'
              and row_a['changed'] and out[allies['u1'].index] == row_a['candidate_action'],
              f"result={row_a['result']} final={row_a['final_action']}")
        check('A kept a one-window permit only',
              sum(1 for r in rows if r['switch'] == 'A' and r.get('changed')) == 1)
        check('A override is local, not an empty forced target set',
              context.forced_hydra_targets('u1') == ('e1',) and
              context._hydra_escape_permit == (2, 'u1', out[1]))
        check('A publishes actual role/action and releases the old plan',
              summary['units']['u1']['skill'] == 'isolated_escape' and
              summary['actions']['u1'] == out[1] and plans == [])
        from unittest.mock import patch
        validator = importlib.import_module(pkg.__name__ + '.validator')
        trial = dict(enumerate(out))
        joint = {f'u{i}': cmd for i, cmd in trial.items()}
        with patch.object(validator, 'ready_hydralisk_shots', return_value=(7,)), \
                patch.object(validator, 'rescue_actions', return_value=(2, 3, 4, 5)):
            check('frozen validator permits only the exact A order',
                  not validator.validate_plans([], context, joint) and
                  validator.validate_plans([], context, {**joint, 'u1': 2 if out[1] != 2 else 3})[0]['type'] ==
                  'ready_shot_before_withdrawal')
        check('A row carries provenance and support',
              row_a['original_skill'] == 'focus_fire'
               and row_a['ally_support']['e0'] == [] and row_a['estimated'] is True)
        switches.confirm(manager, context, out)
        # follower row on the next window
        context.snapshot = copy.copy(context.snapshot)
        object.__setattr__(context.snapshot, 'snapshot_id', 3)
        object.__setattr__(context.snapshot, 'game_loop', 16)
        summary2 = {}
        switches.apply(manager, context, plans, summary2, out)
        follow = [r for r in rows_for(summary2) if r.get('followup')]
        check('A follow-up row logged', len(follow) == 1
              and follow[0]['alive'] and follow[0]['windows_since'] == 1,
              str(follow))

        # ---- A: ally fire support -> keep attacking ----
        allies_spec[1] = ('u2', 2, (8.5, 11.0), {'lost': 0.0, 'cooldown': 0.0})
        allies, enemies, movement, legal, actions = scenario_setup(
            pkg, allies_spec, enemies_spec, {'u1': 6, 'u2': 1}, window=2)
        context = make_world(pkg, allies, enemies, actions, movement, legal, window=2)
        install_stubs(context)
        summary = {}
        switches.apply(FakeManager(), context, plans, summary, actions)
        row = next(r for r in rows_for(summary) if r['actor'] == 'u1'
                   and r['switch'] == 'A')
        check('A veto: ally can return fire', row['result'].startswith('unchanged')
              and row['veto'] == 'ally_fire_support_on_a_threat', str(row['result']))

        # ---- A: immediate kill available -> keep attacking ----
        allies, enemies, movement, legal, actions = scenario_setup(
            pkg, [('u1', 1, (10.0, 10.0), {})],
            [('e0', 0, (9.5, 10.0), {'kind': 'zergling', 'hp': 8.0})],
            {'u1': 6}, window=2)
        context = make_world(pkg, allies, enemies, actions, movement, legal, window=2)
        install_stubs(context, killable=('e0',))
        summary = {}
        switches.apply(FakeManager(), context, [], summary, actions)
        row = next(r for r in rows_for(summary) if r['switch'] == 'A')
        check('A veto: immediate kill', row['veto'] == 'immediate_kill_available',
              str(row['result']))

        # ---- A: weapon ready and not critical -> prefer firing ----
        allies, enemies, movement, legal, actions = scenario_setup(
            pkg, [('u1', 1, (10.0, 10.0), {'cooldown': 0.0, 'ready': True, 'hp': 60.})],
            [('e0', 0, (8.8, 10.0), {'kind': 'zergling'}),
             ('e1', 1, (11.8, 10.0), {'kind': 'hydralisk'})],
            {'u1': 6}, window=2)
        context = make_world(pkg, allies, enemies, actions, movement, legal, window=2)
        install_stubs(context)
        summary = {}
        switches.apply(FakeManager(), context, [], summary, actions)
        row = next(r for r in rows_for(summary) if r['switch'] == 'A')
        check('A veto: ready and not critical',
              row['veto'] == 'weapon_ready_and_not_critical', str(row['result']))

        # ---- A: single-window loss only is not an unfavorable exchange ----
        allies, enemies, movement, legal, actions = scenario_setup(
            pkg, [('u1', 1, (10.0, 10.0), {})],
            [('e0', 0, (8.8, 10.0), {'kind': 'zergling'})],
            {'u1': 6}, window=2)
        context = make_world(pkg, allies, enemies, actions, movement, legal, window=2)
        install_stubs(context)
        summary = {}
        switches.apply(FakeManager(), context, [], summary, actions)
        row = next(r for r in rows_for(summary) if r['switch'] == 'A')
        check('A veto: single window loss', row['veto'] == 'single_window_loss_only',
              str(row['result']))

        # ---- A: every move keeps the pressure (no worthwhile move) ----
        allies, enemies, movement, legal, actions = scenario_setup(
            pkg, [('u1', 1, (10.0, 10.0), {})],
            [('e0', 0, (10.5, 10.5), {'kind': 'zergling'}),
             ('e1', 1, (9.5, 10.5), {'kind': 'hydralisk'}),
             ('e2', 2, (10.5, 9.5), {'kind': 'zergling'}),
             ('e3', 3, (9.5, 9.5), {'kind': 'hydralisk'})],
            {'u1': 6}, window=2)
        context = make_world(pkg, allies, enemies, actions, movement, legal, window=2)
        install_stubs(context)
        summary = {}
        switches.apply(FakeManager(), context, [], summary, actions)
        row = next(r for r in rows_for(summary) if r['switch'] == 'A')
        check('A veto: only moves add crossfire',
              row['veto'] == 'every_move_adds_threat_or_keeps_pressure',
              str(row['result']))

        # ---- B: soon-ready shot preserved over a range-breaking retreat ----
        allies, enemies, movement, legal, actions = scenario_setup(
            pkg, [('u1', 1, (10.0, 12.0), {'lost': 0.0, 'cooldown': 6.0})],
            [('e5', 5, (14.8, 12.0), {'kind': 'hydralisk'})],
            {'u1': 5}, window=2)
        context = make_world(pkg, allies, enemies, actions, movement, legal, window=2)
        install_stubs(context)
        manager = FakeManager()
        summary = {'units': {'u1': {'skill': 'disengage_reenter', 'phase': 'ESCAPE'}}}
        plans = [plan_view(pkg, 'u1', 'disengage_reenter', 'ESCAPE',
                           'escape: forecast improvement', goal='safe',
                           goal_data={'objective': {'kind': 'escape'},
                                      'target_id': 'e5'})]
        out = switches.apply(manager, context, plans, summary, actions)
        row = next(r for r in rows_for(summary) if r['switch'] == 'B')
        check('B intercepts a wasted retreat', row['result'] == 'applied'
              and row['changed'] and out[allies['u1'].index] == row['candidate_action']
              and row['current_target'] == 'e5', str(row['result']))
        # Only a matching runner commit can start a follow-up.
        switches.confirm(manager, context, [0, 5])
        check('unsubmitted B candidate is not tracked', not manager._hydra_switch_state['tracked'])
        for skill, phase in [('coordinated_engage', 'approaching'), ('cohesive_search', 'search'),
                             ('ranged_fire_formation', 'exclusive_control')]:
            context.hydra_move_roles = {('u1', 5): 'support'}
            audit = {'units': {'u1': dict(skill=skill, phase=phase, goal='e5', goal_kind='enemy')}}
            unchanged = switches.apply(FakeManager(), context, [], audit, [0, 5])
            check(f'B excludes {skill}/{phase} even with an attack in reach', unchanged == [0, 5])

        # Shared coverage can be broken even when the preserved target remains reachable.
        a, e, m, menu, cmds = scenario_setup(pkg,
            [('u1', 1, (10., 12.), {}), ('u2', 2, (18., 12.), {})],
            [('e5', 5, (12., 12.), {'kind': 'hydralisk'}), ('e6', 6, (14.8, 12.), {'kind': 'hydralisk'})],
            {'u1': 5, 'u2': 12})
        shared = make_world(pkg, a, e, cmds, m, menu)
        install_stubs(shared)
        shared.snapshot.raw_state['allies']['u1']['orders'] = [dict(kind='attack', target_tag='e5-tag')]
        audit = {'units': {'u1': dict(skill='disengage_reenter', phase='ESCAPE')}}
        switches.apply(FakeManager(), shared, [], audit, cmds)
        decision = next(r for r in rows_for(audit) if r['actor'] == 'u1' and r['switch'] == 'B')
        check('B protects shared lane but preserves observed target/tag', decision['changed'] and
              decision['keeps_range_estimate'] and decision['shared_targets_lost_estimate'] == ['e6'] and
              decision['target_tag'] == 'e5-tag' and cmds == [0, 11, 12])

        # ---- B: urgent rescue plans are exempt (original move restored) ----
        actions[allies['u1'].index] = 5
        plans = [plan_view(pkg, 'u1', 'disengage_reenter', 'ESCAPE',
                           'execute observed emergency relief', goal='safe',
                           goal_data={'objective': {'kind': 'escape',
                                                    'one_window_rescue': True,
                                                    'execution_reason': 'lethal_pressure_with_survivable_route'}})]
        summary = {'units': {'u1': {'skill': 'disengage_reenter', 'phase': 'ESCAPE'}}}
        switches.apply(FakeManager(), context, plans, summary, actions)
        row = next(r for r in rows_for(summary) if r['switch'] == 'B')
        check('B exempts urgent rescue', row['result'] == 'exempt:urgent_rescue',
              str(row['result']))

        # ---- B: real relief moves are exempt ----
        allies, enemies, movement, legal, actions = scenario_setup(
            pkg, [('u1', 1, (10.0, 12.0), {'lost': 0.0, 'cooldown': 6.0})],
            [('e5', 5, (14.8, 12.0), {'kind': 'hydralisk'}),
             ('e6', 6, (11.2, 12.0), {'kind': 'zergling'})],
            {'u1': 5}, window=2)
        context = make_world(pkg, allies, enemies, actions, movement, legal, window=2)
        install_stubs(context)
        plans = [plan_view(pkg, 'u1', 'disengage_reenter', 'ESCAPE',
                           'escape: forecast improvement', goal='safe',
                           goal_data={'objective': {'kind': 'escape'}, 'target_id': 'e5'})]
        summary = {'units': {'u1': {'skill': 'disengage_reenter', 'phase': 'ESCAPE'}}}
        switches.apply(FakeManager(), context, plans, summary, actions)
        row = next(r for r in rows_for(summary) if r['switch'] == 'B')
        check('B exempts a threat-shedding retreat',
              row['result'] == 'exempt:real_relief', str(row['result']))

        # ---- B: destination still reaches the target -> keep the move ----
        allies, enemies, movement, legal, actions = scenario_setup(
            pkg, [('u1', 1, (10.0, 12.0), {'lost': 0.0, 'cooldown': 6.0})],
            [('e5', 5, (12.0, 12.0), {'kind': 'hydralisk'})],
            {'u1': 5}, window=2)
        context = make_world(pkg, allies, enemies, actions, movement, legal, window=2)
        install_stubs(context)
        plans = [plan_view(pkg, 'u1', 'disengage_reenter', 'ESCAPE',
                           'escape: forecast improvement', goal='safe',
                           goal_data={'objective': {'kind': 'escape'}, 'target_id': 'e5'})]
        summary = {'units': {'u1': {'skill': 'disengage_reenter', 'phase': 'ESCAPE'}}}
        switches.apply(FakeManager(), context, plans, summary, actions)
        row = next(r for r in rows_for(summary) if r['switch'] == 'B')
        check('B keeps range-preserving retreats',
              row['result'] == 'unchanged:keeps_range', str(row['result']))

        # ---- B: weapon beyond the next macro step -> no interception ----
        allies, enemies, movement, legal, actions = scenario_setup(
            pkg, [('u1', 1, (10.0, 12.0), {'lost': 0.0, 'cooldown': 20.0})],
            [('e5', 5, (14.8, 12.0), {'kind': 'hydralisk'})],
            {'u1': 5}, window=2)
        context = make_world(pkg, allies, enemies, actions, movement, legal, window=2)
        install_stubs(context)
        plans = [plan_view(pkg, 'u1', 'disengage_reenter', 'ESCAPE',
                           'escape: forecast improvement', goal='safe',
                           goal_data={'objective': {'kind': 'escape'}, 'target_id': 'e5'})]
        summary = {'units': {'u1': {'skill': 'disengage_reenter', 'phase': 'ESCAPE'}}}
        switches.apply(FakeManager(), context, plans, summary, actions)
        row = next(r for r in rows_for(summary) if r['switch'] == 'B')
        check('B veto: cooldown beyond next step',
              row['result'] == 'unchanged:weapon_not_ready_in_time', str(row['result']))

        # ---- A+B interplay: B never reverts an A escape ----
        allies_spec = [('u1', 1, (10.0, 10.0), {}),
                       ('u2', 2, (25.0, 25.0), {'lost': 0.0, 'cooldown': 6.0})]
        enemies_spec = [('e0', 0, (8.8, 10.0), {'kind': 'zergling'}),
                        ('e1', 1, (11.8, 10.0), {'kind': 'hydralisk'}),
                        ('e5', 5, (29.8, 25.0), {'kind': 'hydralisk'})]
        allies, enemies, movement, legal, actions = scenario_setup(
            pkg, allies_spec, enemies_spec, {'u1': 6, 'u2': 5}, window=2)
        context = make_world(pkg, allies, enemies, actions, movement, legal, window=2)
        install_stubs(context)
        manager = FakeManager()
        summary = {'units': {}}
        plans = [plan_view(pkg, 'u1', 'focus_fire', 'attacking', 'volley'),
                 plan_view(pkg, 'u2', 'disengage_reenter', 'ESCAPE',
                           'escape: forecast improvement', goal='safe',
                           goal_data={'objective': {'kind': 'escape'}, 'target_id': 'e5'})]
        out = switches.apply(manager, context, plans, summary, actions)
        changed = {r['actor'] for r in rows_for(summary) if r.get('changed')}
        check('A applied and B applied on different actors',
              changed == {'u1', 'u2'} and out[allies['u1'].index] != 6
              and out[allies['u2'].index] == 6 + enemies['e5'].index,
              f'changed={changed}')

        # ---- flags off: the module is inert ----
        work_off = Path(tmp) / 'off'
        (work_off / 'skills').mkdir(parents=True)
        shutil.copytree(SOURCE / 'skills' / 'tactical_v1',
                        work_off / 'skills' / 'tactical_v1',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copytree(SOURCE / 'sc2_skill_adapter' / 'bridge',
                        work_off / 'sc2_skill_adapter' / 'bridge',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        apply_variant(work_off, escape=False, shot_guard=False)
        pkg_off = load_package(str(work_off / 'skills' / 'tactical_v1'),
                               '_hydra_switch_test_off')
        switches_off = importlib.import_module(pkg_off.__name__ + '.hydra_switches')
        check('switches disabled in off-copy', switches_off.ENABLED is False)
        allies, enemies, movement, legal, actions = scenario_setup(
            pkg_off, [('u1', 1, (10.0, 10.0), {})],
            [('e0', 0, (8.8, 10.0), {'kind': 'zergling'}),
             ('e1', 1, (11.8, 10.0), {'kind': 'hydralisk'})],
            {'u1': 6}, window=2)
        context = make_world(pkg_off, allies, enemies, actions, movement, legal,
                             window=2)
        install_stubs(context)
        summary = {}
        out = switches_off.apply(FakeManager(), context, [], summary, actions)
        check('off-copy apply is inert',
              out == actions and 'hydra_switch_decisions' not in summary)

        replay_baseline(pkg_off)

        # The four-way report has its own offline fixture and negative cases.
        import unittest
        from test_hydra_switch_report import ReportTests
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReportTests)
        check('four-way report regressions', unittest.TextTestRunner().run(suite).wasSuccessful())

    print()
    if FAILURES:
        print(f'{len(FAILURES)} FAILURE(S)')
        for line in FAILURES:
            print(' -', line)
        raise SystemExit(1)
    print('All offline checks passed.')


if __name__ == '__main__':
    main()
