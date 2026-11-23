"""Offline acceptance for the round-2 hydra switches (no SC2, no engine).

Covers the required minimum:
1. With both switches OFF, a patched copy reproduces the frozen baseline's
   final actions AND micro-step authorizations on recorded observations
   (replayed through the real frozen manager and the patched manager).
2. A2 rejects pressure-relief-only escapes that have no named supporting
   mate (the old A's failure mode).
3. B2 does not intercept urgent rescues, beneficial support moves, retreat
   roles, unreadable intents, or A2's own valid withdrawals, and records
   cannot_reissue instead of fabricating an attack.
4. In genuinely qualifying cases A2 and B2 each change the final submitted
   action, and an A2 withdrawal does not acquire a micro-step attack
   authorization role.

No historical map, seed or unit tag is hardcoded in the decision logic under
test; recorded payloads are used as data only.

Run:  python scripts/test_hydra_supported_switches.py
"""
import copy
import importlib
import json
import py_compile
import re
import shutil
import sys
import tempfile
import textwrap
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ADAPTER = HERE.parent
sys.path.insert(0, str(ADAPTER))

from hydra_supported_switches import apply_variant, RUNNER_RECEIPT_HOOK, RUNNER_COLLECT  # noqa: E402

SOURCE = (ADAPTER.parent / 'runs/contact_response_micro_20261121_153552_909880'
          / 'source_snapshot')
REPLAY_RUN = (ADAPTER.parent / 'runs/contact_response_micro_20261121_153552_909880'
              / 'micro8-hydra')
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


THREAT_R = 2.5      # forecast horizon reach (context.risk_at stub)
ENGAGE_R = 2.0      # enemy weapon envelope vs the actor (relations)
ATTACK_R = 5.0      # SMAC command mask radius
NATIVE_REACH = 5.0  # effective_attack_reach stub


def ally_view(pkg, actor, index, position, hp=40.0, max_hp=80.0, lost=10.0,
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
        current_actions={}, expected_effects=[], claims=None, dependencies=(),
        commitment=contracts.Commitment(phase), features={},
        proposed_memory_delta={}, rationale=rationale, goal_kind=goal_kind,
        goal_data=goal_data or {})


class FakeMemory:
    def battle_target(self, context, actor):
        return None


class FakeEvaluator:
    def effects(self, context, actions):
        return {}


class FakeManager:
    def __init__(self):
        self.memory = FakeMemory()
        self.evaluator = FakeEvaluator()
        self._hydra_supported_state = None


def build_context(pkg, allies_spec, enemies_spec, final_by_actor, window=2,
                  mask_overrides=None):
    allies, enemies = {}, {}
    for actor, index, position, kw in allies_spec:
        allies[actor] = ally_view(pkg, actor, index, position, **kw)
    for enemy_id, index, position, kw in enemies_spec:
        enemies[enemy_id] = enemy_view(pkg, enemy_id, index, position, **kw)
    relations, movement, legal = {}, {}, {}
    contracts = importlib.import_module(pkg.__name__ + '.contracts')
    for actor, unit in allies.items():
        row = {}
        for enemy_id, enemy in enemies.items():
            dist = ((unit.position[0] - enemy.position[0]) ** 2 +
                    (unit.position[1] - enemy.position[1]) ** 2) ** 0.5
            override = (mask_overrides or {}).get((actor, enemy_id), {})
            fields = dict(
                distance=dist, in_weapon_range=dist <= ATTACK_R,
                weapon_range_status='in_range' if dist <= ATTACK_R else 'out_of_range',
                native_weapon_range=5.0,
                in_enemy_weapon_range=dist <= ENGAGE_R,
                enemy_weapon_range_status='in_range' if dist <= ENGAGE_R else 'out_of_range',
                smac_attack_available=dist <= ATTACK_R)
            fields.update(override)
            row[enemy_id] = contracts.Relation(**fields)
        relations[actor] = row
        position = unit.position
        movement[actor] = {a: {'destination': [position[0] + dx, position[1] + dy]}
                           for a, (dx, dy) in ((2, (0, 1)), (3, (0, -1)), (4, (1, 0)), (5, (-1, 0)))}
        legal[actor] = sorted([1, *movement[actor], *(6 + e.index for e in enemies.values())])
    contracts = importlib.import_module(pkg.__name__ + '.contracts')
    snapshot = contracts.Snapshot(
        snapshot_id=window, game_loop=8 * (window - 1), step_mul=8,
        steps_remaining=100, map_name='10gen_zerg', episode=1,
        search_location=(16.0, 16.0), allies=allies, enemies=enemies,
        relations=relations, movement_candidates=movement, heal_candidates={},
        legal={a: frozenset(legal[a]) for a in legal}, keep_heal_order={},
        previous_action_ids={}, raw_state={
            'allies': {a: {'tag': f'{a}-tag', 'orders': []} for a in allies},
            'visible_enemies': {e: {'tag': f'{e}-tag', 'alive': True}
                                for e in enemies},
            'unit_type_profiles': {'107': {'movement_speed_native': 1.5},
                                   '105': {'movement_speed_native': 2.0}}},
        n_allies=len(allies), n_enemies=max(e.index for e in enemies.values()) + 1,
        keep_heal_action_id=None)
    context = importlib.import_module(pkg.__name__ + '.context').Context(snapshot)
    install_stubs(context)
    actions = [1] * (max(u.index for u in allies.values()) + 1)
    for actor in allies:
        actions[allies[actor].index] = final_by_actor[actor]
    return context, actions, allies, enemies


def install_stubs(context):
    context_mod = importlib.import_module(_pkg_name(context) + '.context')

    def risk_at(actor, position, horizon=None):
        threats = tuple(e for e, enemy in context.snapshot.enemies.items()
                        if ((position[0] - enemy.position[0]) ** 2 +
                            (position[1] - enemy.position[1]) ** 2) ** 0.5 < THREAT_R)
        return context_mod.RiskReport(threats=threats, heuristic=False,
                                      effect_weapon_near=())

    def fire_estimate(actor, enemy_id, *args, **kwargs):
        return Estimate()

    def reach(actor, enemy_id):
        return NATIVE_REACH

    def contact_status(actor, enemy_id):
        relation = context.snapshot.relations[actor][enemy_id]
        return 'in_range' if relation.distance <= ATTACK_R else 'out_of_range'

    def attackable(actor):
        return sorted((e for e in context.snapshot.enemies
                       if context.snapshot.relations[actor][e].distance <= ATTACK_R
                       and context.attack_action(actor, e) is not None),
                      key=lambda e: (context.snapshot.relations[actor][e].distance, e))

    context.risk_at = risk_at
    context.fire_estimate = fire_estimate
    context.effective_attack_reach = reach
    context.contact_status = contact_status
    context.attackable_enemies = attackable
    context.forced_hydra_targets = lambda actor: ()


def _pkg_name(context):
    return type(context).__module__.rsplit('.', 1)[0]


class Estimate:
    window_shots = 1
    damage_est = 12.0
    damage_lo = 10.0


def stub_module(switches):
    """Deterministic module collaborators for synthetic geometry tests."""
    switches.support_path_clear = lambda *a, **k: True
    switches.rescue_actions = lambda *a, **k: ()
    switches.action_position = lambda *a, **k: (None, None)
    switches.validate_actions = lambda *a, **k: []
    switches.validate_plans = lambda *a, **k: []
    return switches


def rows_for(summary):
    return summary.get('hydra_supported_decisions', [])


def formation_row(actor, goal):
    return {'skill': 'ranged_fire_formation', 'phase': 'exclusive_control',
            'goal': goal, 'goal_kind': 'enemy'}


def main():
    with tempfile.TemporaryDirectory() as tmp:
        # ---------------------------------------------------- patched copies
        work = Path(tmp) / 'work'
        (work / 'skills').mkdir(parents=True)
        for relative in (('skills/tactical_v1',), ('skills/current',),
                         ('sc2_skill_adapter/bridge',)):
            shutil.copytree(SOURCE / relative[0], work / relative[0],
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        flags = apply_variant(work, supported_escape=True, guard_support_move=True)
        check('patch flags recorded', flags['supported_escape'] is True
              and flags['guard_support_move'] is True)
        tactical = work / 'skills' / 'tactical_v1'
        for name in ('hydra_supported.py', 'skill_manager.py',
                     'skills/ranged_fire_formation.py', 'validator.py'):
            py_compile.compile(str(tactical / name), doraise=True)
        py_compile.compile(str(work / 'sc2_skill_adapter' / 'bridge' / 'runner.py'),
                           doraise=True)
        check('patched files compile', True)
        runner_text = (work / 'sc2_skill_adapter' / 'bridge' / 'runner.py').read_text(
            encoding='utf-8')
        for anchor in ("telemetry.get('hydra_supported_decisions', ())",
                       "switch_row['sc2_receipt'] = dict(",
                       "record['hydra_supported']"):
            check(f'runner hook present: {anchor[:30]}', anchor in runner_text)

        pkg = load_package(str(tactical), '_hydra_supported_test_on')
        switches = stub_module(importlib.import_module(pkg.__name__ + '.hydra_supported'))
        check('switches enabled in on-copy', switches.ENABLED is True)

        # Exercise the injected runner hooks (including its telemetry copy), not SC2.
        first = dict(actor='u0', game_loop=8, changed=True, candidate_action=5)
        second = dict(actor='u0', game_loop=16, changed=True, candidate_action=5)
        env = dict(skill_manager=types.SimpleNamespace(commit_final_actions=lambda actions: None),
                   episode=1, scenario={'sc2_seed': 7}, switch_rows=[], env=None,
                   action_feedback=lambda _, loop: {'actors': {'u0': dict(
                       requested_action_id='5', submission='accepted', response={'code': loop})}})
        for row, cmd in [(first, 5), (second, 6)]:
            env.update(telemetry={'hydra_supported_decisions': [row]}, loop=row['game_loop'], actions=[cmd])
            exec(textwrap.dedent(RUNNER_COLLECT), env)
            exec(textwrap.dedent(RUNNER_RECEIPT_HOOK), env)
        check('runner does not attribute a later action or receipt to an earlier decision',
              first['sc2_receipt']['code'] == 8 and not second['changed'] and second['final_action'] == 6)

        # Pursuit is along the enemy-to-landing ray, using the ENEMY weapon.
        for mate_pos, expected in [((3., 4.95), 'if_pursued'), ((7.4, 4.), None)]:
            ctx, _, _, _ = build_context(pkg, [('u0', 0, (3., -1.), {}), ('u1', 1, mate_pos, {})],
                                         [('e0', 0, (0., 0.), {})], {'u0': 6, 'u1': 1})
            ctx._enemy_weapon_range = lambda *args: .15  # plus radii = 1, not relation's own 5
            route = switches._support_route(ctx, 'u0', (4., 0.), 'e0', ['u1'])
            check(f'A2 pursuit geometry: {expected}', (route or {}).get('class') == expected)

        # ------------------------------------------ A2: named support triple
        # u1 attacked by a ling/hydra pincer; u2 stands where its reach
        # already covers the northern landing spot but cannot fire on either
        # pincer threat from where it stands.
        context, actions, allies, enemies = build_context(
            pkg,
            [('u1', 1, (10.0, 10.0), {}),
             ('u2', 2, (10.0, 15.5), {'lost': 0.0, 'cooldown': 6.0})],
            [('e0', 0, (10.9, 10.0), {'kind': 'zergling'}),
             ('e1', 1, (9.1, 10.0), {'kind': 'hydralisk'})],
            {'u1': 6, 'u2': 1})
        context.hydra_move_roles = {}
        manager = FakeManager()
        summary = {}
        plans = [plan_view(pkg, 'u1', 'focus_fire', 'attacking', 'volley')]
        out = switches.apply(manager, context, plans, summary, actions)
        row_a = next(r for r in rows_for(summary) if r['actor'] == 'u1'
                     and r['switch'] == 'A2')
        check('A2 records the named triple', row_a['support_threat'] == 'e0'
              and row_a['support_mate'] == 'u2' and row_a['support_class'] == 'now',
              f"{row_a.get('support_threat')}/{row_a.get('support_mate')}/{row_a.get('support_class')}")
        check('A2 changes the submitted action', row_a['result'] == 'applied'
              and out[allies['u1'].index] == row_a['candidate_action']
              and out[allies['u1'].index] != 6,
              f"result={row_a['result']} final={out[allies['u1'].index]}")
        check('A2 override does not acquire a micro intent role',
              summary['units']['u1']['skill'] == 'supported_escape'
              and summary['units']['u1']['phase'] == 'one_window')
        micro_intents = importlib.import_module(pkg.__name__ + '.skill_manager')._micro_intents
        summary.update(execution_duties={'actors': {}}, joint_rescue_substitutions={})
        check('real frozen micro filter issues no attack authorization for A2',
              not micro_intents(context, summary, plans, out))
        contract = plan_view(pkg, 'u1', 'focus_fire', 'attacking', 'promised volley')
        contract.dependencies = (types.SimpleNamespace(kind='fire_window'),)
        committed, move = switches._evaluate_A2(context, 'u1', [1, 6, 1], [contract], {},
                                                switches._new_state(), switches.CONFIG, ['u1', 'u2'])
        check('A2 keeps a selected fire-window commitment',
              move is None and committed['result'] == 'unchanged:fire_commitment')
        switches.confirm(manager, context, [1, 6, 1])
        check('uncommitted withdrawal is not followed up', not manager._hydra_supported_state['tracked'])

        # Use the real frozen forced-target calculation for a mixed-target window.
        del context.forced_hydra_targets
        forced = context.forced_hydra_targets('u1')
        check('A2 cannot bypass frozen mandatory fire even with an exact escape permit',
              forced == ('e1',) and not switches._select(manager, context, [], [1, 7, 1], set(), row_a)
              and row_a['veto'] == 'forced_hydra_targets')

        # ------------------- A2 rejects pressure-only, support-less escapes
        # Same pincer, but the only mate is far away: every destination is
        # outside every mate's reach (old A would have applied here).
        context, actions, allies, enemies = build_context(
            pkg,
            [('u1', 1, (10.0, 10.0), {}),
             ('u2', 2, (25.0, 25.0), {'lost': 0.0, 'cooldown': 6.0})],
            [('e0', 0, (10.9, 10.0), {'kind': 'zergling'}),
             ('e1', 1, (9.1, 10.0), {'kind': 'hydralisk'})],
            {'u1': 6, 'u2': 1})
        context.hydra_move_roles = {}
        summary = {}
        out = switches.apply(FakeManager(), context, plans, summary, actions)
        row_a = next(r for r in rows_for(summary) if r['actor'] == 'u1'
                     and r['switch'] == 'A2')
        check('A2 rejects pressure-only escape without a named mate',
              row_a['result'] == 'unchanged:no_support_route'
              and out[allies['u1'].index] == 6, str(row_a['result']))

        # -------------------------------- A2 veto: ready, nothing pressing
        context, actions, allies, enemies = build_context(
            pkg,
            [('u1', 1, (10.0, 10.0), {'cooldown': 0.0, 'ready': True, 'lost': 0.0})],
            [('e0', 0, (12.2, 10.0), {'kind': 'zergling'})],
            {'u1': 6})
        context.hydra_move_roles = {}
        summary = {}
        switches.apply(FakeManager(), context, [], summary, actions)
        row_a = next(r for r in rows_for(summary) if r['switch'] == 'A2')
        check('A2 veto: ready without close pressure or bad exchange',
              row_a['result'] == 'unchanged:exchange_not_unfavorable',
              str(row_a['result']))

        # ------------------- A2 follow-up: arrival vs mate support vs fire
        context, actions, allies, enemies = build_context(
            pkg,
            [('u1', 1, (10.0, 10.0), {}),
             ('u2', 2, (10.0, 15.5), {'lost': 0.0, 'cooldown': 6.0})],
            [('e0', 0, (10.9, 10.0), {'kind': 'zergling'}),
             ('e1', 1, (9.1, 10.0), {'kind': 'hydralisk'})],
            {'u1': 6, 'u2': 1}, window=2)
        context.hydra_move_roles = {}
        manager = FakeManager()
        summary = {}
        out = switches.apply(manager, context, plans, summary, actions)
        # The runner normally calls confirm() through the commit hook.
        switches.confirm(manager, context, out)
        # Next window: actor stands on the predicted destination, the mate
        # has fired (cooldown restart beyond decay), but the threats never
        # advanced into the mate's range: arrival, firing and support are
        # three different observed facts.
        context, actions, allies, enemies = build_context(
            pkg,
            [('u1', 1, (10.0, 11.0), {'lost': 0.0, 'cooldown': 0.0}),
             ('u2', 2, (10.0, 15.5), {'lost': 0.0, 'cooldown': 2.0})],
            [('e0', 0, (10.9, 10.0), {'kind': 'zergling'}),
             ('e1', 1, (9.1, 10.0), {'kind': 'hydralisk'})],
            {'u1': 6, 'u2': 1}, window=3)
        context.hydra_move_roles = {}
        summary = {}
        switches.apply(manager, context, [], summary, actions)
        follow = [r for r in rows_for(summary) if r.get('followup')]
        check('A2 follow-up separates arrival, mate range and mate fire',
              len(follow) == 1 and follow[0]['arrived_at_predicted'] is True
              and follow[0]['designated_mate_fire_proxy'] is True
              and all('_in_designated_mate_range' not in v
                      for v in (follow[0]['threat_state'] or {}).values()),
              json.dumps(follow, default=str)[:400])

        # ------------------------------------------ B2 qualifying override
        # Formation support move walking away from its own focus enemy while
        # the next shot is due within this window.
        context, actions, allies, enemies = build_context(
            pkg, [('u1', 1, (10.0, 12.0), {'lost': 0.0, 'cooldown': 6.0})],
            [('e5', 5, (14.8, 12.0), {'kind': 'hydralisk'})],
            {'u1': 5})
        context.hydra_move_roles = {('u1', 5): 'support'}
        context.snapshot.raw_state['allies']['u1']['orders'] = [dict(kind='attack', target_tag='e5-tag')]
        summary = {'units': {'u1': formation_row('u1', 'e5')}}
        plans = [plan_view(pkg, 'u1', 'ranged_fire_formation', 'exclusive_control',
                           'support during weapon cycle', goal='e5', goal_kind='enemy')]
        manager = FakeManager()
        out = switches.apply(manager, context, plans, summary, actions)
        row_b = next(r for r in rows_for(summary) if r['switch'] == 'B2')
        check('B2 overrides an intentless support move',
              row_b['result'] == 'applied' and out[allies['u1'].index] == 6 + 5
              and row_b['current_target'] == 'e5'
              and row_b['support_target'] == 'e5', str(row_b['result']))
        check('B2 records whom the move supported',
              row_b['supporting_for'] == [], str(row_b.get('supporting_for')))
        switches.confirm(manager, context, out)
        follow = switches._followups(manager._hydra_supported_state, context, switches.CONFIG)
        check('B2 follow-up has attack tag but no move arrival',
              follow[0]['target_tag'] == 'e5-tag' and follow[0]['arrived_at_predicted'] is None)

        context, actions, _, _ = build_context(pkg, [('u0', 0, (10., 12.), {})],
            [('e0', 0, (14.8, 12.), {}), ('e1', 1, (4.6, 12.), {})], {'u0': 5})
        context.hydra_move_roles = {('u0', 5): 'entry'}
        summary = {'units': {'u0': formation_row('u0', 'e1')}}
        context.snapshot.raw_state['allies']['u0']['orders'] = [dict(kind='attack', target_tag='e0-tag')]
        r, shot = switches._evaluate_B2(context, 'u0', actions, [], summary, FakeManager(),
                                        switches.CONFIG, ['u0'])
        check('B2 keeps a new support lane even when it costs range on its own target',
              shot is None and r['result'] == 'exempt:support_line_at_destination'
              and r['target_tag'] == 'e0-tag' and r['support_target_tag'] == 'e1-tag')
        context.snapshot.raw_state['allies']['u0']['orders'] = []
        r, shot = switches._evaluate_B2(context, 'u0', actions, [], summary, FakeManager(),
                                        switches.CONFIG, ['u0'])
        check('formation focus alone is not a verifiable existing attack',
              shot is None and r['veto'] == 'no_current_attack_source')

        # --------------------------- B2 exemption: beneficial support move
        # The destination still forms the intended line on the focus enemy.
        context, actions, allies, enemies = build_context(
            pkg, [('u1', 1, (10.0, 12.0), {'lost': 0.0, 'cooldown': 6.0})],
            [('e5', 5, (11.5, 12.0), {'kind': 'hydralisk'})],
            {'u1': 5})
        context.hydra_move_roles = {('u1', 5): 'support'}
        context._debug_stationary = {'u1': 11}
        summary = {'units': {'u1': formation_row('u1', 'e5')}}
        switches.apply(FakeManager(), context, plans, summary, actions)
        row_b = next(r for r in rows_for(summary) if r['switch'] == 'B2')
        check('B2 keeps a support move whose line still forms',
              row_b['result'] == 'unchanged:keeps_range', str(row_b['result']))

        # --------------------------- B2 exemption: urgent rescue provenance
        context, actions, allies, enemies = build_context(
            pkg, [('u1', 1, (10.0, 12.0), {'lost': 0.0, 'cooldown': 6.0})],
            [('e5', 5, (14.8, 12.0), {'kind': 'hydralisk'})],
            {'u1': 5})
        context.hydra_move_roles = {('u1', 5): 'support'}
        summary = {'units': {'u1': formation_row('u1', 'e5')},
                   'joint_rescue_substitutions': {'u1': {'reason': 'lethal_pressure'}}}
        switches.apply(FakeManager(), context, plans, summary, actions)
        row_b = next(r for r in rows_for(summary) if r['switch'] == 'B2')
        check('B2 exempts urgent rescue provenance',
              row_b['result'] == 'exempt:urgent_rescue', str(row_b['result']))

        # ----------------- B2 scope: escape roles and unreadable intents
        for role, expected in (('escape', 'out_of_scope:intent_unreadable'),):
            context, actions, allies, enemies = build_context(
                pkg, [('u1', 1, (10.0, 12.0), {'lost': 0.0, 'cooldown': 6.0})],
                [('e5', 5, (14.8, 12.0), {'kind': 'hydralisk'})],
                {'u1': 5})
            context.hydra_move_roles = {('u1', 5): role}
            summary = {'units': {'u1': formation_row('u1', 'e5')}}
            switches.apply(FakeManager(), context, plans, summary, actions)
            row_b = next(r for r in rows_for(summary) if r['switch'] == 'B2')
            check(f'B2 leaves formation {role} moves alone',
                  row_b['result'] == expected, str(row_b['result']))

        # ---- B2 cannot_reissue: focus enemy no longer on the command mask
        context, actions, allies, enemies = build_context(
            pkg, [('u1', 1, (10.0, 12.0), {'lost': 0.0, 'cooldown': 6.0})],
            [('e5', 5, (15.4, 12.0), {'kind': 'hydralisk'})],
            {'u1': 5}, mask_overrides={('u1', 'e5'): {'smac_attack_available': False}})
        context.hydra_move_roles = {('u1', 5): 'support'}
        context.snapshot.raw_state['allies']['u1']['orders'] = [dict(kind='attack', target_tag='e5-tag')]
        summary = {'units': {'u1': formation_row('u1', 'e5')}}
        out = switches.apply(FakeManager(), context, plans, summary, actions)
        row_b = next(r for r in rows_for(summary) if r['switch'] == 'B2')
        check('B2 records cannot_reissue instead of fabricating an attack',
              row_b['result'] == 'cannot_reissue:target_not_menu_legal'
              and out[allies['u1'].index] == 5, str(row_b['result']))

        # -------------------------- A2 + B2: B2 never reverts an A2 escape
        context, actions, allies, enemies = build_context(
            pkg,
            [('u1', 1, (10.0, 10.0), {}),
             ('u2', 2, (10.0, 15.5), {'lost': 0.0, 'cooldown': 6.0})],
            [('e0', 0, (10.9, 10.0), {'kind': 'zergling'}),
             ('e1', 1, (9.1, 10.0), {'kind': 'hydralisk'})],
            {'u1': 6, 'u2': 1})
        context.hydra_move_roles = {}
        manager = FakeManager()
        summary = {}
        out = switches.apply(manager, context, plans, summary, actions)
        b2_rows = [r for r in rows_for(summary) if r['switch'] == 'B2'
                   and r['actor'] == 'u1']
        changed = {r['actor'] for r in rows_for(summary) if r.get('changed')}
        check('B2 skips the actor A2 already moved',
              changed == {'u1'} and not b2_rows
              and out[allies['u1'].index] != 6, f"changed={changed}")

        # -------- double-off copy reproduces the frozen baseline verbatim
        work_off = Path(tmp) / 'off'
        (work_off / 'skills').mkdir(parents=True)
        for relative in (('skills/tactical_v1',), ('skills/current',),
                         ('sc2_skill_adapter/bridge',)):
            shutil.copytree(SOURCE / relative[0], work_off / relative[0],
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        apply_variant(work_off, supported_escape=False, guard_support_move=False)
        pkg_off = load_package(str(work_off / 'skills' / 'tactical_v1'),
                               '_hydra_supported_test_off')
        switches_off = stub_module(importlib.import_module(
            pkg_off.__name__ + '.hydra_supported'))
        check('switches disabled in off-copy', switches_off.ENABLED is False)
        context, actions, allies, enemies = build_context(
            pkg_off,
            [('u1', 1, (10.0, 10.0), {}),
             ('u2', 2, (10.0, 16.0), {'lost': 0.0, 'cooldown': 6.0})],
            [('e0', 0, (10.9, 10.0), {'kind': 'zergling'}),
             ('e1', 1, (9.1, 10.0), {'kind': 'hydralisk'})],
            {'u1': 6, 'u2': 1})
        context.hydra_move_roles = {}
        summary = {}
        out = switches_off.apply(FakeManager(), context, [], summary, actions)
        check('off-copy apply is inert',
              out == actions and 'hydra_supported_decisions' not in summary)

        # ------- replay recorded observations through both real managers:
        # the double-off patched manager must reproduce the unpatched frozen
        # manager's final actions and micro-step authorizations exactly.
        payloads, n_agents = [], None
        shard_events = REPLAY_RUN / 'zerg_3zergling_0baneling_7hydralisk_00' / 'events.jsonl'
        with shard_events.open('rb') as stream:
            for line in stream:
                if not re.match(rb'^\s*\{\s*"event"\s*:\s*"skill_observation"', line):
                    continue
                event = json.loads(line)
                payload = event['payload']
                payloads.append(payload)
                n_agents = n_agents or len(payload['state']['allies'])
                if len(payloads) >= 2:
                    break
        check('recorded payloads available', len(payloads) == 2 and n_agents == 10,
              f'{len(payloads)} payloads, n_agents={n_agents}')

        def replay(managers):
            results = []
            for manager in managers:
                manager.reset()
                episodes = []
                for payload in payloads:
                    actions, summary = manager.step(payload, n_agents)
                    manager.commit_final_actions(actions)
                    episodes.append((list(actions), [dict(i) for i in manager.last_micro_intents]))
                results.append(episodes)
            return results

        frozen_pkg = load_package(str(SOURCE / 'skills' / 'tactical_v1'),
                                  '_hydra_supported_test_frozen')
        frozen_manager = importlib.import_module(
            frozen_pkg.__name__ + '.skill_manager').SkillManager(tactical_config=None)
        off_manager = importlib.import_module(
            pkg_off.__name__ + '.skill_manager').SkillManager(tactical_config=None)
        frozen_result, off_result = replay([frozen_manager, off_manager])
        check('double-off reproduces frozen final actions',
              frozen_result == off_result,
              f'{frozen_result[:1]} vs {off_result[:1]}')
        check('double-off reproduces frozen micro authorizations',
              all(f[1] == o[1] for f, o in zip(frozen_result, off_result)),
              str([(f[1], o[1]) for f, o in zip(frozen_result, off_result)][:1]))
        # Restore the real validator and all frozen collaborators for replay.
        importlib.reload(switches)
        on_manager = importlib.import_module(
            pkg.__name__ + '.skill_manager').SkillManager(tactical_config=None)
        on_result = replay([on_manager])[0]
        # The on-copy runs the recorded baseline windows without crashing;
        # whether it diverges depends on whether those specific windows
        # contain qualifying situations (data, not logic).
        check('on-copy replays recorded windows without error',
              len(on_result) == len(frozen_result))

    print()
    if FAILURES:
        print(f'{len(FAILURES)} FAILURE(S)')
        for line in FAILURES:
            print(' -', line)
        raise SystemExit(1)
    print('All offline acceptance checks passed.')


if __name__ == '__main__':
    main()
