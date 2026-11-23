"""Hydra firing lines must be priced from executable orders on both sides."""
from ..enemy_motion import record_observation
from ..skills.ranged_fire_formation import _forecast, formation_fronts, formation_value
from .fixtures import TacticalTestCase, enemy, scene
from .test_hydra_approach_support import _build


def firing_line(positions, opposition):
    return scene([(p, 'hydralisk') for p in positions], enemies={
        'e' + str(i): enemy(p, 'hydralisk', tag=100 + i) for i, p in enumerate(opposition)})


class RangedFireFormationTest(TacticalTestCase):
    def contact_scene(self, allies, enemies):
        state = scene(allies, enemies=enemies, distance=2., map_name='10gen_zerg')
        state['scenario_brief'] = {'map_size': [32, 32]}
        state['unit_type_profiles']['105']['weapons'][0]['range'] = .1
        for i, (actor, unit) in enumerate(state['allies'].items()):
            unit['tag'] = str(10 + i)
            own_hydra = unit['type'] == 'hydralisk'
            hp = 90. if own_hydra else 35.
            unit.update(health=hp, max_health=hp, effective_health=hp)
            for eid, relation in state['relations'][actor].items():
                d = relation['distance']
                # The center-distance action mask omits the native weapon radii.
                reach, mask = (5., 5.) if own_hydra else (.1, 2.)
                enemy_reach = 5. if enemies[eid]['type'] == 'hydralisk' else .1
                radii = unit['radius'] + enemies[eid]['radius']
                in_range, threatened = d <= reach + radii, d <= enemy_reach + radii
                relation.update(native_weapon_range=reach, smac_attack_distance_limit=mask,
                                smac_attack_available=d <= mask, in_weapon_range=in_range,
                                weapon_range_status='in_range' if in_range else 'out_of_range',
                                in_enemy_weapon_range=threatened,
                                enemy_weapon_range_status='in_range' if threatened else 'out_of_range')
        return state

    def test_isolated_healthy_hydra_closes_mask_gap_then_returns_fire(self):
        enemies = {'e0': enemy((15.32, 10.), 'hydralisk', health=90., max_health=90., tag=100)}
        manager = self.manager()
        for i, (x, expected) in enumerate(((10., 4), (11.125, 6))):
            state = self.contact_scene([((x, 10.), 'hydralisk'), ((4., 20.), 'hydralisk')], enemies)
            state['game_loop'] += 8 * i
            actions, _ = self.step(manager, state)
            self.assertEqual(actions[0], expected)
            manager.commit_final_actions(actions)

    def test_ready_ling_shot_precedes_advancing_into_covered_firing_lane(self):
        state = self.contact_scene(
            [((10., 10.), 'hydralisk'), ((10.8, 13.), 'hydralisk'),
             ((12.6, 10.7), 'zergling')],
            {'e0': enemy((16.05, 10.5), 'hydralisk', health=90., max_health=90., tag=100),
             'e1': enemy((14., 12.), 'zergling', tag=101)})
        actions, trace = self.step(self.manager(), state)
        self.assertEqual(actions[0], 7)
        self.assertEqual(trace['units']['u0']['skill'], 'focus_fire')

    def test_shooter_switches_from_ling_once_flank_hydra_is_shootable(self):
        allies = [((10., 10.), 'hydralisk'), ((4., 20.), 'hydralisk')]
        enemies = {'e0': enemy((13., 10.), 'zergling', tag=100)}
        manager = self.manager()
        actions, _ = self.step(manager, self.contact_scene(allies, enemies))
        self.assertEqual(actions[0], 6)
        manager.commit_final_actions(actions)
        enemies['e0']['health'] = 23.
        enemies['e1'] = enemy((10., 15.32), 'hydralisk', health=90., max_health=90., tag=101)
        state = self.contact_scene(allies, enemies)
        state['game_loop'] += 16
        actions, _ = self.step(manager, state)
        self.assertEqual(actions[0], 6)
        manager.commit_final_actions(actions)
        enemies['e1']['position'] = [10., 14.9]
        state = self.contact_scene(allies, enemies)
        state['game_loop'] += 24
        actions, _ = self.step(manager, state)
        self.assertEqual(actions[0], 7)

    def test_wounded_hydra_advances_to_enemy_hydra_inside_approach_band(self):
        state = self.contact_scene(
            [((10., 10.), 'hydralisk'), ((4., 20.), 'hydralisk')],
            {'e0': enemy((15.32, 10.), 'hydralisk', tag=100),
             'e1': enemy((15.32, 12.), 'hydralisk', tag=101)})
        state['allies']['u0'].update(health=10., effective_health=10.,
                                     health_lost_since_previous_decision=24.)
        actions, trace = self.step(self.manager(), state)
        self.assertEqual(actions[0], 4)
        self.assertEqual(trace['units']['u0']['phase'], 'approach_hydralisk')

    def test_first_sighting_considers_unexposed_targets_and_places_the_group(self):
        state = firing_line([(4., 10.), (4., 12.), (2., 11.)], [(12., 10.), (12., 12.)])
        manager = self.manager()
        ctx = _build(state, manager)
        ctx.baseline_actions = {a: 1 for a in ctx.snapshot.allies}
        manager.memory.update_task_facts(ctx.snapshot, ctx)
        self.assertFalse(any(_forecast(ctx).current_threats.values()))
        candidates = manager.skills['ranged_fire_formation'].propose(ctx, manager.memory, manager.telemetry)
        opening = [p for p in candidates if p.goal_data.get('first_engagement')]
        self.assertEqual({p.goal for p in opening}, {'e0', 'e1'})
        self.assertTrue(all(len(p.current_actions) == 3 for p in opening))
        manager.memory.reset()
        self.assertIsNone(manager.memory.first_hydra_sighting)

    def test_two_fillers_keep_the_leads_shot_and_improve_complete_branch(self):
        state = firing_line([(6., 10.), (4.5, 11.5), (5.5, 13.)], [(10.5, 10.), (11., 8.)])
        manager = self.manager()
        ctx = _build(state, manager)
        base = ctx.baseline_actions = {'u0': 6, 'u1': 1, 'u2': 1}
        self.assertEqual(formation_fronts(ctx), (('u0', 'e0'),))
        ctx.ranged_fire_reference = (('u0', 'e0', formation_value(ctx, 'u0', 'e0', base)),)
        plans = manager.skills['ranged_fire_formation'].propose(ctx, manager.memory, manager.telemetry)
        three = next(p for p in plans if len(p.controlled_actors) == 3)
        self.assertEqual(three.current_actions, {'u0': 6, 'u1': 4, 'u2': 4})
        self.assertEqual(sum(c.kind == 'damage' for c in three.expected_effects), 1)
        # Exercise the real joint evaluator on complete teams. The identical
        # command vector has the same firing value whatever skill supplied it.
        _, p0 = manager.evaluator.score_joint(ctx, [], base, manager.memory)
        _, grouped = manager.evaluator.score_joint(ctx, [three], three.current_actions, manager.memory)
        _, unlabelled = manager.evaluator.score_joint(ctx, [], three.current_actions, manager.memory)
        self.assertEqual(p0['ranged_fire_formation'], 0.)
        self.assertGreater(grouped['ranged_fire_formation'], 0.)
        self.assertEqual(grouped['ranged_fire_formation'], unlabelled['ranged_fire_formation'])

    def test_close_before_a_moving_enemy_reaches_the_front(self):
        state = firing_line([(6., 10.), (5.5, 11.5)], [(12.5, 10.)])
        manager = self.manager()
        stationary = _build(state, manager)
        stationary.baseline_actions = {'u0': 1, 'u1': 1}
        self.assertFalse(formation_fronts(stationary))
        ctx = _build(state, manager)
        ctx.baseline_actions = dict(stationary.baseline_actions)
        ctx.enemy_motion = {}
        record_observation(ctx.enemy_motion, 'e0', 100, 72, (13.75, 10.), 8)
        record_observation(ctx.enemy_motion, 'e0', 100, 80, (12.5, 10.), 8)
        self.assertEqual(formation_fronts(ctx), (('u0', 'e0'),))
        plans = manager.skills['ranged_fire_formation'].propose(ctx, manager.memory, manager.telemetry)
        self.assertTrue(any(p.current_actions['u1'] == 4 for p in plans))

    def test_two_steps_to_cover_only_claims_first_move_and_discounted_future_line(self):
        state = firing_line([(6., 10.), (3.5, 11.5)], [(10.5, 10.)])
        state['scenario_brief'] = {'map_size': [32, 32]}
        manager = self.manager()
        ctx = _build(state, manager)
        ctx.baseline_actions = {'u0': 6, 'u1': 1}
        plans = manager.skills['ranged_fire_formation'].propose(ctx, manager.memory, manager.telemetry)
        pair = next(p for p in plans if p.current_actions == {'u0': 6, 'u1': 4})
        self.assertEqual(_forecast(ctx).firing('u1', 4, 'e0'), (0., .5))
        self.assertEqual(sum(c.kind == 'damage' for c in pair.expected_effects), 1)
        self.assertEqual(len(pair.current_actions), 2)
        # No progress score just for closing part of an arbitrarily large gap.
        far = firing_line([(6., 10.), (2., 11.5)], [(10.5, 10.)])
        far['scenario_brief'] = state['scenario_brief']
        self.assertEqual(_forecast(_build(far, manager)).firing('u1', 4, 'e0'), (0., 0.))
        # The hypothetical second step would cross the known public boundary.
        state['scenario_brief']['map_size'] = [5.5, 32]
        self.assertEqual(_forecast(_build(state, manager)).firing('u1', 4, 'e0'), (0., 0.))

    def test_threatened_covering_shooter_keeps_current_shot_but_needs_a_replacement(self):
        state = firing_line([(6., 10.), (6., 11.6), (4.75, 12.7)], [(10.5, 10.)])
        state['unit_type_profiles']['107']['weapons'][0]['attack_period_loops'] = 12
        manager = self.manager()
        base = {'u0': 6, 'u1': 6, 'u2': 1}
        replace = dict(base, u2=4)
        healthy = _build(state, manager)
        state['allies']['u1'].update(health=5., health_lost_since_previous_decision=12.)
        hurt = _build(state, manager)
        hurt.baseline_actions = base
        front = ('u0', 'e0')
        gain = lambda context: formation_value(context, *front, replace) - formation_value(context, *front, base)
        self.assertGreater(gain(hurt), gain(healthy))
        plans = manager.skills['ranged_fire_formation'].propose(hurt, manager.memory, manager.telemetry)
        self.assertTrue(any({**base, **p.current_actions} == replace for p in plans))
        # If the following volley cannot occur within 2H anyway, the
        # only remaining contribution is the legal current shot, at full value.
        state['unit_type_profiles']['107']['weapons'][0]['attack_period_loops'] = 20
        hurt = _build(state, manager)
        state['allies']['u1']['health_lost_since_previous_decision'] = 0.
        no_recent_loss = _build(state, manager)
        self.assertEqual(formation_value(hurt, *front, base), formation_value(no_recent_loss, *front, base))
