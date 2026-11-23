"""Committed movement continuity and observed route reliability across skills."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.contracts import Commitment, ResourceClaims
from skills.tactical_v1.evaluator import Evaluator
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skill_manager import SkillManager
from skills.tactical_v1 import config as tactical_config
from skills.tactical_v1.skills.common import make_plan
from skills.tactical_v1.skills.disengage_reenter import DisengageReenterSkill
from skills.tactical_v1.skills.local_formation import LocalFormationSkill
from skills.tactical_v1.skills.coordinated_engage import CoordinatedEngageSkill
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import enemy, scene, with_questions


class MovementCoordinationTests(unittest.TestCase):
    def setUp(self):
        self.memory = TacticalMemory()

    def observe(self, state):
        payload = state if 'state' in state else with_questions(state)
        c = Context(build_snapshot(payload, payload['state']['game_loop']))
        self.memory.begin_step(c.snapshot)
        self.memory.observe(c.snapshot)
        self.memory.update_task_facts(c.snapshot, c)
        return c

    def state(self, loop=80, x=0., health=45.):
        s = scene([((x, 0.), 'marine'), ((x + .5, 0.), 'marine')], loop=loop,
                  enemies={'e0': enemy((15., 0.))}, distance=2.)
        s['allies']['u0']['health'] = health
        return s

    def commit(self, c, actor='u0', action=4, skill='coordinated_engage', goal='e0'):
        p = make_plan(skill, (actor,), goal, {actor: action}, [],
                      ResourceClaims(controls=(actor,)), Commitment('approaching'), {},
                      c.snapshot.snapshot_id, 'test', goal_kind='enemy')
        self.memory.commit(c.snapshot, p.current_actions, [p], context=c)
        return p

    def test_reversal_cost_is_pure_and_expires_without_renewal_on_skill_switch(self):
        c = self.observe(self.state())
        self.commit(c)
        until = self.memory.movement_intents['u0']['until_loop']
        c = self.observe(self.state(88, x=1.125))
        before = copy.deepcopy(self.memory.__dict__)
        self.assertEqual(self.memory.movement_conflict(c, 'u0', 5), 1.)
        self.assertEqual(self.memory.movement_conflict(c, 'u0', 4), 0.)
        self.assertEqual(self.memory.movement_conflict(c, 'u0', 2), 0.)
        self.assertEqual(self.memory.movement_conflict(c, 'u0', 1), 0.)
        self.assertEqual(self.memory.__dict__, before)
        self.commit(c, skill='regroup', goal='different')
        self.assertEqual(self.memory.movement_intents['u0']['until_loop'], until)
        c = self.observe(self.state(until, x=2.25))
        self.assertEqual(self.memory.movement_conflict(c, 'u0', 5), 0.)

    def test_zero_commit_duration_does_not_penalize_reversal(self):
        self.memory = TacticalMemory(encounter_config={'lifecycle': {'combat_cycle_commit_h': 0.}})
        c = self.observe(self.state())
        self.commit(c)
        c = self.observe(self.state(88, 1.125))
        self.assertEqual(self.memory.movement_conflict(c, 'u0', 5), 0.)

    def test_actual_blockage_demotes_route_and_releases_direction_lock(self):
        c = self.observe(self.state())
        self.commit(c)
        c = self.observe(self.state(88, .05))
        before = copy.deepcopy(self.memory.__dict__)
        self.assertGreater(self.memory.movement_blockage(c, 'u0', 4), .9)
        self.assertEqual(self.memory.movement_blockage(c, 'u0', 5), 0.)
        self.assertEqual(self.memory.movement_conflict(c, 'u0', 5), 0.)
        self.assertEqual(before, self.memory.__dict__)
        c = self.observe(self.state(112, .05))
        self.assertEqual(self.memory.movement_blockage(c, 'u0', 4), 0.)

    def test_identity_change_clears_observed_movement_records(self):
        c = self.observe(self.state())
        self.commit(c)
        s = self.state(88)
        s['allies']['u0']['tag'] = 'new-actor'
        c = self.observe(s)
        self.assertNotIn('u0', self.memory.movement_intents)
        self.assertFalse(self.memory.blocked_directions)

    def test_severe_damage_allows_reverse_emergency(self):
        c = self.observe(self.state())
        self.commit(c)
        s = self.state(88, 1.125, health=5.)
        s['allies']['u0']['health_lost_since_previous_decision'] = 30.
        c = self.observe(s)
        self.assertEqual(self.memory.movement_conflict(c, 'u0', 5), 0.)

    def test_formation_changes_a_recently_blocked_direction(self):
        skill = LocalFormationSkill({})
        c = self.observe(self.state())
        p = next(p for p in skill.propose(c, self.memory, Telemetry())
                 if p.controlled_actors == ('u0',))
        self.memory.commit(c.snapshot, p.current_actions, [p], context=c)
        c = self.observe(self.state(88))  # no movement happened
        q = next(p for p in skill.propose(c, self.memory, Telemetry())
                 if p.controlled_actors == ('u0',))
        self.assertNotEqual(p.current_actions['u0'], q.current_actions['u0'])

    def recorded(self, episode, steps):
        path = Path(__file__).resolve().parents[3] / 'runs/tactical_6z4s_20_rejoin_20261113/6z4s/events.jsonl'
        if not path.exists():
            self.skipTest('optional recorded regression observations unavailable')
        result = {}
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if '"event": "skill_observation"' not in line:
                    continue
                row = json.loads(line)
                if row['episode'] == episode and row['step'] in steps:
                    result[row['step']] = row['payload']
                    if len(result) == len(steps):
                        break
        self.assertEqual(set(result), set(steps))
        return result

    def test_seed175_g6_cannot_reverse_committed_g2_contact_move(self):
        observations = self.recorded(6, {19, 20})
        c = self.observe(observations[19])
        # Isolate direction continuity. Default arrival safety now rejects
        # this recorded 66-HP unit's 72-damage observed loss window.
        config = {'common_combat': {'arrival_exposure_enabled': False}}
        p = next(p for p in CoordinatedEngageSkill(config).propose(c, self.memory, Telemetry())
                 if p.controlled_actors == ('u5',) and p.current_actions['u5'] == 4)
        self.memory.commit(c.snapshot, p.current_actions, [p], context=c)
        c = self.observe(observations[20])
        self.assertGreater(self.memory.movement_conflict(c, 'u5', 5), .9)
        plans = LocalFormationSkill({}).propose(c, self.memory, Telemetry())
        self.assertFalse(any(p.current_actions.get('u5') == 5 for p in plans))

    def test_seed175_complete_manager_keeps_contact_direction(self):
        observations = self.recorded(6, {19, 20})
        c = self.observe(observations[19])
        config = tactical_config.load()
        config['common_combat']['arrival_exposure_enabled'] = False
        p = next(p for p in CoordinatedEngageSkill(config).propose(c, self.memory, Telemetry())
                 if p.controlled_actors == ('u5',) and p.current_actions['u5'] == 4)
        self.memory.commit(c.snapshot, p.current_actions, [p], context=c)
        with patch('skills.tactical_v1.skill_manager.config_module.load', return_value=config):
            manager = SkillManager()
        manager.memory = self.memory
        intent = copy.deepcopy(self.memory.movement_intents['u5'])
        actions, summary = manager.step(observations[20], 10)
        # A joint formation adjustment can use the northward component of
        # the same approach. It must make actual legal-contact progress,
        # rather than be forced to repeat the former east-only tie break.
        c = Context(build_snapshot(observations[20], observations[20]['state']['game_loop']))
        self.assertIn(actions[5], c.legal_actions('u5'))
        self.assertIn(actions[5], (2, 4))
        before = c.projected_attack_opportunity('u5', intent['enemy'],
                                                c.snapshot.allies['u5'].position)
        after = c.projected_attack_opportunity('u5', intent['enemy'],
                                               c.predict_position('u5', actions[5]).position)
        self.assertLess(after.center_gap, before.center_gap)
        manager.commit_final_actions(actions)
        current = manager.memory.movement_intents['u5']
        self.assertEqual(current['direction'], intent['direction'])
        self.assertEqual(current['until_loop'], intent['until_loop'])
        self.assertGreater(manager.memory.movement_conflict(c, 'u5', 5), .9)
        self.assertEqual(manager.memory.pending_moves['u5']['action'], actions[5])
        task = manager.memory.instance_for('coordinated_engage', 'u5')
        if summary['units']['u5']['skill'] == 'coordinated_engage':
            self.assertIsNotNone(task)
            self.assertEqual(task.instance_id, p.instance_id)
        else:
            self.assertIsNone(task)  # baseline cannot fabricate a continued G2 task

    def test_final_evaluator_penalizes_baseline_reversal_without_plan_wrapper(self):
        c = self.observe(self.state())
        self.commit(c)
        c = self.observe(self.state(88, 1.125))
        evaluator = SkillManager().evaluator
        _, reversing = evaluator.score_joint(c, [], {'u0': 5, 'u1': 1}, self.memory)
        _, continuing = evaluator.score_joint(c, [], {'u0': 4, 'u1': 1}, self.memory)
        self.assertLess(reversing['movement_continuity'], 0.)
        self.assertEqual(continuing['movement_continuity'], 0.)

    def test_seed184_g4_keeps_lower_forecast_risk_west_alternative(self):
        c = self.observe(self.recorded(15, {24})[24])
        skill = DisengageReenterSkill({})
        plans = [p for p in skill.propose(c, self.memory, Telemetry())
                 if p.controlled_actors == ('u0',) and p.commitment.phase == 'ESCAPE']
        self.assertIn(5, [p.current_actions['u0'] for p in plans])
        self.assertGreaterEqual(len(plans), 2)
        evaluator = Evaluator()
        orders = []
        for action in c.move_destinations('u0'):
            prediction = c.predict_position('u0', action)
            probe = skill._forecast_pressure(c, 'u0', prediction.position, prediction.certain)
            full = evaluator._forecast_cost(c, evaluator.effects(c, {'u0': action}))[0]
            orders.append((action, probe, full))
        self.assertEqual([x[0] for x in sorted(orders, key=lambda x: (x[1], x[0]))],
                         [x[0] for x in sorted(orders, key=lambda x: (x[2], x[0]))])

    def test_seed183_observed_collision_marks_only_failed_direction(self):
        observations = self.recorded(14, {12, 13})
        c = self.observe(observations[12])
        self.commit(c, 'u5', 5, 'disengage_reenter', 'safe')
        c = self.observe(observations[13])
        self.assertGreater(self.memory.movement_blockage(c, 'u5', 5), .8)
        self.assertEqual(self.memory.movement_blockage(c, 'u5', 2), 0.)


if __name__ == '__main__':
    unittest.main()
