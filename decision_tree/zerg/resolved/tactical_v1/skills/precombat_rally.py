"""Bounded opening organization using the same duties as final execution."""
from ..contracts import Commitment, Dependency, ResourceClaims
from ..deployment import opening_available, squads as deployment_squads
from ..encounters import rally_readiness
from .common import h_loops, make_plan, space_claim


class PreCombatRallySkill:
    name = "precombat_rally"

    def __init__(self, config):
        self.config = config
        params = config.get(self.name, {})
        self.ready_radius = float(params.get("ready_radius", 4.0))
        self.ready_fraction = float(params.get("ready_fraction", .8))
        self.max_h = float(params.get("max_h", 8.0))
        self.min_commit_h = float(params.get("min_commit_h", 2.0))
        self.no_progress_h = float(params.get("no_progress_h", 3.0))

    def propose(self, context, memory, telemetry, ledger=None):
        from ..action_duties import actor_duty
        tracker = memory.encounters
        plans = []
        for group in deployment_squads(context):
            members = tuple(group['ordinary'])
            if len(members) < 2:
                continue
            instance = next((i for i in memory.instances.values()
                             if i.skill == self.name and not i.termination_reason
                             and set(i.actors) == set(members)), None)
            if instance is None and not opening_available(context, memory):
                continue
            if instance is not None and instance.extra.get('encounter_id') != tracker.current_id:
                continue
            # Rescue, interception and real fire retain precedence here too.
            # An already fighting/placed member has no deployment duty: do
            # not STOP it or pull it inward in a parallel candidate path.
            duties = {a: actor_duty(context, a, memory) for a in members}
            duties = {a: d for a, d in duties.items()
                      if d.get('required') and d.get('reason', '').startswith('deployment_')}
            if not duties:
                continue
            first = next(iter(duties.values()))
            anchor = tuple(first['anchor'])
            phase = first['deployment_phase']
            actions = {a: d['actions'][0] for a, d in duties.items()}
            goal = f"encounter:{tracker.current_id}"
            key = memory.task_retry_key(self.name, members, 'internal', goal)
            if memory.retry_blocked(*key, context.snapshot.game_loop):
                continue
            quality = rally_readiness(context, first['pending_members'], anchor, self.ready_radius)
            data = dict(instance.extra) if instance else {
                'anchor': list(anchor), 'waypoint': list(anchor),
                'members': list(members), 'encounter_id': tracker.current_id,
                'enemy_ids': sorted(context.snapshot.enemies),
                'ready_radius': self.ready_radius, 'ready_fraction': self.ready_fraction,
                'best_gap': quality['gap'], 'squad_id': group['id'],
                'front_actor': first.get('front_actor'), 'deployment_phase': phase,
                'rally_last_progress_loop': context.snapshot.game_loop,
                'no_progress_loops': h_loops(context, self.no_progress_h),
                'retry_key': key}
            remaining = max(1, h_loops(context, self.max_h)
                            - (context.snapshot.game_loop - tracker.started_loop))
            delta = ({'kind': 'continue', 'instance_id': instance.instance_id}
                     if instance else {'kind': 'create', 'skill': self.name,
                                       'actors': list(members), 'goal': goal,
                                       'goal_kind': 'area', 'phase': 'rallying',
                                       'max_loops': remaining, 'extra': data})
            telemetry.count(self.name, 'applicable')
            plan = make_plan(
                self.name, actions, goal, actions, [],
                ResourceClaims(controls=tuple(sorted(actions)),
                               space=tuple(space_claim(context, a, action)
                                           for a, action in actions.items())),
                Commitment('rallying', h_loops(context, self.min_commit_h),
                           remaining, interrupt_on=('danger_change', 'member_lost')),
                {}, context.snapshot.snapshot_id, self.config.get('_fingerprint', ''),
                instance_id=instance.instance_id if instance else None,
                dependencies=(Dependency('member', tuple(sorted(actions))),),
                memory_delta=delta, atomic=True, goal_kind='area',
                goal_data={'objective': {'kind': 'rally'}, 'anchor': list(anchor),
                           'members': list(members), 'ready_radius': self.ready_radius,
                           'ready_fraction': self.ready_fraction,
                           'encounter_id': tracker.current_id,
                           'deployment_phase': phase,
                           'front_actor': first.get('front_actor'),
                           'member_reasons': {a: d['benefit'] for a, d in duties.items()}},
                rationale=f'{phase.lower()}: execute shared deployment duties')
            plans.append(plan)
            telemetry.count(self.name, 'proposed')
        return plans
