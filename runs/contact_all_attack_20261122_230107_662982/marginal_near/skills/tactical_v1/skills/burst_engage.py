"""S3: bounded approach to a real, one-use ground explosion opportunity.

Commands never claim that a Baneling exploded. Region allocation uses the
projected source center and residual health. Effects recomputes final joint
actions, including death-triggered explosions, independently of plan labels.
"""
from typing import Dict, List

from ..burst import (action_region, action_value, candidate_actions, outcome,
                     useful_area_value, cover_window)
from ..context import Context, STOP
from ..contracts import Commitment, EffectClaim, HEURISTIC, ResourceClaims
from .common import make_plan, space_claim


class BurstEngageSkill:
    name = "burst_engage"

    def __init__(self, config: Dict):
        self.config = config

    def cluster_value(self, context: Context, center) -> float:
        return useful_area_value(context, center)

    def _delta(self, context, instance, actor, enemy_id, phase, center):
        progress = getattr(context, 'burst_progress', None)
        record = progress.records.get(actor) if progress is not None else None
        if record and record.get('deadline_expired'):
            # A new one-window proposal must not recreate the expired
            # persistent approach under another goal or phase.
            return ({'kind': 'drop', 'instance_id': instance.instance_id,
                     'reason': 'approach_deadline_expired'} if instance else None)
        data = {"goal": enemy_id, "goal_kind": "enemy", "phase": phase,
                "extra": {"waypoint": center,
                          "target_start_position": context.snapshot.enemies[enemy_id].position,
                          "no_progress_loops": 3 * context.H}}
        if phase == 'staging':
            data['extra']['burst_wait_started'] = (
                instance.extra.get('burst_wait_started', context.snapshot.game_loop)
                if instance else context.snapshot.game_loop)
        if instance is None:
            data.update(kind="create", skill=self.name, actors=[actor],
                        max_loops=6 * context.H)
        elif instance.goal != enemy_id:
            data.update(kind="retarget", instance_id=instance.instance_id)
        elif instance.phase != phase:
            data.update(kind="transition", instance_id=instance.instance_id)
        else:
            return None
        return data

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        plans = []
        actors = [actor for actor, cap in sorted(context.capabilities.items())
                  if cap.suicide_attack and context.snapshot.allies[actor].alive]
        telemetry.count(self.name, "applicable", len(actors))
        for actor in actors:
            instance = memory.instance_for(self.name, actor)
            progress = getattr(context, 'burst_progress', None)
            execution = progress.execution(context, actor) if progress is not None else None
            if execution and execution['required'] and execution.get('target') is None:
                for action in execution['actions']:
                    plans.append(make_plan(
                        self.name, [actor], 'support', {actor: action}, [],
                        ResourceClaims(controls=(actor,), space=(space_claim(context, actor, action),)),
                        Commitment('supporting', 0, max_loops=context.H), {},
                        context.snapshot.snapshot_id, fingerprint, provider=self.name,
                        goal_kind='area', rationale=execution['reason']))
            allow_wait = progress is not None and progress.wait_available(context, actor)
            options = candidate_actions(context, actor, allow_wait=allow_wait,
                                        weights=self.config.get('weights'))
            proposed = 0
            for action in options:
                region = action_region(context, actor, action)
                if region is None:
                    continue
                enemy_id, center = region.target_id, region.position
                if memory.retry_blocked(*memory.task_retry_key(self.name, [actor], 'enemy', enemy_id),
                                        context.snapshot.game_loop) and not (
                        execution and execution['required'] and action in execution['actions']):
                    continue
                result = outcome(context, actor, action)
                semantic = context.decode(actor, action)
                productive = result.opportunity > 0. and useful_area_value(context, result.position) > 0.
                phase = ('bursting' if productive
                         else 'staging' if semantic.kind == 'stop' else 'approaching')
                value = action_value(context, actor, action, weights=self.config.get('weights'))
                effects = [EffectClaim("coverage", enemy_id, None, context.H,
                            HEURISTIC, "one_use_trigger_opportunity_not_observed_hit")]
                if semantic.kind == 'attack' and semantic.target != enemy_id:
                    # A projected death burst can favor a different victim
                    # than the native attack order. Declare that order too;
                    # all actual damage remains final-action recomputation.
                    effects.append(EffectClaim("coverage", semantic.target, None,
                        context.H, HEURISTIC, "native_attack_attempt_not_observed_hit"))
                plans.append(make_plan(
                        self.name, [actor], enemy_id, {actor: action},
                        effects,
                        ResourceClaims(controls=(actor,),
                                       space=(space_claim(context, actor, action),)),
                        Commitment(phase, context.H, max_loops=6 * context.H,
                                   interrupt_on=("target_lost", "illegal", "no_progress", "danger_change")),
                        {}, context.snapshot.snapshot_id, fingerprint,
                        provider=self.name,
                        instance_id=(instance.instance_id if instance and not (
                            progress and progress.records.get(actor, {}).get('deadline_expired')) else None),
                        memory_delta=self._delta(context, instance, actor, enemy_id, phase, center),
                        goal_kind="enemy",
                        goal_data={"target_id": enemy_id, "waypoint": list(center),
                                   "objective": {"kind": "burst_region"},
                                   "burst_region": list(center),
                                   "trigger_position": list(result.position),
                                   "trigger_opportunity": result.trigger,
                                   "ready_cover": cover_window(context, actor, enemy_id)[0],
                                   "execution_duty": execution,
                                   "approach_deadline_due": bool(progress and progress.overdue(context, actor))},
                        rationale=f"{phase} one-use region {enemy_id}; native residual value={value:.3f}"))
                proposed += 1
            if not proposed and (not context.snapshot.enemies or not context.burst_profile().geometry_known):
                plans.append(make_plan(
                    self.name, [actor], "hold", {actor: STOP}, [],
                    ResourceClaims(controls=(actor,)), Commitment("declined", 0,
                    max_loops=context.H), {}, context.snapshot.snapshot_id, fingerprint,
                    provider=self.name, rationale="no visible compatible unblocked ground region"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans
