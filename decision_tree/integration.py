"""One owner for skill scheduling; previews never mutate the live manager."""
from copy import deepcopy
import hashlib

from .schema import load_policy
from .features import extract_features, phase_name, infer_races
from .policy import candidate_calls, choose, initial_call, bind_call, extract_policy_features

CALL_KEYS = {'operation', 'skill', 'posture', 'priority_target', 'max_steps',
             'adaptive_params', 'movement_profile'}
SPATIAL = {'regroup', 'cohesive_search'}


def executable_call(call):
    return {k: deepcopy(v) for k, v in call.items() if k in CALL_KEYS and v is not None}


def merge_actions(result, payload, n_agents, fallback):
    actions, telemetry = result
    if isinstance(actions, dict):
        merged = fallback(payload, n_agents)
        for actor, action in actions.items():
            merged[int(str(actor).lstrip('u'))] = action
        telemetry = {**telemetry, 'base_policy': 'nearest', 'merged_units': sorted(actions)}
        actions = merged
    return actions, telemetry


class SkillTreeController:
    def __init__(self, manager, policy, *, race, posture='balanced'):
        self.manager = manager
        self.policy = load_policy(policy)
        self.race = race
        self.posture = posture
        self.reset()

    def reset(self):
        self.memory = {'contacted': False, 'last_visible_loop': None}
        self.opportunities = 0
        self.interventions = 0
        self.override_active = False
        self.call_number = 0
        self.using_legacy = True

    def _available(self, call, state):
        skill = call.get('skill')
        if call.get('movement_profile') and not self.manager.spatial_params.get('enable_combat_positioning'):
            return False
        if skill in SPATIAL:
            if not self.manager.spatial_params.get('enable_' + skill, False):
                return False
            # These executors explicitly hand control back on contact/damage.
            if state.get('visible_enemies') or any(
                u.get('health_lost_since_previous_decision', 0) +
                u.get('shield_lost_since_previous_decision', 0) > 0
                for u in state['allies'].values() if u.get('alive')):
                return False
        if skill == 'heal_assign' and not any(state.get('heal_candidates', {}).values()):
            return False
        return True

    def _preview(self, call, payload, n_agents, fallback):
        trial = deepcopy(self.manager)
        trial.apply_call(executable_call(call))
        result = trial.step(payload, n_agents)
        if result[1].get('status') == 'interrupted':
            return None
        return {'actions': merge_actions(result, payload, n_agents, fallback)[0],
                'effective_config': {'skill': trial.current_skill, 'priority_target': trial.priority_target,
                                     **deepcopy(trial.last_invocation['effective_config'])}}

    def _start(self, payload, n_agents, fallback, spatial_choice, *, allow_gate=True):
        state = payload['state']
        features = extract_policy_features(self.policy, state, self.memory)
        # Availability is a runtime capability fact, not hidden scenario data.
        for key in ('enable_regroup', 'enable_cohesive_search', 'enable_combat_positioning'):
            if key in features:
                features[key] = float(self.manager.spatial_params.get(key, False))
        base = {'operation': 'start', 'skill': spatial_choice or 'adaptive_focus_fire',
                'posture': self.posture}
        initial = initial_call(self.policy, features)
        races = infer_races(state)
        supported = self.race in self.policy['supported_races'] and all(
            r in self.policy['supported_races'] for r in races)
        self.using_legacy = True
        if supported and initial.get('delegate') != 'legacy':
            bound = bind_call(initial, state, base)
            if bound is not None and self._available(bound, state):
                base = bound
                self.using_legacy = False
        phase = phase_name(features)
        reason = 'unsupported_race' if not supported else 'outside_gate'
        selected = {'candidate_id': 'BASE', 'call': base, 'predicted_gains': {}, 'reason': reason}
        alternatives, base_actions, previews = [], None, {}
        configurations, generated, rejected = {}, [], {}
        base_preview = None
        gate = self.policy['gate']
        expanded = self.policy.get('exploration', {}).get('enabled', False)
        next_call = self.call_number + 1
        eligible = (supported and allow_gate and
                    (expanded or self.opportunities < self.policy['selection']['max_interventions_per_episode']) and
                    phase in gate['phases'] and state.get('game_loop', 0) >= gate['min_game_loop'])
        if eligible:
            proposals = candidate_calls(self.policy, state, base, features)
            generated = [row['id'] for row in proposals]
            alternatives = []
            for row in proposals:
                if self._available(row['call'], state):
                    alternatives.append(row)
                else:
                    rejected[row['id']] = 'capability_unavailable'
            base_preview = self._preview(base, payload, n_agents, fallback)
            base_actions = base_preview['actions'] if base_preview is not None else None
            for row in alternatives:
                preview = self._preview(row['call'], payload, n_agents, fallback)
                previews[row['id']] = preview['actions'] if preview is not None else None
                configurations[row['id']] = preview['effective_config'] if preview is not None else None
            if expanded:
                # Every available alternative is independently eligible. An
                # equal first action is not evidence of equal full-call effect.
                usable = []
                for row in alternatives:
                    cid = row['id']
                    if previews[cid] is None:
                        rejected[cid] = 'preview_interrupted'
                    elif base_preview is None:
                        rejected[cid] = 'base_preview_interrupted'
                    elif configurations[cid] == base_preview['effective_config']:
                        rejected[cid] = 'identical_effective_config'
                    else:
                        usable.append(row)
                seed = self.policy['exploration']['seed']
                usable.sort(key=lambda row: hashlib.sha256(
                    f"{seed}:{next_call}:{row['id']}".encode()).hexdigest())
                alternatives = usable
                eligible = bool(alternatives)
                reason = 'no_distinct_available_candidate'
            else:
                required = {row['id'] for row in self.policy['candidates']}
                # Preserve the old fixed multi-output collection contract.
                eligible = (set(previews) == required and base_actions is not None and
                            all(a is not None and (a != base_actions or
                                (gate.get('effect_check', 'first_action') == 'call_config' and
                                 configurations[cid] != base_preview['effective_config']))
                                for cid, a in previews.items()))
                reason = 'no_distinct_complete_candidate_set'
            if eligible:
                self.opportunities += 1
                selection = self.policy['selection']
                target = selection.get('anchor_call_number')
                if expanded and target is not None and target != next_call:
                    reason = 'outside_selected_anchor'
                elif expanded and self.interventions >= selection['max_interventions_per_episode']:
                    reason = 'intervention_budget_exhausted'
                else:
                    selected = choose(self.policy, features, alternatives, base, sampling_key=next_call)
                    reason = selected['reason']
        if (expanded and self.policy['selection'].get('anchor_call_number') == next_call
                and not eligible):
            reason = 'selected_anchor_unavailable:' + reason
        self.manager.apply_call(executable_call(selected['call']))
        self.override_active = selected['candidate_id'] != 'BASE'
        if self.override_active:
            self.interventions += 1
        self.call_number += 1
        event = {'event': 'skill_tree_decision', 'game_loop': state['game_loop'],
                'phase': phase, 'features': features, 'opportunity': bool(eligible),
                'opportunity_index': self.opportunities if eligible else None,
                'base_call': base, 'candidate_calls': alternatives,
                'base_actions': base_actions, 'candidate_actions': previews,
                'selected_id': selected['candidate_id'], 'selected_call': selected['call'],
                'predicted_gains': selected.get('predicted_gains', {}), 'reason': reason,
                'call_number': self.call_number}
        if expanded:
            selected_config = {'skill': self.manager.current_skill,
                               'priority_target': self.manager.priority_target,
                               **deepcopy(self.manager.last_invocation['effective_config'])}
            event.update(candidate_generated=generated, candidate_rejected=rejected,
                         base_effective_config=base_preview['effective_config'] if base_preview else None,
                         candidate_effective_configs=configurations,
                         selected_effective_config=selected_config, invocation_applied=True,
                         first_action_changed=(previews.get(selected['candidate_id']) != base_actions
                                               if self.override_active and base_actions is not None else None),
                         exploration_mode=self.policy['selection']['mode'],
                         exploration_seed=self.policy['exploration']['seed'],
                         sampling_probability=selected.get('sampling_probability'),
                         interventions_applied=self.interventions)
            if 'candidate_support' in selected:
                event['candidate_support'] = deepcopy(selected['candidate_support'])
        return event

    def step(self, payload, n_agents, fallback):
        state = payload['state']
        events = []
        if not self.manager.active:
            self.override_active = False
        spatial_choice = None
        if not self.manager.active:
            features = extract_policy_features(self.policy, state, self.memory)
            initial = initial_call(self.policy, features)
            supported = self.race in self.policy['supported_races'] and set(infer_races(state)) <= set(self.policy['supported_races'])
            bound = bind_call(initial, state) if initial.get('delegate') != 'legacy' else None
            self.using_legacy = not (supported and bound is not None and self._available(bound, state))
        if self.using_legacy and not self.override_active and self.manager.spatial_params.get('auto_spatial_dispatch'):
            spatial_choice = self.manager.spatial_dispatch(payload)
        if not self.manager.active:
            events.append(self._start(payload, n_agents, fallback, spatial_choice))
        result = self.manager.step(payload, n_agents)
        if (self.manager.current_skill in SPATIAL and isinstance(result[0], dict)
                and result[1].get('status') == 'interrupted' and not result[0]):
            # Single same-observation handoff, never two simultaneous schedulers.
            self.override_active = False
            events.append(self._start(payload, n_agents, fallback, None, allow_gate=False))
            result = self.manager.step(payload, n_agents)
        actions, telemetry = merge_actions(result, payload, n_agents, fallback)
        telemetry = {**telemetry, 'skill_tree_call_number': self.call_number,
                     'skill_tree_override_active': self.override_active}
        if state.get('visible_enemies'):
            self.memory['contacted'] = True
            self.memory['last_visible_loop'] = state['game_loop']
        return actions, telemetry, events
