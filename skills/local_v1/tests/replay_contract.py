"""Offline contract audit on an existing observation log; never launches SC2.

The supplied next observation remains from the recorded trajectory, so this
checks state consistency, not counterfactual outcomes or policy performance.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path

from decision_tree.local.runtime import LocalController, load_policy
from sc2_skill_adapter.bridge.skill_runtime import load_manager
from skills.local_v1 import FEATURE_NAMES, FAMILIES


CORE = ('unit_states', 'target_assignments', 'all_in', 'all_in_reason',
        'stall_steps', 'contacted', '_last_move', '_last_enemy_health',
        '_last_seen_enemies', '_race_memory', '_regroup_memory', '_search_memory',
        '_dispatch_state', 'elapsed_steps', 'current_skill', 'termination_reason',
        'mode_remaining', 'stats')
COMBAT = ('window_fire', 'range_preserving_move', 'target_scoring',
          'coordinated_approach', 'safe_reentry', 'accurate_damage',
          'critical_escape', 'planned_reentry', 'engagement_commitment',
          'approach_progress_guard')
LIFECYCLE = {'role', 'retreat_steps', 'retreat_cycles', 'recover_steps',
             'progress_since_retreat'}


def legacy_step(manager, payload, n, posture):
    # Existing runner path when spatial switches and other owners are off.
    if not manager.active:
        manager.select('adaptive_focus_fire', posture=posture)
    actions, telemetry = manager.step(payload, n)
    assert isinstance(actions, list)
    return actions, telemetry


def normalized(value):
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items()
                if k != 'unit_value_evaluation_ms'}
    if isinstance(value, (list, tuple)):
        return [normalized(v) for v in value]
    return value


def same_manager(actual, expected):
    for key in CORE:
        assert normalized(getattr(actual, key)) == normalized(getattr(expected, key)), key


def new_manager(metadata):
    p = metadata['skill_parameters']
    return load_manager()(
        max_steps=p['max_steps'], combat_params={k: p[k] for k in COMBAT if k in p},
        unit_value_params=p.get('unit_value_parameters'))


def audit(path):
    metadata = json.loads(path.with_name('run.json').read_text(encoding='utf-8-sig'))
    base = load_policy(Path(__file__).resolve().parents[3] / 'decision_tree/local/base.json')
    old, new = new_manager(metadata), new_manager(metadata)
    posture = metadata['skill_parameters']['posture']
    controller = LocalController(new, base, posture=posture)
    counts, opportunity_steps = Counter(), Counter()
    maxima = Counter()
    tested, pending, episodes = {}, [], set()
    previous_episode = object()
    observations = 0
    for line in path.open(encoding='utf-8-sig'):
        row = json.loads(line)
        if row.get('event') != 'skill_observation':
            continue
        payload, episode = row['payload'], row['episode']
        if episode != previous_episode:
            assert not pending, 'A first candidate was found on an episode final observation'
            old.reset(); new.reset(); controller.reset()
            previous_episode = episode
        episodes.add(episode)
        n = len(payload['state']['allies'])
        for family, forced, expected in pending:
            want, log = legacy_step(expected, deepcopy(payload), n, posture)
            got, actual_log, events = forced.step(deepcopy(payload), n)
            assert got == want and normalized(actual_log) == normalized(log), family
            assert events[0]['selected_id'] == 'BASE', family
            assert events[0]['reason'] == 'intervention_budget_exhausted', family
            assert forced.interventions == 1 and forced.proposals_used == 1, family
            expected.commit_final_actions(want); forced.manager.commit_final_actions(got)
            same_manager(forced.manager, expected)
            tested[family]['next_step_base_and_history_match'] = True
        pending = []
        pre = deepcopy(old) if len(tested) < len(FAMILIES) else None
        untouched = deepcopy(payload)
        want, log = legacy_step(old, deepcopy(payload), n, posture)
        got, actual_log, events = controller.step(payload, n)
        assert payload == untouched, 'Input mutation'
        assert got == want, (episode, row['step'], 'actions')
        assert normalized(actual_log) == normalized(log), (episode, row['step'], 'telemetry')
        event = events[0]
        assert event['selected_id'] == 'BASE'
        assert controller.interventions == controller.proposals_used == 0
        old.commit_final_actions(want); new.commit_final_actions(got)
        same_manager(new, old)
        candidates = event['candidates']
        per_family = Counter(c['family'] for c in candidates)
        assert len(candidates) <= 24 and max(per_family.values(), default=0) <= 8
        assert len({c['effect_key'] for c in candidates}) == len(candidates)
        maxima['total'] = max(maxima['total'], len(candidates))
        for family, count in per_family.items():
            counts[family] += count
            opportunity_steps[family] += 1
            maxima[family] = max(maxima[family], count)
        for c in candidates:
            assert len(c['features']) == 26 and tuple(c['features']) == FEATURE_NAMES
            assert all(isinstance(v, (float, int)) and math.isfinite(v)
                       for v in c['features'].values())
            actor, = c['actors']
            action = c['actions'][actor]
            assert str(action) in payload['questions'][actor]['criteria']
            assert action != got[int(actor[1:])]
            assert c['duration_steps'] == 1
            if c['family'] in tested:
                continue
            cfg = deepcopy(base)
            cfg['selection'].update(mode='forced', anchor_step=0, candidate_id=c['id'])
            forced = LocalController(deepcopy(pre), cfg, posture=posture)
            selected, selected_log, selected_events = forced.step(deepcopy(payload), n)
            differences = [f'u{i}' for i, (a, b) in enumerate(zip(selected, got)) if a != b]
            assert differences == [actor], (c['family'], differences)
            assert selected_events[0]['selected_id'] == c['id']
            assert forced.manager.unit_states[actor]['role'] == c['roles'][actor]
            assert set(forced.manager.unit_states[actor]) <= LIFECYCLE
            assignments = {f'u{i}': f'e{a - 6}' for i, a in enumerate(selected) if a >= 6}
            assert forced.manager.target_assignments == assignments
            assert selected_log['attack_assignments'] == assignments
            if action >= 6:
                assert actor not in forced.manager._last_move
                assert actor not in selected_log['retreat_units']
                assert actor not in selected_log['retreat_actions']
                assert actor not in selected_log['retreat_reasons']
            else:
                assert forced.manager._last_move[actor] == action
                assert actor not in assignments
            forced.manager.commit_final_actions(selected)
            expected = deepcopy(forced.manager)
            expected.local_decision_hook = None
            pending.append((c['family'], forced, expected))
            tested[c['family']] = {'episode': episode, 'step': row['step'],
                                   'candidate_id': c['id'], 'changed_actors': differences,
                                   'next_step_base_and_history_match': False}
        observations += 1
    assert not pending, 'Need a following observation for final forced probe'
    return {'source': str(path), 'episodes': len(episodes), 'observations': observations,
            'base_action_telemetry_core_history_match': True,
            'excluded_timing_field': 'unit_value_evaluation_ms',
            'feature_count': len(FEATURE_NAMES), 'candidate_counts': dict(counts),
            'opportunity_steps': dict(opportunity_steps), 'max_per_step': dict(maxima),
            'forced_family_checks': tested, 'untested_families': sorted(set(FAMILIES) - set(tested)),
            'semantics': 'Current managers, spatial off; recorded observations only; no outcome claims'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('events', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.events), ensure_ascii=False, indent=2))
