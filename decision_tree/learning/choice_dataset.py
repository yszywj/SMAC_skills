"""Versioned, observation-derived shared vectors for sparse candidate labels."""
from copy import deepcopy

from decision_tree.choice_features import CHOICE_FEATURE_NAMES, extract_choice_features
from decision_tree.learning.dataset import read_run, episode, require
from decision_tree.learning.expanded_dataset import build_expanded_dataset


def promote_choice_dataset(data, base_run):
    """Add shared vectors after auditing isolated calls; old labels stay intact.

    Recompute from observations and earlier visible contact memory rather than
    trusting arbitrary extra fields in a serialized feature row.
    """
    result = deepcopy(data)
    base = read_run(base_run)
    require(base['run.json']['skill_tree']['policy'].get('exploration', {}).get('feature_schema') == 3,
            'choice_feature_protocol_required')
    require(result['provenance']['BASE']['hashes'] == base['hashes'], 'base_provenance_changed')
    cache = {}
    for row in result['rows']:
        number = row['base_episode']
        if number not in cache:
            ep = episode(base, number)
            memory = {'contacted': False, 'last_visible_loop': None}
            vectors = {}
            wanted = {r['gate']['step'] for r in result['rows'] if r['base_episode'] == number}
            for step, observation in sorted(ep['observations'].items()):
                state = observation['payload']['state']
                if step in wanted:
                    vectors[step] = extract_choice_features(state, memory)
                if state.get('visible_enemies'):
                    memory['contacted'] = True
                    memory['last_visible_loop'] = state['game_loop']
            cache[number] = vectors
        expected = cache[number][row['gate']['step']]
        require(row['state_features'] == expected, 'choice_feature_observation_derivation_mismatch')
        row['vector_features'] = expected
    result.update(schema_version=3, vector_feature_names=list(CHOICE_FEATURE_NAMES))
    result['protocol']['shared_vector'] = 'same_predecision_state_and_semantic_target_context_for_every_candidate'
    result['protocol']['vector_derivation'] = 'recomputed_from_observations_and_public_contact_memory'
    return result


def build_choice_dataset(base_run, arm_runs, plan=None):
    return promote_choice_dataset(build_expanded_dataset(base_run, arm_runs, plan), base_run)
