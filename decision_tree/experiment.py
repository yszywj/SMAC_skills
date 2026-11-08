"""Frozen execution context, independent of scenario and selected study arm."""
import hashlib
import json
from copy import deepcopy
import importlib.metadata
import os
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def policy_contract(policy):
    return {k: deepcopy(v) for k, v in policy.items() if k != 'selection'}


def execution_context(metadata):
    fields = ('method', 'skill_version', 'env_args', 'skill_parameters', 'versions',
              'observation_protocol', 'source_hashes', 'native_teacher_control',
              'medivac_policy', 'medical_patient_coordination')
    context = {k: deepcopy(metadata[k]) for k in fields}
    context['action_extensions'] = deepcopy(metadata.get('action_extensions', {}))
    context['policy_contract'] = deepcopy(metadata['skill_tree']['policy_contract'])
    return context


def validate_deployment(metadata, policy):
    model = policy['selection'].get('model')
    context = execution_context(metadata)
    if model is not None:
        if model.get('context_hash') != digest(context) or model.get('context') != context:
            raise ValueError('Skill return model was fitted under another execution context; collect new data')
    return context


def expected_context(args, manifest, policy):
    """Preflight every evaluation arm before any game is started."""
    from sc2_skill_adapter.bridge.config import ROOT, SMAC_ROOT, map_file
    from sc2_skill_adapter.bridge.skill_runtime import manager_configuration, source_files
    from sc2_skill_adapter.bridge.keep_order import validate_keep_heal_order
    config = manager_configuration(args)
    values = config['value_metadata']
    race = getattr(args, 'race_profile', 'off')
    paths = list((ROOT/'bridge').glob('*.py'))
    paths += [SMAC_ROOT/'smacv2/env/starcraft2'/name for name in ('starcraft2.py', 'wrapper.py', 'distributions.py')]
    paths += [map_file(manifest['env_args']), *source_files('current')]
    paths += sorted((ROOT.parent/'decision_tree').glob('*.py'))
    metadata = {
        'method': 'adaptive-skill', 'skill_version': 'current', 'env_args': manifest['env_args'],
        'skill_parameters': {'posture': args.adaptive_posture, 'max_steps': args.skill_max_steps,
            **config['manager_kwargs']['combat_params'],
            'spatial_params': config['manager_kwargs'].get('spatial_params', {}),
            **({'race_profile': race} if race != 'off' else {}), **values},
        'versions': {n: importlib.metadata.version(n) for n in ('pysc2', 'numpy', 'protobuf', 's2clientprotocol', 'httpx')},
        'observation_protocol': 'smacv2-visible-rows-v2',
        'source_hashes': {os.path.relpath(p, ROOT.parent): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'native_teacher_control': 'off', 'medivac_policy': 'legacy',
        'medical_patient_coordination': config['medical_patient_coordination'],
        'action_extensions': validate_keep_heal_order(args, 'adaptive-skill', manifest['env_args']) or {},
        'skill_tree': {'policy_contract': policy_contract(policy)}}
    return validate_deployment(metadata, policy)
