"""One-step local decisions made BEFORE the baseline commits its histories."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def load_policy(source):
    p = json.loads(Path(source).read_text(encoding='utf-8-sig')) if isinstance(source, (str, Path)) else deepcopy(source)
    if not isinstance(p, dict) or p.get('schema_version') != 1 or p.get('kind') != 'local_patch_policy':
        raise ValueError('Expected local_patch_policy schema 1')
    if set(p) - {'schema_version', 'kind', 'selection', 'candidates', 'model', 'veto_model', 'description'}:
        raise ValueError('Unknown local policy fields')
    s = p.setdefault('selection', {})
    if set(s) - {'mode', 'candidate_id', 'anchor_step', 'max_interventions', 'min_gain'}:
        raise ValueError('Unknown local selection fields')
    s.setdefault('mode', 'base'); s.setdefault('max_interventions', 1); s.setdefault('min_gain', 0.0)
    if s['mode'] not in ('base', 'forced', 'model'):
        raise ValueError('Local mode must be base, forced or model')
    if type(s['max_interventions']) is not int or s['max_interventions'] != 1:
        raise ValueError('Local v1 evaluates exactly one intervention per episode at most')
    if type(s['min_gain']) not in (int, float) or not math.isfinite(s['min_gain']) or s['min_gain'] < 0:
        raise ValueError('min_gain must be finite and nonnegative')
    if s['mode'] == 'forced':
        if type(s.get('anchor_step')) is not int or s['anchor_step'] < 0 or not isinstance(s.get('candidate_id'), str) or not s['candidate_id']:
            raise ValueError('Forced mode requires candidate_id and nonnegative anchor_step')
    elif 'anchor_step' in s or 'candidate_id' in s:
        raise ValueError('Anchors and forced IDs are collection-only')
    c = p.setdefault('candidates', {})
    if set(c) - {'max_candidates', 'max_per_family'}:
        raise ValueError('Unknown candidate bounds')
    for key, default in (('max_candidates', 24), ('max_per_family', 8)):
        c.setdefault(key, default)
        if type(c[key]) is not int or not 1 <= c[key] <= 128:
            raise ValueError('Candidate bounds must be integers in [1,128]')
    if s['mode'] == 'model':
        from .models import validate_model, model_hash
        from skills.local_v1 import FEATURE_NAMES
        validate_model(p.get('model'))
        if p['model']['kind'] != 'local_gain_model' or p['model']['feature_names'] != list(FEATURE_NAMES):
            raise ValueError('Wrong gain model or feature schema')
        if p.get('veto_model') is not None:
            validate_model(p['veto_model'])
            v = p['veto_model']
            if v['kind'] != 'local_veto_model' or v.get('old_model_hash') != model_hash(p['model']):
                raise ValueError('Veto must reference this exact frozen gain model')
            if v['context_hash'] != p['model']['context_hash'] or v['feature_names'] != p['model']['feature_names']:
                raise ValueError('Veto execution context does not match')
            if v['old_min_gain'] != s['min_gain']:
                raise ValueError('Veto selection threshold differs from the frozen old selector')
            if v.get('game_version') != p['model'].get('game_version'):
                raise ValueError('Veto game version differs from the frozen gain model')
    elif p.get('model') is not None or p.get('veto_model') is not None:
        raise ValueError('Models require model selection mode')
    return p


def validate_options(args, method, env_args):
    if method != 'adaptive-skill' or getattr(args, 'skill_version', 'current') != 'current':
        raise ValueError('--local-policy requires adaptive-skill with current skills')
    if env_args.get('map_name') != '10gen_protoss' or getattr(args, 'race_profile', 'off') != 'off':
        raise ValueError('Local v1 is restricted to the Protoss baseline')
    if (getattr(args, 'skill_tree', None) is not None
            or getattr(args, 'medivac_policy', 'legacy') != 'legacy'
            or getattr(args, 'pair_mode', 'off') != 'off'
            or getattr(args, 'native_teacher_control', 'off') != 'off'):
        raise ValueError('Local policy must own selection; other tree/teacher channels must be off')
    if any(getattr(args, key, False) for key in ('enable_regroup', 'enable_cohesive_search',
                                               'enable_combat_positioning', 'auto_spatial_dispatch')):
        raise ValueError('Local v1 fixes shared-memory semantics: legacy spatial switches must be off')


def execution_context(metadata, policy):
    """Exclude dataset identity and selection arm, include baseline and candidate semantics."""
    env = {k: v for k, v in metadata['env_args'].items() if k not in ('seed', 'replay_dir')}
    keys = ('source_hashes', 'skill_parameters', 'observation_protocol', 'versions', 'action_extensions')
    return digest({'contract': 'local-one-step-v1', 'env_args': env,
                   'candidates': policy['candidates'], **{k: metadata.get(k) for k in keys}})


class LocalController:
    def __init__(self, manager, policy, *, posture='balanced', context_hash=None):
        self.manager = manager
        self.policy = load_policy(policy)
        self.posture = posture
        if getattr(manager, 'race_profile', 'off') != 'off' or manager._spatial_features_enabled():
            raise ValueError('Local controller requires fixed non-spatial baseline')
        if getattr(manager, 'local_decision_hook', None) is not None:
            raise ValueError('Only one local decision owner is allowed')
        if self.policy['selection']['mode'] == 'model':
            if context_hash is None or self.policy['model']['context_hash'] != context_hash:
                raise ValueError('Model does not match the frozen execution context')
        manager.local_decision_hook = self._decide
        self.reset()

    def reset(self):
        self.step_number = 0
        self.interventions = 0
        self.proposals_used = 0
        self.events = []
        self.last_key = None
        self.last_input_hash = None
        self.cached = None

    def _decide(self, payload, baseline):
        from skills.local_v1 import generate_candidates, apply_candidate
        from .models import predict, select_candidate
        candidates = generate_candidates(payload, baseline, self.policy['candidates'])
        selection = self.policy['selection']
        selected = proposed = None
        scores = {}
        vetoed = False
        reason = 'base'
        if self.proposals_used < selection['max_interventions']:
            if selection['mode'] == 'forced' and self.step_number == selection['anchor_step']:
                selected = next((c for c in candidates if c['id'] == selection['candidate_id']), None)
                if selected is None:
                    raise ValueError('Forced candidate missing at the declared anchor')
                proposed = selected
                reason = 'forced'
            elif selection['mode'] == 'model':
                choice = select_candidate(self.policy['model'], candidates, selection['min_gain'])
                scores = choice['predictions']
                reason = choice['reason']
                if choice['candidate_id'] != 'BASE':
                    selected = proposed = next(c for c in candidates if c['id'] == choice['candidate_id'])
                    if self.policy.get('veto_model'):
                        veto = predict(self.policy['veto_model'], selected['family'], selected['features'])
                        if veto['supported'] and veto['value'] > 1e-9:
                            selected = None; vetoed = True; reason = 'veto_to_base'
        else:
            reason = 'intervention_budget_exhausted'
        result = baseline if selected is None else apply_candidate(payload, baseline, selected)
        # VETO cancels the old policy's one intervention, not reschedules it.
        # Subsequent decisions must remain BASE to match its training estimand.
        if proposed is not None:
            self.proposals_used += 1
        if selected is not None:
            self.interventions += 1
        # This is a proposal/commit record, not evidence of native command acceptance.
        event = {'event': 'local_decision', 'step': self.step_number,
                 'game_loop': payload['state']['game_loop'], 'baseline_actions': list(baseline['actions']),
                 'baseline_roles': deepcopy(baseline.get('roles', {})),
                 'baseline_predicted_damage': deepcopy(baseline.get('predicted_damage') or {}),
                 'candidates': candidates, 'selected_id': selected['id'] if selected else 'BASE',
                 'proposed_id': proposed['id'] if proposed else 'BASE', 'vetoed': vetoed,
                 'reason': reason, 'predictions': scores, 'final_actions': list(result['actions']),
                 'proposals_used': self.proposals_used, 'interventions': self.interventions}
        self.events.append(event)
        return result

    def step(self, payload, n_agents, fallback=None):
        key = (payload['state'].get('episode'), payload['state']['game_loop'])
        input_hash = digest({'payload': payload, 'n_agents': n_agents})
        if key == self.last_key:
            if input_hash != self.last_input_hash:
                raise ValueError('Observation changed at the same clock')
            return deepcopy(self.cached[0]), deepcopy(self.cached[1]), []
        if self.last_key is not None and (key[0] != self.last_key[0] or key[1] <= self.last_key[1]):
            raise ValueError('Observation clock moved backwards; reset episode explicitly')
        self.events = []
        if not self.manager.active:
            self.manager.select('adaptive_focus_fire', posture=self.posture)
        actions, telemetry = self.manager.step(payload, n_agents)
        if len(self.events) != 1 or actions != self.events[0]['final_actions']:
            raise RuntimeError('Local decision was bypassed or silently overwritten')
        self.cached = deepcopy((actions, telemetry))
        self.last_key = key
        self.last_input_hash = input_hash
        self.step_number += 1
        return actions, telemetry, deepcopy(self.events)
