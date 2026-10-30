"""Opt-in adapter action: leave an observed native Medivac heal order alone.

This is not a native SC2 command, a new heal target, or proof of a heal tick.
The raw environment mask is authoritative; historical semantic snapshots have
rounded fields and are useful only for explicitly labelled offline checks.
"""
import copy
import math

from .semantics import KEEP_HEAL_ORDER_DESCRIPTION


def _enough_energy(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 1


def raw_keep_heal_target(agents, agent_id):
    """Return an allied index from the first actually observed heal order."""
    healer = agents[agent_id]
    if (healer.unit_type != 54 or healer.health <= 0
            or not _enough_energy(healer.energy) or not healer.orders
            or healer.orders[0].ability_id != 386):
        return None
    tag = healer.orders[0].target_unit_tag
    for index, patient in agents.items():
        if (patient.tag == tag and patient.unit_type != 54
                and 0 < patient.health < patient.health_max):
            return index
    return None


def semantic_keep_heal_target(state, actor):
    """Historical counterpart, using only the supplied semantic snapshot."""
    healer = state['allies'][actor]
    orders = healer.get('orders', [])
    if (healer.get('unit_type_id') != 54 or not healer.get('alive')
            or not _enough_energy(healer.get('energy')) or not orders
            or orders[0].get('ability_id') != 386):
        return None
    target = orders[0].get('target_unit')
    patient = state['allies'].get(target, {})
    if (patient.get('alive') and patient.get('unit_type_id') != 54
            and 0 < patient.get('health', 0) < patient.get('max_health', 0)):
        return target
    return None


def action_extension_metadata(action_id):
    return {'terran_keep_heal_order': {
        'enabled': True, 'version': 'terran-keep-heal-order-v1',
        'action_id': action_id, 'native_n_actions': action_id, 'n_actions': action_id + 1,
        'submission': 'retained_order', 'native_command': False}}


def validate_keep_heal_order(args, method, config):
    if not getattr(args, 'terran_keep_heal_order', False):
        return None
    if (method != 'adaptive-skill' or getattr(args, 'skill_version', None) != 'current'
            or getattr(args, 'race_profile', None) != 'terran'
            or config.get('map_name') != '10gen_terran' or config.get('unit_balance') != 'native'):
        raise ValueError('--terran-keep-heal-order requires adaptive-skill/current/terran/native')
    return action_extension_metadata(6 + config['capability_config']['n_enemies'])


def extend_keep_heal_order_request(payload, action_id, candidates=None):
    """Return an extended copy. Explicit raw candidates take priority in play."""
    result = copy.deepcopy(payload)
    state = result['state']
    if candidates is None:
        candidates = {
            actor: {'action_id': action_id, 'target': target}
            for actor in result['questions']
            if (target := semantic_keep_heal_target(state, actor)) is not None}
    state['action_contract']['keep_heal_order'] = {'enabled': True, 'action_id': action_id}
    state['action_contract']['actions'] += f'; {action_id} KeepHealOrder submits no new command.'
    state['keep_heal_order_candidates'] = copy.deepcopy(candidates)
    for actor, candidate in candidates.items():
        if candidate['action_id'] != action_id or actor not in result['questions']:
            raise ValueError('Invalid keep-heal-order candidate')
        result['questions'][actor]['criteria'][str(action_id)] = (
            KEEP_HEAL_ORDER_DESCRIPTION + f" Existing target: {candidate['target']}.")
    return result
