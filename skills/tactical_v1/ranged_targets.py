"""Current-window return fire alternatives for a local ranged formation."""
from .hydra_support import _can_fire_now, _enemy_covers, _threats_of


def support_attack(context, supporter, target, protected, incumbent_actions):
    """Choose whether to offer B's shot on E alongside the incumbent orders.

    The formation caller supplies E from its current or projected fire lines.
    Retasking a gun already shooting elsewhere requires observed coverage of A.
    """
    if not _can_fire_now(context, supporter, target):
        return None
    shot = context.attack_action(supporter, target)
    order = incumbent_actions.get(supporter)
    if order is None:
        return shot
    original = context.decode(supporter, order)
    if original.kind != 'attack' or original.target == target:
        return shot
    old = original.target
    if old not in context.snapshot.enemies or not _can_fire_now(context, supporter, old):
        return shot
    threats = _threats_of(context, protected)
    if target not in threats or old in threats:
        return None
    front = context.snapshot.allies[protected]
    if len(threats) < 2 and front.health_lost + front.shield_lost <= 0.:
        return None
    if context.enemy_capabilities[old].suicide_attack:
        return None

    # Preserve a finishing volley only when this particular source matters.
    ally_damage = 0.
    for peer, action in incumbent_actions.items():
        unit = context.snapshot.allies.get(peer)
        if peer == supporter or unit is None or not unit.alive:
            continue
        if context.capabilities[peer].healer or context.capabilities[peer].suicide_attack:
            continue
        other = context.decode(peer, action)
        if other.kind == 'attack' and other.target == old and _can_fire_now(context, peer, old):
            ally_damage += context.fire_estimate(peer, old).damage_lo or 0.
    old_hp = context.snapshot.enemies[old].effective_health
    damage = context.fire_estimate(supporter, old).damage_lo or 0.
    if ally_damage < old_hp <= ally_damage + damage:
        return None
    for peer, unit in context.snapshot.allies.items():
        if peer == protected or not unit.alive or unit.type_name != 'hydralisk':
            continue
        if _enemy_covers(context, peer, old) and (
                unit.health_lost + unit.shield_lost > 0. or len(_threats_of(context, peer)) >= 2):
            return None
    return shot
