"""Evidence-conditioned passive recovery, separate from allocated healing.

No race-specific regeneration rate or delay is guessed. A verified profile
may supply loop-based parameters; otherwise clean observations establish a
conservative rate and delay bound within this episode. Unknown earns zero.
"""
import math


def _number(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


class RecoveryMemory:
    def __init__(self, config=None):
        self.enabled = (config or {}).get('common_combat', {}).get('mechanism_recovery_enabled', True)
        self.previous, self.models = {}, {}
        self.loop = None
        self.healer_present = False

    def __eq__(self, other):
        return isinstance(other, RecoveryMemory) and self.__dict__ == other.__dict__

    def observe(self, context):
        context._recovery_memory = self
        loop = context.snapshot.game_loop
        if not self.enabled or (self.loop is not None and loop <= self.loop):
            return
        self.loop = loop
        healer_present = any(c.healer and context.snapshot.allies[a].alive
                             for a, c in context.capabilities.items())
        healing_interval = healer_present or self.healer_present
        self.healer_present = healer_present
        for actor, unit in context.snapshot.allies.items():
            old = self.previous.get(actor)
            if not unit.alive:
                self.previous.pop(actor, None)
                continue
            if old and old['tag'] != unit.tag:
                old = None
            last_hit = old.get('last_hit') if old else None
            damaged = unit.health_lost + unit.shield_lost > 0
            if old and (unit.health < old['health'] - .01 or unit.shield < old['shield'] - .01):
                damaged = True
            if damaged:
                last_hit = loop
            elapsed = loop - old['loop'] if old else 0
            if old and elapsed > 0 and not damaged:
                for layer in ('health', 'shield'):
                    # Health gains with a healer nearby cannot identify natural
                    # regeneration. Do not learn healing as a passive mechanism.
                    if layer == 'health' and healing_interval:
                        continue
                    amount = getattr(unit, layer) - old[layer]
                    if amount <= .01:
                        continue
                    key = (unit.unit_type_id, layer)
                    model = self.models.get(key)
                    rate = amount / elapsed
                    onset = not old.get('growing_' + layer, False)
                    delay = max(0, loop - last_hit) if last_hit is not None and onset else None
                    if model is None:
                        model = {'rate_per_loop': rate, 'delay_loops': delay,
                                 'samples': 0, 'source': 'observed_clean_intervals'}
                    model['rate_per_loop'] = min(model['rate_per_loop'], rate)
                    if delay is not None:
                        model['delay_loops'] = max(model.get('delay_loops') or 0, delay)
                    model['samples'] += 1
                    self.models[key] = model
            self.previous[actor] = {'tag': unit.tag, 'loop': loop,
                'health': unit.health, 'shield': unit.shield, 'last_hit': last_hit,
                **{'growing_' + layer: bool(old and not damaged and
                    getattr(unit, layer) > old[layer] + .01) for layer in ('health', 'shield')}}

    def opportunity(self, context, actor, *, position=None, horizon_loops=None):
        """Bounded prospective regain, never proof of a future recovery tick."""
        empty = {'gain': 0., 'health': 0., 'shield': 0., 'known': False, 'sources': []}
        if not self.enabled:
            return empty
        unit = context.snapshot.allies[actor]
        point = unit.position if position is None else position
        risk = context.forecast_risk(actor, point)
        if risk.current_threats or risk.effect_weapon_near or unit.health_lost + unit.shield_lost > 0:
            return empty
        H = max(context.H, 1)
        horizon = min(2 * H, max(0, horizon_loops if horizon_loops is not None else H))
        profile = context.snapshot.raw_state.get('unit_type_profiles', {}).get(str(unit.unit_type_id), {})
        declared = profile.get('passive_recovery', {})
        previous = self.previous.get(actor, {})
        result = dict(empty, sources=[])
        for layer in ('health', 'shield'):
            model = declared.get(layer, {})
            if not model.get('verified', False):
                model = self.models.get((unit.unit_type_id, layer), {})
                if model.get('samples', 0) < 2:
                    continue
            rate, delay = model.get('rate_per_loop'), model.get('delay_loops')
            if not _number(rate) or not _number(delay):
                continue
            last_hit = previous.get('last_hit')
            if last_hit is None and delay > 0 and not previous.get('growing_' + layer):
                continue
            quiet = context.snapshot.game_loop - last_hit if last_hit is not None else delay
            remaining_delay = max(0, delay - quiet)
            missing = max(0., getattr(unit, 'max_' + layer) - getattr(unit, layer))
            gain = min(missing, rate * max(0, horizon - remaining_delay))
            # Reachable enemies make an uninterrupted recovery less credible.
            if risk.reachable_threats:
                gain *= .5
            result[layer] = gain
            result['known'] = True
            result['sources'].append(model.get('source', 'verified_profile'))
        result['gain'] = result['health'] + result['shield']
        return result


def recovery_opportunity(context, actor, **kwargs):
    memory = getattr(context, '_recovery_memory', None)
    if memory is None:
        memory = RecoveryMemory()
    return memory.opportunity(context, actor, **kwargs)
