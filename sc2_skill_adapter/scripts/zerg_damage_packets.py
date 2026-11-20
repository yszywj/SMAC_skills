"""Conservative packet-model attribution for short native Zerg battle intervals."""

from itertools import product
from math import floor, isfinite
from numbers import Integral, Real


_TYPES = ('Zergling', 'Hydralisk', 'Baneling')
_REGEN = 0.4  # upper allowance for one observed interval of at most eight loops
_ROUNDING = 0.01  # difference of two HP readings rounded to hundredths


def infer_packets(loss, counts, victim_type, killer_type=None):
    """Return the shared lower bounds on effective HP over feasible hit mixtures.

    Counts are upper bounds on incoming packets, including the final hit. For
    a survivor, a packet sum D is feasible when D - net loss fits regeneration
    and display rounding. Each type gets its share of the observed net loss.
    For a death, loss is the last observed HP. Earlier hits must leave positive
    HP and the confirmed killer must finish it. Only its effective final damage
    is credited. Across possible mixtures we retain each type's minimum, so
    ambiguous HP need not be assigned. These are conditional model quantities.

    The caller supplies consecutive observations (<=8 loops), guaranteed
    zero armor/upgrades/shields, and skips deaths without an opposing killer.
    """
    if victim_type not in _TYPES or (killer_type is not None and killer_type not in _TYPES):
        return {}
    if not isinstance(loss, Real) or not isfinite(loss) or loss <= 0.:
        return {}
    if any(t not in _TYPES or not isinstance(n, Integral) or n < 0 for t, n in counts.items()):
        return {}

    packets = (5., 12., 16. if victim_type == 'Baneling' else 35.)
    maxima = [counts.get(t, 0) for t in _TYPES]
    killer = _TYPES.index(killer_type) if killer_type is not None else None
    if killer is not None:
        if not maxima[killer]:
            return {}
        maxima[killer] -= 1  # enumerate full hits preceding the last one

    upper_hp = loss + _REGEN + _ROUNDING
    ranges = [range(1 + min(n, floor(upper_hp / d))) for n, d in zip(maxima, packets)]
    minima = None
    for hits in product(*ranges):
        damage = [n * d for n, d in zip(hits, packets)]
        full = sum(damage)
        if killer is None:
            if full <= 0. or not loss - _ROUNDING <= full <= upper_hp:
                continue
            share = min(1., loss / full)
            credit = [d * share for d in damage]
        else:
            available = full + packets[killer]
            high = min(upper_hp, available)
            if high <= full or available < loss - _ROUNDING:
                continue
            # Effective HP removed, including regeneration, ranges from
            # low to high. The last hit consumes the residual beyond full.
            # After normalization to the observed loss, earlier types have
            # least credit at high, the killer at low. A prior sum equal to
            # low can leave arbitrarily little HP for the actual last hit.
            low = min(available, max(loss, full))
            credit = [d * min(1., loss / high) for d in damage]
            credit[killer] = (damage[killer] + low - full) * min(1., loss / low)
        minima = credit if minima is None else [min(a, b) for a, b in zip(minima, credit)]
        if not any(minima):
            return {}

    return {} if minima is None else {t: d for t, d in zip(_TYPES, minima) if d > 0.}
