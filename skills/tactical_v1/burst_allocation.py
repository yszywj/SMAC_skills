"""Share target HP between gun and explosion opportunities by timing bucket.

This only allocates utility: uncertain explosions never become known damage
or a kill claim. Same-window/unknown arrivals share a bucket proportionally;
there is no invented exact ordering of missiles or death-triggered effects.
"""
import math

from .targeting import DAMAGE_SCALE


def allocate_exchange(context, actions, report):
    buckets = {}
    H = max(context.H, 1)
    for enemy, actors in report.window_sources.items():
        for actor in actors:
            estimate = context.fire_estimate(actor, enemy)
            if estimate.damage_est is None or estimate.damage_est <= 0.:
                continue
            delay = estimate.effective_after_loops if estimate.delay_known else None
            bucket = max(1, math.ceil(delay / H)) if delay is not None else 1
            row = buckets.setdefault(enemy, {}).setdefault(bucket, [0., 0.])
            row[0] += estimate.damage_est
    for outcome in report.burst_outcomes.values():
        if outcome.opportunity <= 0.:
            continue
        bucket = max(1, math.ceil(outcome.trigger_loops / H))
        for enemy, damage in outcome.damage_potential.items():
            if damage is None or damage <= 0.:
                continue
            row = buckets.setdefault(enemy, {}).setdefault(bucket, [0., 0.])
            row[1] += damage * outcome.opportunity
    gun = burst = 0.
    for enemy, windows in buckets.items():
        target = context.snapshot.enemies.get(enemy)
        if target is None:
            continue
        remaining = max(0., target.effective_health)
        for _, (ordinary, explosion) in sorted(windows.items()):
            total = ordinary + explosion
            if total <= 0.:
                continue
            useful = min(remaining, total)
            # Same fixed reference pool as candidate allocation and the
            # final team scoring — the mixed gun/burst settlement must not
            # keep a small-health-pool bias the other two paths dropped.
            gun += useful * ordinary / total / DAMAGE_SCALE
            burst += useful * explosion / total / DAMAGE_SCALE
            remaining -= useful
    return gun, burst
