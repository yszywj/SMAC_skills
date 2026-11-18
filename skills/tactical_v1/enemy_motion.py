"""Short-window visible-enemy motion estimates (pure, read-only queries).

Only legally visible, differently-timed samples of the SAME reliable tag
estimate a velocity. No hidden-enemy tracking, no long extrapolation: the
usable horizon is one control window (1H) with linear prediction. Sudden
turns, stale data or missing speeds are reported as unreliable — the
callers keep their static/conservative result instead of cancelling legal
actions or inventing guaranteed hits.
"""
import math

MAX_SPAN_H = 2.0        # samples older than this many windows are ignored
PREDICT_HORIZON_H = 1.0  # never extrapolate further than one control window
MAX_SAMPLES = 6
JUMP_PER_H = 12.0       # position jump beyond this per window = identity/teleport doubt


def record_observation(records, enemy_id, tag, loop, position, step_mul):
    """Observation-phase sample append; same-clock replaces, never duplicates."""
    step = max(step_mul, 1)
    row = records.get(enemy_id)
    if row is None or row.get('tag') != tag:
        records[enemy_id] = {'tag': tag, 'samples': [(loop, tuple(position))]}
        return
    samples = row['samples']
    if samples and samples[-1][0] == loop:
        samples[-1] = (loop, tuple(position))
        return
    if loop < samples[-1][0]:
        row['samples'] = [(loop, tuple(position))]   # time rollback: restart
        return
    samples.append((loop, tuple(position)))
    del samples[:-MAX_SAMPLES]


def velocity(context, enemy_id):
    """Velocity estimate from visible history, or None when unreliable."""
    records = getattr(context, 'enemy_motion', None) or {}
    row = records.get(enemy_id)
    if not row or not row.get('samples'):
        return None
    enemy = context.snapshot.enemies.get(enemy_id)
    if enemy is None:
        return None
    H = max(context.H, 1)
    now = context.snapshot.game_loop
    samples = [(loop, point) for loop, point in row['samples']
               if now - loop <= MAX_SPAN_H * H]
    samples.append((now, tuple(enemy.position)))
    if len(samples) < 2:
        return None
    first_loop, first = samples[0]
    last_loop, last = samples[-1]
    span = last_loop - first_loop
    if span <= 0:
        return None
    # Reject suspicious jumps: likely identity mismatch or forced reposition.
    for (l1, p1), (l2, p2) in zip(samples, samples[1:]):
        dt = max(l2 - l1, 1)
        if math.dist(p1, p2) / dt * H > JUMP_PER_H:
            return None
    return {'vx': (last[0] - first[0]) / span,
            'vy': (last[1] - first[1]) / span,
            'loops_span': span, 'reliable': True,
            'reason': 'visible_same_tag_linear_estimate'}


def predicted_position(context, enemy_id, at_loop):
    """Enemy position estimate at a future loop within 1H, or None.

    Linear short-window prediction only. ``at_loop`` beyond the horizon,
    missing/unreliable velocity or a vanished enemy degrade to None; the
    caller then keeps its static-position result.
    """
    if not getattr(context, 'burst_config', {}).get('motion_prediction_enabled', True):
        return None
    motion = velocity(context, enemy_id)
    enemy = context.snapshot.enemies.get(enemy_id)
    if motion is None or enemy is None:
        return None
    delta = float(at_loop) - context.snapshot.game_loop
    if delta < 0 or delta > PREDICT_HORIZON_H * max(context.H, 1):
        return None
    return (enemy.position[0] + motion['vx'] * delta,
            enemy.position[1] + motion['vy'] * delta)


def summary(context, limit=3):
    rows = {}
    for enemy_id in list(getattr(context, 'enemy_motion', {}) or {})[:limit]:
        motion = velocity(context, enemy_id)
        if motion is not None:
            rows[enemy_id] = {'speed_per_loop': round(
                math.hypot(motion['vx'], motion['vy']), 3),
                'loops_span': motion['loops_span']}
    return {'enabled': bool(getattr(context, 'burst_config', {}).get(
                'motion_prediction_enabled', True)),
            'estimated': rows,
            'semantics': 'short-window linear estimates; unreliable data '
                         'degrades to static positions'}