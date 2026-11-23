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
    """Velocity estimate from visible tagged history, or a failure reason.

    Read-only: this query NEVER appends to the persistent records. In the
    normal manager path the observation phase has already recorded the
    current position at the current game_loop, so the samples are used as
    stored. Only for direct calls whose records do NOT yet contain the
    current observation does a LOCAL copy gain one current sample
    (last.sample < now); a sample already stamped at ``now`` is REPLACED in
    the local copy, never appended again; a sample in the future fails as
    invalid_time. Records are never mutated.
    """
    records = getattr(context, 'enemy_motion', None) or {}
    row = records.get(enemy_id)
    if not row or not row.get('samples'):
        return {'reliable': False, 'reason': 'insufficient_samples'}
    enemy = context.snapshot.enemies.get(enemy_id)
    if enemy is None:
        return {'reliable': False, 'reason': 'missing_identity'}
    current_tag = context.snapshot.raw_state.get('visible_enemies', {}) \
        .get(enemy_id, {}).get('tag')
    if not current_tag:
        return {'reliable': False, 'reason': 'missing_identity'}
    if current_tag != row.get('tag'):
        return {'reliable': False, 'reason': 'identity_changed'}
    H = max(context.H, 1)
    now = context.snapshot.game_loop
    # Local copies only — the persistent record list is never modified.
    samples = [(loop, tuple(point)) for loop, point in row['samples']
               if now - loop <= MAX_SPAN_H * H]
    last_loop, last_point = (samples[-1] if samples else (None, None))
    if last_loop is None or last_loop > now:
        return {'reliable': False, 'reason': 'invalid_time'}
    if last_loop < now:
        samples.append((now, tuple(enemy.position)))     # once, local copy
    elif last_loop == now:
        samples[-1] = (now, tuple(enemy.position))       # replace, no append
    if any(loop > now for loop, _ in samples):
        return {'reliable': False, 'reason': 'invalid_time'}
    dedup = {}
    for loop, point in samples:
        dedup[loop] = point              # at most one sample per game_loop
    samples = sorted(dedup.items())
    if len(samples) < 2 or samples[-1][0] == samples[0][0]:
        return {'reliable': False, 'reason': 'insufficient_samples'}
    if any(not (math.isfinite(loop) and math.isfinite(point[0])
                and math.isfinite(point[1]))
           for loop, point in samples):
        return {'reliable': False, 'reason': 'invalid_position'}
    first_loop, first = samples[0]
    last_loop, last = samples[-1]
    span = last_loop - first_loop
    if span <= 0:
        return {'reliable': False, 'reason': 'invalid_time'}
    segments = []
    for (loop_a, point_a), (loop_b, point_b) in zip(samples, samples[1:]):
        dt = loop_b - loop_a
        if dt <= 0:
            return {'reliable': False, 'reason': 'invalid_time'}
        if math.dist(point_a, point_b) / dt * H > JUMP_PER_H:
            return {'reliable': False, 'reason': 'motion_unreliable'}
        segments.append(((point_b[0] - point_a[0]) / dt,
                         (point_b[1] - point_a[1]) / dt))
    mean = ((last[0] - first[0]) / span, (last[1] - first[1]) / span)
    mean_norm = math.hypot(*mean)
    if mean_norm > 1e-9:
        # Clear reversal inside the window: the freshest segment opposes the
        # mean direction with comparable magnitude — averaged motion would
        # be fiction, so degrade instead.
        latest = segments[-1]
        latest_norm = math.hypot(*latest)
        dot = (mean[0] * latest[0] + mean[1] * latest[1]) \
            / (mean_norm * max(latest_norm, 1e-9))
        if latest_norm > .5 * mean_norm and dot < -.25:
            return {'reliable': False, 'reason': 'motion_unreliable'}
    return {'vx': mean[0], 'vy': mean[1],
            'loops_span': span, 'reliable': True,
            'reason': 'valid_short_motion'}


def predicted_position(context, enemy_id, at_loop):
    """Enemy position estimate at a future loop within 1H, or None.

    Linear short-window prediction only. ``at_loop`` beyond the horizon,
    missing/unreliable velocity or a vanished enemy degrade to None; the
    caller then keeps its static-position result. velocity() failures carry
    a diagnosis reason (insufficient_samples / missing_identity /
    identity_changed / invalid_time / invalid_position / motion_unreliable)
    retrievable via velocity(context, enemy_id)['reason'].
    """
    if not getattr(context, 'burst_config', {}).get('motion_prediction_enabled', True):
        return None
    motion = velocity(context, enemy_id)
    if not motion.get('reliable'):
        return None
    enemy = context.snapshot.enemies.get(enemy_id)
    if enemy is None:
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
        if motion is not None and motion.get('reliable'):
            rows[enemy_id] = {'speed_per_loop': round(
                math.hypot(motion['vx'], motion['vy']), 3),
                'loops_span': motion['loops_span']}
    return {'enabled': bool(getattr(context, 'burst_config', {}).get(
                'motion_prediction_enabled', True)),
            'estimated': rows,
            'semantics': 'short-window linear estimates; unreliable data '
                         'degrades to static positions'}