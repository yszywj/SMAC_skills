"""Post-contact Zerg support relay. Construct once per episode, call once per macro step.

Uses visible positions, native weapon geometry, actual masks and projected legal
move endpoints. A legal target in weapon range is contact, not proof of a hit.
An empty native menu disables submission (opening or runtime veto). The bridge
may supply relay_protected_actors on the payload for current R6 execution duties.
"""
from math import dist
from numbers import Integral


def _legal(actions, masks):
    return len(actions) == len(masks) and all(
        isinstance(a, Integral) and 0 <= a < len(row) and row[a]
        for a, row in zip(actions, masks))


class CombatRelayR8:
    def __init__(self, mode):
        if mode not in ('menu', 'full'):
            raise ValueError('expected menu or full')
        self.mode = mode
        self.contact = self.loss = self.last_loop = None
        self.live = {}
        self.previous = set()
        self.focus = self.last_gap = self.stop = None
        self.beats = 0

    def propose(self, payload, masks, r6_actions, candidates):
        if not _legal(r6_actions, masks):
            raise ValueError('illegal or incomplete R6 vector')
        base = list(map(int, r6_actions))
        s = payload['state']
        loop = s['game_loop']
        contiguous = self.last_loop is None or loop == self.last_loop + s['step_mul']
        if self.last_loop is not None and loop <= self.last_loop:
            raise ValueError('expected increasing episode game loops')
        self.last_loop = loop
        allies = s['allies']
        live = {i: allies[f'u{i}'] for i in range(len(base)) if allies[f'u{i}']['alive']}
        pos = {i: u['position'] for i, u in live.items()}
        enemies = {e: u for e, u in s['visible_enemies'].items() if u['alive']}
        # A helper joins a real firing lane, not a map waypoint or a fog target.
        front = {e for e in enemies for i in live if masks[i][6 + int(e[1:])]
                 and s['relations'][f'u{i}'].get(e, {}).get('in_weapon_range') is True}
        damaged = any(u.get('health_lost_since_previous_decision', 0) > 0 or
                      u.get('shield_lost_since_previous_decision', 0) > 0 for u in live.values())
        if self.contact is None and (front or damaged):
            self.contact = loop
        if self.loss is None and self.contact is not None and any(
                self.live[i] != 'baneling' for i in self.live.keys() - live.keys()):
            self.loss = loop
        self.live = {i: u['type'].lower() for i, u in live.items()}
        protected = set(payload.get('relay_protected_actors', ()))
        locked = {i for i in live if any(masks[i][6:]) or f'u{i}' in protected}
        gaps = {}
        if self.loss is not None:
            for i, u in live.items():
                if any(masks[i][6:]) or self.live[i] == 'baneling' or u['health'] < .25 * u['max_health']:
                    continue
                for e in front:
                    rel = s['relations'][f'u{i}'].get(e, {})
                    if rel.get('native_weapon_range') is None or rel.get('weapon_range_status') == 'no_compatible_weapon':
                        continue
                    gap = dist(pos[i], enemies[e]['position'])
                    if gap <= 12:
                        gaps[i, e] = gap
        eligible = sorted(self.previous.intersection(gaps)) if contiguous else []
        self.previous = set(gaps)
        audit = dict(contact_loop=self.contact, first_nonbane_loss_loop=self.loss,
                     witnesses=[[i, e] for i, e in eligible], protected=sorted(locked))

        def result(reason, actions=None, end=False):
            if end:
                self.stop = reason
            audit.update(focus=list(self.focus) if self.focus else None, relay_steps=self.beats)
            return dict(actions=base if actions is None else actions, reason=reason,
                        phase='r6' if actions is None else 'relay', audit=audit)

        if self.stop or self.beats >= 4:
            return result(self.stop or 'step_limit', end=True)
        if self.focus is not None:
            i, e = self.focus
            if e not in enemies or e not in front:
                return result('target_absent_or_front_lost', end=True)
            if i in live and any(masks[i][6:]):
                return result('support_restored', end=True)
            if self.focus not in eligible or gaps[self.focus] >= self.last_gap - .05:
                return result('no_observed_progress', end=True)
            eligible = [self.focus]
        if not eligible or not candidates:
            return result('no_persistent_witness' if not eligible else 'submission_veto', end=bool(self.beats))

        # Only a legal closer endpoint can beat both the current position and
        # the frozen R6 prediction. B scores intact vectors; C may introduce a
        # single targeted move in an otherwise untouched complete R6 vector.
        menu = {k: list(map(int, a)) for k, a in candidates.items() if _legal(a, masks)}
        if not menu:
            return result('no_legal_menu', end=bool(self.beats))
        best = None
        for i, e in eligible:
            if i in locked:
                continue
            target = enemies[e]['position']
            moves = {int(a): m['destination'] for a, m in s['movement_candidates'][f'u{i}'].items()
                     if 2 <= int(a) <= 5 and masks[i][int(a)]}
            old = min(gaps[i, e], dist(moves.get(base[i], pos[i]), target))
            options = list(menu.items())
            if self.mode == 'full':
                options.extend((None, base[:i] + [a] + base[i+1:]) for a in moves)
            for label, vector in options:
                if vector[i] not in moves or any(vector[j] != base[j] for j in locked):
                    continue
                gain = old - dist(moves[vector[i]], target)
                if gain < .25:
                    continue
                rank = (-gain, sum(a != b for a, b in zip(vector, base)), gaps[i, e], i, e,
                        tuple(vector), label is None, label or '')
                if best is None or rank < best[0]:
                    best = rank, (i, e), vector, label
        if best is None:
            return result('no_improving_command', end=bool(self.beats))
        rank, self.focus, actions, label = best
        self.last_gap = gaps[self.focus]
        self.beats += 1
        audit.update(native_label=label, outside_menu=actions not in menu.values(),
                     predicted_gain=-rank[0], observed_distance=self.last_gap)
        return result('close_disconnected_support', list(actions))
