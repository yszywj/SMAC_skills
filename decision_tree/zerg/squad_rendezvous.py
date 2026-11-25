"""Bounded, opening-only SMAC squad movement. Use one instance per episode."""
from math import dist
from operator import index


def _squads(pos):
    # Partition once. Start at the periphery so isolated actors get partners,
    # rather than creating a singleton to chase the entire team from behind.
    remaining = set(pos)
    groups = []
    while remaining:
        mid = tuple(sum(pos[i][k] for i in remaining) / len(remaining) for k in (0, 1))
        seed = max(remaining, key=lambda i: (dist(pos[i], mid), -i))
        count = 3 if len(remaining) > 4 else len(remaining)
        group = sorted(remaining, key=lambda i: (dist(pos[i], pos[seed]), i))[:count]
        groups.append(group)
        remaining.difference_update(group)
    return groups


class SquadRendezvous:
    """Modify only legal moves; handoff returns the supplied v2 vector immediately."""

    def __init__(self, mode='adaptive'):
        if mode not in ('adaptive', 'fixed'):
            raise ValueError('mode must be adaptive or fixed')
        self.mode = mode
        self.beats = 0
        self.groups = None
        self.exit_reason = None

    def propose(self, payload, masks, base_actions):
        base = [index(a) for a in base_actions]
        if len(base) != len(masks) or any(not 0 <= a < len(row) or not row[a]
                                          for row, a in zip(masks, base)):
            raise ValueError('expected a legal v2 action for each actor')

        def finish(reason):
            self.exit_reason = reason
            return dict(actions=base, phase='v2', handoff=True, reason=reason)

        if self.exit_reason:
            return finish(self.exit_reason)
        if self.beats >= (4 if self.mode == 'fixed' else 8):
            return finish('timeout')
        state = payload['state']
        units = [state['allies'][f'u{i}'] for i in range(len(base))]
        pos = {i: u['position'] for i, u in enumerate(units) if u['alive']}
        if len(pos) < len(units) or any(u.get('health_lost_since_previous_decision', 0) > 0 or
                                       u.get('shield_lost_since_previous_decision', 0) > 0 for u in units):
            return finish('casualty_or_damage')
        if any(sum(bool(row[a]) for row in masks) >= 2 for a in range(6, len(masks[0]))):
            return finish('shared_target')
        if any(dist(p, e['position']) <= 2.5 for e in state['visible_enemies'].values()
               if e['alive'] for p in pos.values()):
            return finish('close_enemy')
        if self.groups is None:
            if max((dist(p, q) for p in pos.values() for q in pos.values()), default=0) < 12:
                return finish('compact_start')
            self.groups = _squads(pos)

        actions = base.copy()
        needs_move = False
        for group in self.groups:
            goal = tuple(sum(pos[i][k] for i in group) / len(group) for k in (0, 1))
            for i in group:
                gap = dist(pos[i], goal)
                if gap <= 2.5 or any(masks[i][6:]):
                    continue
                moves = state['movement_candidates'][f'u{i}']
                options = [(dist(moves[str(a)]['destination'], goal), a != base[i], a)
                           for a in range(2, 6) if masks[i][a]]
                if options:
                    distance, _, a = min(options)
                    if distance < gap - .25:
                        actions[i] = a
                        needs_move = True
        if not needs_move and self.mode == 'adaptive':
            return finish('formed_or_blocked')
        self.beats += 1
        return dict(actions=actions, phase='rendezvous', handoff=False,
                    reason='local_approach' if needs_move else 'fixed_window', squads=self.groups)
