"""Sticky online router. Inputs at the anchor are public and pre-action only."""
import hashlib
import itertools
import json
import math
from pathlib import Path

from decision_tree.scripts.linux.fit_swarm_explorer import CANDIDATES, use
from decision_tree.zerg.policy_route_gate import anchor_features, valid_features
from decision_tree.zerg.radical_swarm import RadicalSwarm


FROZEN_GATE = '6d3a2ad32e0f9543c9d1ed9734de681592b887c874c3cb4dc02d5dbe0007fb2f'


def features(payload, parent, other):
    values = anchor_features(payload, {'actions': parent}, {'actions': other})
    if not valid_features(values):
        raise ValueError('invalid live macro features')
    state = payload['state']
    units = [u for u in state['allies'].values() if u['alive']]
    guns = [u for u in units if u['type'] == 'hydralisk']
    escorts = [u for u in units if u['type'] != 'hydralisk']
    def gap(a, b):
        return math.dist(a['position'], b['position'])
    values.update(spread=max((gap(a, b) for a, b in itertools.combinations(units, 2)), default=0),
                  escort_gap=sum(min(gap(a, b) for b in escorts) for a in guns)/len(guns) if escorts and guns else -1,
                  seen=len(state['visible_enemies']), banes=sum(u['type'] == 'baneling' for u in units),
                  n_lings=sum(u['type'] == 'zergling' for u in units))
    if values.keys() != CANDIDATES.keys() or not all(map(math.isfinite, values.values())):
        raise ValueError('incomplete explorer features')
    return values


class OnlineRoute:
    def __init__(self, environ, parent, controller_sha):
        raw = Path(environ['ZERG_SWARM_GATE']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != FROZEN_GATE or environ['ZERG_SWARM_GATE_SHA256'] != FROZEN_GATE:
            raise ValueError('not the frozen two-rule explorer gate')
        gate = json.loads(raw)
        hashes = {k: parent[k] for k in ('execution_context_hash', 'selector_sha256', 'model_sha256', 'opening_sha256')}
        if (gate['schema'] != 'swarm_first_macro_exploration_v1' or parent['policy_version'] != 'v2'
                or gate['default'] != 'v2' or gate['when_any_conjunction_matches'] != 'RadicalSwarm'
                or gate['policy_hashes'] != dict(hashes, radical_swarm=controller_sha)
                or controller_sha != environ['ZERG_SWARM_SHA256']):
            raise ValueError('inconsistent frozen route policies')
        self.rules = gate['rules']
        self.contract = dict(kind='swarm_first_macro_online_v1', gate_sha256=FROZEN_GATE,
                             controller_sha256=controller_sha, trial_id=parent['trial_id'])

    def reset(self):
        self.choice = None
        self.swarm = RadicalSwarm()

    def decide(self, payload, masks, parent):
        if (parent is None) != (self.choice == 'swarm'):
            raise RuntimeError('missing or unexpected live v2 proposal')
        swarm = None if self.choice == 'v2' else self.swarm.choose(payload, masks)
        other = swarm['actions'] if swarm else None
        anchor = None
        reason = 'locked'
        if self.choice is None:
            reason = 'common_macro'
            if parent != other:
                anchor = features(payload, parent, other)
                self.choice = 'swarm' if use(self.rules, {'features': anchor}) else 'v2'
                reason = 'first_macro_difference'
        return dict(choice=self.choice, reason=reason, v2_actions=parent, swarm_actions=other,
                    actions=other if self.choice == 'swarm' else parent, features=anchor,
                    avail_actions=masks, swarm_decision=swarm)

    def after_engine(self, submissions):
        if submissions and self.choice == 'swarm':
            raise RuntimeError('unexpected Swarm interior command')
        if submissions and self.choice is None:
            self.choice = 'v2'
            return True
        return False
