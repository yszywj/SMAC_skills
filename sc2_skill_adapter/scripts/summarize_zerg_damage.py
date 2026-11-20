"""Summarize sampled HP losses for a completed native Zerg tactical matrix."""
import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from zerg_damage_packets import infer_packets
from zerg_damage_replay import read_roster


SIDES = ('ally', 'enemy')
OTHER = {'ally': 'enemy', 'enemy': 'ally'}
MAX_HP = {'Zergling': 35, 'Hydralisk': 90, 'Baneling': 30}
# Only these calibrated zero-upgrade catalogs support the packet model.
CATALOG = {
    'Zergling': '077b51ff8172cf53a8ee812f7f11441f8b7b9303fdfd76df1187b9cbe5a8fdea',
    'Hydralisk': '63304311eb18ed7a365f2e6543da047ebffd6d9ee494b7620d94dda535fd02f2',
    'Baneling': '6cb488ceb0838f8d1348b391a7bdc6dd60ad65e32a39be25e28d3c74a3def0d9',
}


class Episode:
    def __init__(self, directory, number):
        self.roster = read_roster(directory / 'replays' / f'episode-{number:04}.SC2Replay')
        audit = json.loads((directory / 'balance' / f'episode-{number:04}.json').read_bytes())
        if not audit['passed'] or audit['mode'] != 'native':
            raise ValueError('Damage attribution requires a verified native episode')
        for kind, profile in audit['profiles'].items():
            upgrades = profile['shared_attributes']
            if (profile['shared_catalog_sha256'] != CATALOG[kind]
                    or upgrades['attack_upgrade_level'] or upgrades['armor_upgrade_level']
                    or any(hp != MAX_HP[kind] or shield != 0
                           for units in profile['initial_health_shields'].values()
                           for hp, shield in units)):
                raise ValueError(f'Unsupported damage catalog, upgrades or initial HP for {kind}')
        self.stats = {side: Counter() for side in SIDES}
        self.counts = {side: Counter() for side in SIDES}
        self.observations = 0
        self.last = {}
        for tag, unit in self.roster.items():
            side, kind = unit['side'], unit['type']
            self.last[tag] = (unit['born'], MAX_HP[kind])
            self.stats[side]['initial_hp'] += MAX_HP[kind]
            self.counts[side][kind] += 1

    def sample(self, tag, loop, health, dead=False):
        if tag not in self.last:
            return
        prev, hp = self.last[tag]
        if dead:
            del self.last[tag]
        else:
            self.last[tag] = (loop, health)
        loss = hp - health
        if loss <= 0:
            return
        unit = self.roster[tag]
        attacking = OTHER[unit['side']]
        stats = self.stats[attacking]
        killer = self.roster.get(unit['killer']) if dead else None
        hostile_kill = killer is not None and killer['side'] == attacking
        if dead and unit['type'] == 'Baneling' and not hostile_kill:
            stats['excluded_baneling_removal_hp'] += loss
            return
        stats['observed_opponent_hp_loss'] += loss
        attributed = {}
        if 0 < loop - prev <= 8 and (not dead or hostile_kill):
            # Ordinary hits can arrive after their shooter dies. Keep the full
            # initial ordinary roster as a permissive upper bound per window.
            counts = self.counts[attacking].copy()
            counts['Baneling'] = sum(
                u['type'] == 'Baneling' and u['side'] == attacking
                and u['death'] is not None and prev < u['death'] <= loop
                for u in self.roster.values())
            attributed = infer_packets(loss, counts, unit['type'],
                                       killer['type'] if dead else None)
        for kind, amount in attributed.items():
            stats[kind] += amount
            if not dead:
                stats['nonlethal_' + kind] += amount
        stats['unassigned_hp'] += max(0., loss - sum(attributed.values()))

    def observe(self, state):
        self.observations += 1
        for side, key in [('ally', 'allies'), ('enemy', 'visible_enemies')]:
            for u in state[key].values():
                if not u['alive']:
                    continue
                tag = int(u['tag'])
                if (self.roster[tag]['side'], self.roster[tag]['type']) != (side, u['type']):
                    raise ValueError(f'Replay/observation unit mismatch for {tag}')
                self.sample(tag, state['game_loop'], u['health'])

    def finish(self, record):
        if self.observations != record['steps']:
            raise ValueError('Expected one skill observation per decision')
        deaths = Counter()
        for tag, unit in self.roster.items():
            if unit['death'] is not None:
                self.sample(tag, unit['death'], 0., dead=True)
                deaths[unit['side']] += 1
                killer = self.roster.get(unit['killer'])
                if killer and killer['side'] != unit['side']:
                    self.stats[killer['side']]['last_hits_' + killer['type']] += 1
        if (deaths['ally'], deaths['enemy']) != (record['info']['dead_allies'],
                                                 record['info']['dead_enemies']):
            raise ValueError('Replay/episode death count mismatch')
        return self.stats


def rounded(stats):
    return {side: {k: round(v, 2) for k, v in sorted(stats[side].items())} for side in SIDES}


def summarize(batch):
    matrix = json.loads((batch / 'matrix_summary.json').read_bytes())
    if matrix['status'] != 'complete':
        raise ValueError('Expected a completed matrix')
    groups = {}
    total = {side: Counter() for side in SIDES}
    for group in matrix['groups']:
        if not group.startswith('z_'):
            continue
        directory = batch / 'current' / group
        episodes = []
        subtotal = {side: Counter() for side in SIDES}
        with (directory / 'events.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                event = json.loads(line)
                if event['event'] == 'episode_started':
                    episode = Episode(directory, event['episode'])
                elif event['event'] == 'skill_observation':
                    episode.observe(event['payload']['state'])
                elif event['event'] == 'episode_completed':
                    stats = episode.finish(event)
                    episodes.append({'episode': event['episode'], **rounded(stats)})
                    for side in SIDES:
                        subtotal[side].update(stats[side])
        if len(episodes) != matrix['games_per_group']:
            raise ValueError(f'Incomplete observation log for {group}')
        groups[group] = {'total': rounded(subtotal), 'episodes': episodes}
        for side in SIDES:
            total[side].update(subtotal[side])
    if not groups:
        raise ValueError('No Zerg groups in batch')
    report = {
        'method': 'native_zerg_hp_packets_v1',
        'units': 'HP, by attacking side. initial_hp is the starting HP of that side.',
        'interpretation': ('By-type estimates use sampled net losses over at most 8 loops and the minimum '
                           'contribution shared by feasible native damage-packet combinations. Nonlethal '
                           'fields are subsets; last_hits fields count replay-confirmed opposing kills. '
                           'Omitted counters are zero. Unresolved HP stays unassigned. Uncredited/self Baneling '
                           'death-interval HP is excluded. Survivors use last observed HP; healing and '
                           'unobserved final damage are not reconstructed.'),
        'groups': groups, 'total': rounded(total)}
    path = batch / 'zerg_damage_by_type.json'
    with NamedTemporaryFile('w', encoding='utf-8', dir=batch, delete=False) as f:
        json.dump(report, f, separators=(',', ':'), allow_nan=False)
    Path(f.name).replace(path)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    print(summarize(parser.parse_args().batch))
