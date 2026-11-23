"""Offline Zerg first-dog-mask forecasts. No actions, SC2, or replay decoder.

Run with no arguments for the 150 baseline / earlier 40 contact_response games.
Use --json PATH for every pre-contact update and the observed per-dog labels.
"""
import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

if __package__:
    from .contact_front_report import COMPOSITIONS, _check, _digest, _first_contact, _layout
else:
    from contact_front_report import COMPOSITIONS, _check, _digest, _first_contact, _layout

RUNS = Path(__file__).resolve().parents[2] / 'runs'
DEV = RUNS / 'zerg_early_contact_20261121_002523_932869/baseline'
TEST = RUNS / 'contact_540_20261120_225957_302289/contact_response'
RADIUS = 3.0  # local neighborhood in _first_contact; also our fixed point tolerance
MODELS = ('nearest', 'kinematic')


@dataclass(frozen=True)
class Unit:
    kind: str
    pos: tuple
    speed: float  # native game units per loop


@dataclass(frozen=True)
class Frame:
    loop: int
    stride: int
    units: dict
    dog_range: float
    gun_range: float

    @classmethod
    def observe(cls, s):
        units = {}
        for side in ('allies', 'visible_enemies'):
            for key, u in s[side].items():
                if u['alive']:
                    speed = s['unit_type_profiles'][str(u['unit_type_id'])]['movement_speed_native']
                    units[key] = Unit(u['type'], tuple(u['position']), speed / 16)
        ranges = s['action_contract']['candidate_center_distance_by_type']
        return cls(s['game_loop'], s['step_mul'], units, ranges['zergling'], ranges['hydralisk'])


def _centroid(points):
    return [mean(axis) for axis in zip(*points)]


def _toward(p, goal, amount):
    ratio = min(1, amount / max(1e-9, math.dist(p, goal)))
    return tuple(a + ratio * (b - a) for a, b in zip(p, goal))


def forecasts(f, prev=None):
    """Pure functions of whitelisted visible fields. No label, actions or run metadata."""
    units = f.units
    allies = sorted(k for k in units if k.startswith('u'))
    enemies = sorted(k for k in units if k.startswith('e'))
    dogs = [k for k in allies if units[k].kind == 'Zergling']
    guns = [k for k in allies if units[k].kind == 'Hydralisk']
    if not dogs or not enemies:
        return dict.fromkeys(MODELS)  # no stale enemy locations when vision is lost
    p = {k: u.pos for k, u in units.items()}

    def make(pos, dog, targets, method, horizon):
        cluster = [e for e in enemies if any(math.dist(pos[t], pos[e]) <= RADIUS for t in targets)]
        support = [h for h in guns if any(math.dist(pos[h], pos[e]) <= f.gun_range for e in cluster)]
        return dict(dog=dog, dog_position=pos[dog], targets=sorted(targets), cluster=cluster,
                    point=_centroid(pos[e] for e in cluster), support=support, method=method,
                    projected_loops=horizon)

    # Stationary enemy/hydras; advance just the nearest dog to its mask edge.
    d, dog, target = min((math.dist(p[a], p[e]), a, e) for a in dogs for e in enemies)
    pos = dict(p)
    pos[dog] = _toward(p[dog], p[target], max(0, d - f.dog_range))
    targets = [e for e in enemies if math.dist(pos[dog], pos[e]) <= f.dog_range + 1e-9]
    nearest = make(pos, dog, targets, 'nearest', None)

    # Sample on the decision clock, not at continuous ray crossings. Bound straight-
    # line persistence to four decisions. Cold start: head toward nearest visible
    # opponent at catalog speed, ranged units stopping at their mask edge.
    paths = {}
    for k, u in units.items():
        if prev and k in prev.units:
            dt = f.loop - prev.loop
            v = tuple((a-b)/dt for a, b in zip(p[k], prev.units[k].pos))
            paths[k] = lambda t, x=p[k], v=v: tuple(a+b*t for a, b in zip(x, v))
        else:
            foe = min(enemies if k in allies else allies, key=lambda e: (math.dist(p[k], p[e]), e))
            stop = f.gun_range if u.kind == 'Hydralisk' else 0
            gap = max(0, math.dist(p[k], p[foe]) - stop)
            paths[k] = lambda t, x=p[k], y=p[foe], gap=gap, v=u.speed: _toward(x, y, min(v*t, gap))
    projected = dict(nearest, method='nearest_fallback')
    for i in range(1, 5):
        t = i * f.stride
        pos = {k: path(t) for k, path in paths.items()}
        for a in sorted(dogs, key=lambda k: int(k[1:])):
            targets = [e for e in enemies if math.dist(pos[a], pos[e]) <= f.dog_range]
            if targets:
                projected = make(pos, a, targets, 'kinematic', t)
                break
        else:
            continue
        break
    return dict(nearest=nearest, kinematic=projected)


def _labels(payload, first, where):
    s = payload['state']
    result = {}
    for dog in first['tied_dogs']:
        label = first
        if dog != first['dog']:
            # Select this tied dog in the unchanged observed geometry. Removing
            # other dogs cannot affect its legal targets or *allied hydra* cover.
            allies = {k: u for k, u in s['allies'].items() if k == dog or u['type'] != 'Zergling'}
            label = _first_contact(dict(payload, state=dict(s, allies=allies)), where)
        cluster = {e: s['visible_enemies'][e]['position'] for e in label['local_enemies']}
        cover = label['ally_support_by_hydra']
        result[dog] = dict(dog_position=label['dog_position'], targets=label['legal_target_positions'],
                           cluster=cluster, point=_centroid(cluster.values()), support_by_hydra=cover,
                           support=None if None in cover.values() else sorted(h for h in cover if cover[h]))
    return dict(loop=first['loop'], canonical=first['dog'], dogs=result)


class Episode:
    def __init__(self):
        self.prev = self.first_vision = self.label = None
        self.updates = []

    def observe(self, payload, where):
        if self.label is not None:
            return
        f = Frame.observe(payload['state'])
        _check(not self.prev or f.loop == self.prev.loop + self.prev.stride, where, 'missing/unordered frame')
        if self.first_vision is None and any(k.startswith('e') for k in f.units):
            self.first_vision = f.loop
        contact = _first_contact(payload, where)
        if contact is not None:
            self.label = _labels(payload, contact, where)
        elif self.first_vision is not None:
            self.updates.append(dict(loop=f.loop, **forecasts(f, self.prev)))
        self.prev = f

    def result(self):
        status = ('no_vision' if self.first_vision is None else 'no_contact' if self.label is None else
                  'already_in_contact' if not self.updates else 'forecastable')
        return dict(status=status, first_vision=self.first_vision, label=self.label, updates=self.updates)


def score(pred, label):
    # Marginals for a missed dog use the canonical observed front. On a tie,
    # every component of a joint hit must match the SAME dog.
    hit = pred['dog'] in label['dogs']
    actual = label['dogs'][pred['dog'] if hit else label['canonical']]
    error = math.dist(pred['point'], actual['point'])
    cluster = set(pred['cluster']) == set(actual['cluster'])
    expected = actual['support']
    support = None if expected is None else set(pred['support']) == set(expected)
    joint = None if support is None else hit and error <= RADIUS and support
    return dict(dog=hit, canonical_dog=pred['dog'] == label['canonical'], region=error <= RADIUS,
                cluster_exact=cluster, support_exact=support, actual_empty=expected == [],
                both_empty=expected == [] and not pred['support'], joint=joint,
                joint_nonempty=bool(joint and expected), joint_with_cluster=None if joint is None else joint and cluster,
                point_error=error, dog_position_error=math.dist(pred['dog_position'], actual['dog_position']))


def summarize(rows):
    result = dict(games=len(rows), statuses=dict(Counter(r['status'] for r in rows)),
                  tied_contacts=sum(bool(r['label'] and len(r['label']['dogs']) > 1) for r in rows))
    eligible = [r for r in rows if r['status'] == 'forecastable']
    for stage, index in [('earliest', 0), ('last', -1)]:
        leads = [r['label']['loop'] - r['updates'][index]['loop'] for r in eligible]
        stage_stats = dict(eligible=len(eligible), lead_loops_hist=dict(sorted(Counter(leads).items())),
                           median_lead=median(leads) if leads else None)
        for model in MODELS:
            scores, methods = [], Counter()
            for row in eligible:
                p = row['updates'][index][model]
                if p is not None:
                    scores.append(score(p, row['label']))
                    methods[p['method']] += 1
            hits = {k: sum(s[k] is True for s in scores) for k in (
                'dog', 'canonical_dog', 'region', 'cluster_exact', 'support_exact', 'joint',
                'joint_with_cluster', 'actual_empty', 'both_empty', 'joint_nonempty')}
            stage_stats[model] = dict(emitted=len(scores), abstained=len(eligible)-len(scores),
                                      unknown_support=sum(s['support_exact'] is None for s in scores),
                                      methods=dict(methods), **hits, **{
                                          f'mean_{k}': mean(s[k] for s in scores) if scores else None
                                          for k in ('point_error', 'dog_position_error')})
        result[stage] = stage_stats
    return result


def design(root):
    shards = {}
    for name, comp in COMPOSITIONS.items():
        for path in sorted(root.glob(f'{name}_*/manifest.json')):
            m = json.loads(path.read_bytes())
            _check(m['sha256'] == _digest({k: v for k, v in m.items() if k != 'sha256'}), path, 'manifest hash')
            rows = []
            for i, e in enumerate(m['episodes'], 1):
                cfg = e['episode_config']
                _check(e['sha256'] == _digest(cfg) and type(e['sc2_seed']) is int and all(
                    Counter(cfg['team_gen'][side]) == {u.lower(): n for u, n in comp.items()}
                    for side in ('ally_team', 'enemy_team')), path, 'scenario hash/seed/composition')
                # Cross-check the seed actually recorded by the runner. No score
                # or manifest fields are supplied to Episode.observe/forecasts.
                record = json.loads((path.parent / f'score_details/episode-{i:04d}.json').read_bytes())
                _check((record['episode'], record['sc2_seed'], record['scenario_sha256'], record['manifest_sha256'])
                       == (i, e['sc2_seed'], e['sha256'], m['sha256']), path, 'runner/manifest seed mismatch')
                rows.append(dict(shard=path.parent.name, episode=i, seed=e['sc2_seed'], scenario=e['sha256'],
                                 composition=name, layout=_layout(cfg, path)))
            shards[path.parent / 'events.jsonl'] = rows
    _check(shards, root, 'no pure 3/7 or 6/4 shards')
    expected = {DEV: 75, TEST: 20}.get(root.resolve())
    if expected is not None:
        sizes = Counter(r['composition'] for rows in shards.values() for r in rows)
        _check(sizes == dict.fromkeys(COMPOSITIONS, expected), root, f'incomplete cohort: {sizes}')
    return shards


def check_splits(designs):
    audit = {}
    for field in ('seed', 'scenario'):
        a, b = ([r[field] for rows in shards.values() for r in rows] for shards in designs)
        _check(len(a) == len(set(a)) and len(b) == len(set(b)), field, 'duplicate within cohort')
        shared = set(a) & set(b)
        _check(not shared, field, f'overlap across cohorts: {sorted(shared)}')
        audit[field] = dict(development=len(a), heldout=len(b), overlap=len(shared))
    return audit


def replay(path, metadata):
    rows, episode, n = [], None, 0
    with path.open('rb') as stream:
        for line in stream:
            event = json.loads(line)
            kind = event['event']
            if kind == 'episode_started':
                i = len(rows)
                _check(episode is None and i < len(metadata) and metadata[i]['episode'] == event['episode'] and
                       metadata[i]['scenario'] == event['scenario_sha256'], path, 'episode/manifest mismatch')
                episode, n = Episode(), 0
            elif kind == 'skill_observation':
                s = event['payload']['state']
                _check(episode is not None and s['episode'] == event['episode'] == len(rows)+1 and n == event['step'],
                       path, 'observation gap or wrong episode')
                episode.observe(event['payload'], path)
                n += 1
            elif kind == 'episode_completed':
                _check(episode is not None and n == event['steps'] > 0 and event['episode'] == len(rows)+1,
                       path, 'incomplete episode')
                rows.append(dict(metadata[len(rows)], **episode.result()))
                episode = None
    _check(episode is None and len(rows) == len(metadata), path, 'incomplete log')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development', type=Path, default=DEV, help='policy directory containing pure shards')
    p.add_argument('--heldout', type=Path, default=TEST, help='independent policy directory containing pure shards')
    p.add_argument('--json', type=Path, help='write full report including updates, labels and seed provenance')
    args = p.parse_args()
    designs = [design(root) for root in (args.development, args.heldout)]
    audit = check_splits(designs)
    report = dict(split_audit=audit, cohorts={}, methodology={
        'point': 'centroid of enemies within 3 of any dog legal target; region hit within 3 world units of observed centroid',
        'support': 'exact live allied hydra ID set; both empty matches, only one empty misses, unknown is not empty',
        'joint': 'any actual tied dog + region + support for that same dog; joint_with_cluster also requires exact enemy IDs',
        'denominators': 'one row per forecastable game per stage, never pooled updates; abstentions and unknown labels not hits',
        'scope': 'sampled mask and geometric opportunity, not attacks, hits, or causal policy effects'})
    print('overlap checks:', audit)
    for group, root, shards in zip(('development', 'heldout'), (args.development, args.heldout), designs):
        games = [r for path, metadata in shards.items() for r in replay(path, metadata)]
        sets = {'all': games, **{key: [r for r in games if r['layout'] == key] for key in ('center', 'scattered')}}
        stats = {key: summarize(rows) for key, rows in sets.items()}
        report['cohorts'][group] = dict(root=str(root.resolve()), stats=stats, games=games)
        for key, summary in stats.items():
            print(f"{group}/{key}: {summary['games']} games {summary['statuses']}; tied contacts {summary['tied_contacts']}")
            for stage in ('earliest', 'last'):
                s = summary[stage]
                for model in MODELS:
                    m = s[model]
                    print(f"  {stage:8} {model:9} /{s['eligible']} median lead={s['median_lead']} loops: "
                          f"dog={m['dog']} region={m['region']} cluster={m['cluster_exact']} support={m['support_exact']} "
                          f"joint={m['joint']} (nonempty={m['joint_nonempty']}) strict+cluster={m['joint_with_cluster']}")
    if args.json:
        with args.json.open('w', encoding='utf-8') as out:
            json.dump(report, out, allow_nan=False, indent=2)
            out.write('\n')


if __name__ == '__main__':
    main()
