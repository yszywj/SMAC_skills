"""Read the native 10v10 Zerg roster and last hits from an SC2 replay."""

from collections import Counter
from pathlib import Path

import mpyq
from s2protocol import versions


def read_roster(replay_path):
    """Return rows keyed by integer raw-API tag for the 98310 benchmark.

    Each row holds side (ally/enemy), native type, birth loop, death loop
    or None, and the killer's raw tag or None. A suicide retains its self
    tag. The tracker identifies last hits but carries no hit-point amounts.
    """
    path = Path(replay_path)
    with path.open('rb') as source:
        archive = mpyq.MPQArchive(source, listfile=False)
        header = versions.latest().decode_replay_header(archive.header['user_data_header']['content'])
        build = header['m_version']['m_baseBuild']
        if build != 98310:
            raise ValueError(f'{path}: raw unit-tag mapping has not been verified for base build {build}')
        protocol = versions.build(build)
        tracker = archive.read_file('replay.tracker.events')
        if not tracker:
            raise ValueError(f'{path}: missing replay tracker events')

    def raw_tag(index, recycle):
        # The runner's raw API includes a high word absent from tracker
        # handles. Verified against native observations for both players.
        return (1 << 32) | protocol.unit_tag(index, recycle)

    roster = {}
    sides = {1: 'ally', 2: 'enemy'}
    native_types = {'Zergling', 'Hydralisk', 'Baneling'}
    for event in protocol.decode_replay_tracker_events(tracker):
        kind = event['_event'].rsplit('.', 1)[-1]
        if kind in ('SUnitOwnerChangeEvent', 'SUnitTypeChangeEvent', 'SUnitInitEvent'):
            raise ValueError(f'{path}: unexpected {kind} in a fixed-roster battle')
        if kind not in ('SUnitBornEvent', 'SUnitDiedEvent'):
            continue
        tag = raw_tag(event['m_unitTagIndex'], event['m_unitTagRecycle'])
        loop = event['_gameloop']
        if kind == 'SUnitBornEvent':
            side = sides.get(event['m_controlPlayerId'])
            unit_type = event['m_unitTypeName'].decode('ascii')
            if tag in roster or side is None or unit_type not in native_types:
                raise ValueError(f'{path}: unexpected birth {tag=} {side=} {unit_type=}')
            roster[tag] = dict(side=side, type=unit_type, born=loop, death=None, killer=None)
        else:
            unit = roster.get(tag)
            if unit is None or unit['death'] is not None or loop < unit['born']:
                raise ValueError(f'{path}: unexpected death for tag {tag} at loop {loop}')
            index, recycle = event['m_killerUnitTagIndex'], event['m_killerUnitTagRecycle']
            if (index is None) != (recycle is None):
                raise ValueError(f'{path}: incomplete killer tag at loop {loop}')
            unit['death'] = loop
            unit['killer'] = raw_tag(index, recycle) if index is not None else None

    counts = Counter(unit['side'] for unit in roster.values())
    if counts != {'ally': 10, 'enemy': 10}:
        raise ValueError(f'{path}: expected ten native units per side, found {dict(counts)}')
    for unit in roster.values():
        if unit['killer'] is not None and unit['killer'] not in roster:
            raise ValueError(f"{path}: killer {unit['killer']} is outside the battle roster")
    return roster
