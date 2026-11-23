"""Instrument a local SC2 map copy. Never replaces an output.

With --resources, use one of the three shipped damage_maps directly or
generate a new copy. SMPQ may point at an executable; otherwise search PATH.
Resource mode requires the historical 32x32_flat template and explicit race.
Without smpq it copies the corresponding hash-verified prebuilt map.

Run one match per generated map (or isolate SC2 user profiles). The printed
bank name identifies the XML .SC2Bank file under the local player's Banks.
In section hits, keys 1..count contain pipe-separated records:
source player, source Galaxy tag, source type, target player, target Galaxy
tag, target type, damage effect, damage taken. Galaxy tags are not API tags.
"""
import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid

RACE_TYPES = {'zerg': ('Zergling', 'Baneling', 'Hydralisk'),
              'terran': ('Marine', 'Marauder', 'Medivac'),
              'protoss': ('Zealot', 'Stalker', 'Colossus')}
SOURCE_SHA256 = 'c6dc0ce9eb59387a91828b434f2ab802890da75d49a9c7a6e677f78ff6ab4a8c'
MAP_DIR = Path(__file__).resolve().parents[1] / 'damage_maps'
PREBUILT_SHA256 = {'zerg': '80c4268ff9eaf8d5b0284c0178391a50721f9d291346d92c64bf1b1ceb0a8588',
                  'terran': '0487ae14cca22055eb339c5f44b71819783d9167684502e6e1402985604d8c23',
                  'protoss': '62d16a66496059446f93e939f16247f63b567fb88218154378d95123434a2236'}


HANDLER = '''
bank smacHitBank;
trigger smacHitTrigger;
int smacHitCount = 0;
bool SmacHit(bool testConds, bool runActions) {
    unit s = EventUnitDamageSourceUnit();
    unit t = EventUnit();
    string src = "0|";
    string row;
    if (!runActions) { return true; }
    if (s != null) { src = IntToString(UnitGetTag(s)) + "|" + UnitGetType(s); }
    row = IntToString(EventUnitDamageSourcePlayer()) + "|" + src + "|" + IntToString(EventPlayer()) + "|" + IntToString(UnitGetTag(t)) + "|" + UnitGetType(t) + "|" + EventUnitDamageEffect() + "|" + FixedToString(EventUnitDamageAmount(), 8);
    smacHitCount += 1;
    BankValueSetFromString(smacHitBank, "hits", IntToString(smacHitCount), row);
    BankValueSetFromInt(smacHitBank, "hits", "count", smacHitCount);
    BankSave(smacHitBank);
    return true;
}
'''


RESOURCE_HANDLER = '''
fixed[8] smacTotals;
fixed[8] smacFrozen;
trigger smacHitTrigger;
trigger smacQuery;
bool SmacHit(bool testConds, bool runActions) {
    unit s = EventUnitDamageSourceUnit();
    int p = EventUnitDamageSourcePlayer();
    int t = UnitGetOwner(EventUnit());
    string kind;
    int i = 3;
    if (!runActions || (p != 1 && p != 2) || t != 3-p) { return true; }
    if (s != null) { kind = UnitGetType(s); }
    // CLASSIFY_SOURCE
    i += (p-1)*4;
    smacTotals[i] += EventUnitDamageAmount();
    return true;
}
bool SmacRead(bool testConds, bool runActions) {
    string cmd = EventChatMessage(false);
    int i = 0;
    int page;
    fixed x;
    if (!runActions) { return true; }
    if (cmd == "SMACHITbegin") {
        while (i < 8) { smacTotals[i] = 0; smacFrozen[i] = 0; i += 1; }
        return true;
    }
    if (cmd == "SMACHITfreeze") {
        while (i < 8) { smacFrozen[i] = smacTotals[i]; i += 1; }
        PlayerModifyPropertyInt(1,c_playerPropMinerals,c_playerPropOperSetTo,9108);
        return true;
    }
    while (i < 8) {
        if (cmd == "SMACHIT"+IntToString(i)) {
            x = smacFrozen[i];
            page = FixedToInt(x);
            PlayerModifyPropertyInt(1,c_playerPropMinerals,c_playerPropOperSetTo,10000+page);
            PlayerModifyPropertyInt(1,c_playerPropVespene,c_playerPropOperSetTo,i*4096+FixedToInt((x-IntToFixed(page))*4096));
        }
        i += 1;
    }
    return true;
}
'''

RESOURCE_INIT = '''
    smacHitTrigger = TriggerCreate("SmacHit");
    TriggerAddEventUnitDamaged(smacHitTrigger, null, c_unitDamageTypeAny, c_unitDamageEither, null);
    smacQuery = TriggerCreate("SmacRead");
    TriggerAddEventChatMessage(smacQuery,1,"SMACHIT",false);
'''


def begin_resource_damage(controller):
    """Enqueue counter reset after warm-up, before the next simulation step."""
    from s2clientprotocol import sc2api_pb2 as sc
    controller.act(sc.Action(action_chat=sc.ActionChat(channel=1, message='SMACHITbegin')))


def read_resource_damage(controller, record_replay=True):
    """Freeze and read both sides. Advances simulation: use only at episode end.

    Returns cumulative integer ticks (1/4096 damage), RACE_TYPES[race] + other
    for player 1 then player 2. Start a fresh game or begin after reset.
    Optional replay echo is a controller-authored message, not a tracker event.
    """
    from s2clientprotocol import sc2api_pb2 as sc

    def send(message):
        controller.act(sc.Action(action_chat=sc.ActionChat(channel=1, message=message)))
        controller.step(2)
        return controller.observe().observation

    before = controller.observe().observation.game_loop
    freeze = send('SMACHITfreeze')
    if freeze.player_common.minerals != 9108:
        raise RuntimeError('SMACHIT freeze not acknowledged')
    ticks = []
    for page in range(8):
        common = send('SMACHIT' + str(page)).player_common
        if not (10000 <= common.minerals < 100000 and common.vespene // 4096 == page):
            raise RuntimeError(f'SMACHIT invalid page {page}: {common}')
        ticks.append((common.minerals - 10000)*4096 + common.vespene % 4096)
    if record_replay:
        send('SMACHITv1:' + ':'.join(map(str, [before, freeze.game_loop, *ticks])))
    return ticks


def replay_resource_damage(path):
    """Read controller-authored resource snapshots (never infer from kills)."""
    import mpyq
    from s2protocol import versions

    with Path(path).open('rb') as f:
        a = mpyq.MPQArchive(f, listfile=False)
        h = versions.latest().decode_replay_header(a.header['user_data_header']['content'])
        proto = versions.build(h['m_version']['m_baseBuild'])
        snapshots = []
        for event in proto.decode_replay_message_events(a.read_file('replay.message.events') or b''):
            msg = event.get('m_string', b'')
            if not msg.startswith(b'SMACHITv1:') or event['_userid']['m_userId'] != 0:
                continue
            fields = msg.split(b':')[1:]
            if len(fields) != 10 or not all(x.isdigit() for x in fields):
                raise ValueError('Malformed SMACHITv1 snapshot')
            lo, hi, *ticks = map(int, fields)
            if not lo <= hi <= event['_gameloop']:
                raise ValueError('Invalid SMACHITv1 snapshot loop')
            snapshots.append(dict(freeze_loop_range=[lo, hi], ticks=ticks))
    if not snapshots:
        raise ValueError('Replay contains no SMACHITv1 snapshots')
    return snapshots


def generate(source, output, resources=False, race=None, smpq=None):
    source, output = Path(source), Path(output)
    if resources:
        if race not in RACE_TYPES:
            raise ValueError('Resource mode requires explicit --race protoss, terran or zerg')
        if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
            raise ValueError('Resource mode requires the verified 32x32_flat.SC2Map template')
    tool = smpq or os.environ.get('SMPQ') or shutil.which('smpq')
    if tool:
        tool = shutil.which(str(tool))
        if not tool:
            raise FileNotFoundError('Invalid smpq executable; check --smpq or SMPQ')
        tool = str(Path(tool).resolve())
    bank = 'SmacHits' + uuid.uuid4().hex[:16]
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        work = Path(tmp).resolve()
        archive = work / 'copy.SC2Map'
        if not tool:
            if resources:
                shutil.copyfile(MAP_DIR / f'10gen_{race}.SC2Map', archive)
                if hashlib.sha256(archive.read_bytes()).hexdigest() != PREBUILT_SHA256[race]:
                    raise ValueError(f'Unexpected prebuilt {race} map checksum')
                os.link(archive, output)
                return None
            raise FileNotFoundError('Custom map generation requires smpq on PATH, --smpq or SMPQ')
        shutil.copyfile(source, archive)
        def mpq(*args):
            subprocess.run([tool, *args, str(archive), 'MapScript.galaxy'], cwd=work, check=True)
        mpq('-x')
        member = work / 'MapScript.galaxy'
        script = member.read_text(encoding='utf-8-sig')
        signature = r'(?m)^void\s+InitMap\s*\(\s*\)\s*\{'
        if len(re.findall(signature, script)) != 1 or 'smacHitTrigger' in script:
            raise ValueError('Expected exactly one uninstrumented InitMap')
        init = f'''\n    BankLoad("{bank}", 1);
    smacHitBank = BankLastCreated();
    BankValueSetFromInt(smacHitBank, "hits", "count", 0);
    BankSave(smacHitBank);
    smacHitTrigger = TriggerCreate("SmacHit");
    TriggerAddEventUnitDamaged(smacHitTrigger, null, c_unitDamageTypeAny, c_unitDamageEither, null);
'''
        handler = RESOURCE_HANDLER if resources else HANDLER
        if resources:
            init = RESOURCE_INIT
            handler = handler.replace('// CLASSIFY_SOURCE', '\n    '.join(
                f'if (kind == "{name}" || kind == "{name}_RL") {{ i = {idx}; }}'
                for idx, name in enumerate(RACE_TYPES[race])))
        payload = re.sub(signature, lambda m: handler + m[0] + init, script).encode('utf-8')
        member.write_bytes(payload)
        mpq('-a', '-f')
        member.unlink()
        mpq('-x')
        if member.read_bytes() != payload:
            raise ValueError('Archive read-back mismatch')
        os.link(archive, output)
    return None if resources else bank


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--resources', action='store_true', help='Use chat-frozen player resources, no Bank')
    parser.add_argument('--race', choices=RACE_TYPES, help='Required with --resources; selects source-unit bins')
    parser.add_argument('--smpq', help='Executable path (default: SMPQ, then PATH)')
    args = parser.parse_args()
    bank = generate(args.source, args.output, args.resources, args.race, args.smpq)
    print(args.output if args.resources else bank + '.SC2Bank')
