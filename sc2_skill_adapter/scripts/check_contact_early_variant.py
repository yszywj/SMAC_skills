"""Replay recorded policy inputs, not SC2 or omniscient replay data."""
import importlib
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

from contact_early_variant import SOURCE, apply_variant


def main():
    sys.dont_write_bytecode = True
    with TemporaryDirectory() as temp:
        sys.path[:0] = [temp, str(SOURCE.parents[1])]
        copy = Path(temp) / 'experiment'
        shutil.copytree(SOURCE, copy, ignore=shutil.ignore_patterns('__pycache__'))
        try:
            apply_variant(SOURCE)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted frozen source')
        apply_variant(copy)
        mgr = importlib.import_module('experiment.skill_manager').SkillManager()
        for lings, banes, guns in ((3, 0, 7), (4, 1, 5), (6, 0, 4)):
            mgr.reset()
            shard = f'zerg_{lings}zergling_{banes}baneling_{guns}hydralisk_0'
            changed = protected = after_contact = 0
            seen = set()
            with (SOURCE.parents[1] / shard / 'events.jsonl').open() as lines:
                for line in lines:
                    event = json.loads(line)
                    episode = event.get('episode', 1)
                    if episode > (10 if lings == 6 else 1):
                        break
                    if event['event'] == 'episode_started':
                        mgr.reset()
                        seen.clear()
                    if episode > 1 and event.get('step', 0) >= 8:
                        continue
                    if event['event'] == 'skill_observation':
                        actions, summary = mgr.step(event['payload'], 10)
                        ctx = mgr._pending_commit[-1]
                    if event['event'] != 'skill_step':
                        continue
                    original = event['actions']
                    for i, cmd in enumerate(actions):
                        actor = f'u{i}'
                        unit = ctx.snapshot.allies[actor]
                        has_target = any(a >= 6 for a in ctx.legal_actions(actor))
                        if actor in seen and not has_target and unit.alive:
                            after_contact += 1
                        if has_target or unit.weapon_cooldown_loops > 0:
                            seen.add(actor)
                        row = event['units'][actor]
                        if (banes or not unit.alive or unit.type_name != 'hydralisk' or
                                actor in seen or original[actor] != 1 or event['step'] >= 8):
                            assert cmd == original[actor], (shard, event['step'], actor)
                            protected += 1
                        if cmd != original[actor]:
                            assert row['skill'] == 'ranged_fire_formation'
                            assert summary['units'][actor]['goal'] == row['goal']
                            assert cmd in ctx.move_destinations(actor)
                            from math import dist
                            goal = ctx.snapshot.enemies[row['goal']].position
                            assert dist(ctx.predict_position(actor, cmd).position, goal) < dist(unit.position, goal)
                            changed += 1
                    # Teacher-force actual committed history: only this decision's
                    # actions are comparable on the subsequently recorded state.
                    mgr.commit_final_actions([original[f'u{i}'] for i in range(10)])
            print(f'{shard}: changed={changed}, protected={protected}, after_contact={after_contact}')
            assert (changed > 0) == (banes == 0) and protected and after_contact


if __name__ == '__main__':
    main()
