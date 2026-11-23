"""Paired, teacher-forced replay on real policy observations. No SC2 imports."""
import importlib
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

if __package__:
    from .contact_front_variant import SOURCE, apply_variant
else:
    from contact_front_variant import SOURCE, apply_variant


def replay(path, control, variant, full_first=True):
    from skills.tactical_v1.joint_combat import action_position

    counts = Counter()
    examples = {}
    active = phase = None
    with path.open() as lines:
        for line in lines:
            ev = json.loads(line)
            kind, ep = ev['event'], ev.get('episode')
            if kind == 'episode_started':
                assert phase is None
                active = ep
                control.reset()
                variant.reset()
                contact = None
                engaged = set()
                counts['episodes'] += 1
            if kind not in ('skill_observation', 'skill_step', 'step'):
                continue
            step = ev['step'] - int(kind == 'step')
            if not (ep == 1 and full_first) and contact is not None and step > contact + 1:
                continue
            assert ep == active
            if kind == 'skill_observation':
                assert phase is None
                phase = ('skill_step', step)
                payload = ev['payload']
                s = payload['state']
                dogs = [a for a, u in s['allies'].items() if u['type'] == 'Zergling']
                guns = [a for a, u in s['allies'].items() if u['type'] == 'Hydralisk']
                cohort = (len(dogs), len(guns)) in ((3, 7), (6, 4))
                if any(any(int(k) >= 6 for k in payload['questions'].get(a, {}).get('criteria', {})) for a in dogs):
                    if contact is None:
                        contact = step
                engaged.update(a for a in dogs if not s['allies'][a]['alive'] or
                               s['allies'][a]['weapon_cooldown_loops'] or s['allies'][a]['health'] < s['allies'][a]['max_health'] or
                               any(o['kind'] == 'attack' for o in s['allies'][a].get('orders', ())))
                old, baseline = control.step(payload, 10)
                new, summary = variant.step(payload, 10)
                ctx = control._pending_commit[-1]
                counts['decisions'] += 1
            elif kind == 'skill_step':
                assert phase == (kind, step) and ev['game_loop'] == s['game_loop']
                phase = ('step', step)
                assert baseline['config_fingerprint'] == ev['config_fingerprint']
                assert dict(enumerate(old)) == {int(a[1:]): cmd for a, cmd in ev['actions'].items()}
                changes = summary['front_route']['changes']
                assert set(changes) == {f'u{i}' for i in range(10) if old[i] != new[i]}
                for i, (before, after) in enumerate(zip(old, new)):
                    a = f'u{i}'
                    row = ev['units'][a]
                    duty = ev['execution_duties']['actors'].get(a, {})
                    protected = (not cohort or contact is not None or a not in dogs or a in engaged or
                                 before not in (2, 3, 4, 5) or duty.get('kind') != 'entry' or
                                 duty.get('branch_reason') or duty.get('joint_rescue') or a in ev['joint_rescue_substitutions'] or
                                 (row['skill'], row['phase']) not in (('baseline', ''), ('cohesive_search', 'searching'),
                                                                     ('coordinated_engage', 'approaching')))
                    assert summary['actions'][a] == summary['units'][a]['action'] == after
                    if protected:
                        assert before == after, (path, ep, step, a)
                        counts['protected'] += 1
                        counts['post_contact'] += int(contact is not None)
                    if before == after:
                        continue
                    ch = changes[a]
                    assert (ch['old_action_id'], ch['new_action_id']) == (before, after)
                    assert after in ctx.move_destinations(a) and str(after) in payload['questions'][a]['criteria']
                    assert (before in (2, 3)) != (after in (2, 3))
                    assert not duty['required'] or after in duty['actions']
                    gun_ends = [action_position(ctx, h, old[int(h[1:])])[0] for h in guns if s['allies'][h]['alive']]
                    for cmd, m in ((before, ch['old']), (after, ch['new'])):
                        end = ctx.predict_position(a, cmd).position
                        assert list(end) == m['endpoint']
                        radius = s['action_contract']['candidate_center_distance_by_type']['hydralisk']
                        assert m['ally_cover'] == sum(math.dist(p, m['contact_zone']) <= radius for p in gun_ends)
                        assert m['enemy_cover'] == sum(math.dist(v['position'], end) <= ctx._enemy_weapon_range(e, False) +
                                                       v['radius'] + s['allies'][a]['radius'] for e, v in s['visible_enemies'].items()
                                                       if v['type'] == 'Hydralisk')
                        assert m['local_advantage'] == m['ally_cover'] - m['enemy_cover']
                        assert math.isclose(m['nearest_hydra_gap'], min(math.dist(end, p) for p in gun_ends))
                    o, n = ch['old'], ch['new']
                    assert not (o['ally_cover'] and o['local_advantage'] >= -1)
                    if ch['reason'] == 'covered_flank':
                        assert n['ally_cover'] and (n['local_advantage'] >= 0 or n['local_advantage'] > o['local_advantage'])
                    else:
                        assert ch['reason'] == 'limit_separation' and o['ally_cover'] == n['ally_cover'] == 0
                        assert n['nearest_hydra_gap'] < o['nearest_hydra_gap'] and n['enemy_cover'] <= o['enemy_cover']
                    counts['changed'] += 1
                    counts[ch['reason']] += 1
                    counts['with_vision' if s['visible_enemies'] else 'without_vision'] += 1
                    examples.setdefault(ch['reason'], dict(ep=ep, step=step, actor=a, **ch))
            else:
                # Only compare decisions on the same history: both copies commit
                # the actual submitted vector, not the counterfactual variant.
                assert phase == (kind, step) and ev['actions'] == old and ev['game_loop'] == s['game_loop']
                phase = None
                engaged.update(a for a in dogs if ev['actions'][int(a[1:])] >= 6)
                control.commit_final_actions(ev['actions'])
                variant.commit_final_actions(ev['actions'])
    assert phase is None and counts['episodes'] == 10
    print(path.parent.name, dict(counts), 'examples:', json.dumps(examples), flush=True)
    return counts


def main():
    if not __debug__:
        raise RuntimeError('run without -O: assertions perform the replay audit')
    sys.dont_write_bytecode = True
    with TemporaryDirectory() as temp:
        packages = Path(temp) / 'skills'
        packages.mkdir()
        copy = packages / 'route_experiment'
        shutil.copytree(SOURCE, copy, ignore=shutil.ignore_patterns('__pycache__'))
        (packages / 'current').symlink_to(SOURCE.parent / 'current', target_is_directory=True)
        try:
            apply_variant(SOURCE)
        except ValueError:
            pass
        else:
            raise AssertionError('installer accepted frozen package')
        apply_variant(copy)
        sys.path[:0] = [str(SOURCE.parents[1]), str(packages)]
        control = importlib.import_module('skills.tactical_v1.skill_manager').SkillManager()
        variant = importlib.import_module('route_experiment.skill_manager').SkillManager()
        total = Counter()
        for dogs, banes, guns in ((3, 0, 7), (6, 0, 4), (4, 1, 5)):
            group = Counter()
            for shard in (0, 1) if not banes else (0,):
                path = SOURCE.parents[1] / f'zerg_{dogs}zergling_{banes}baneling_{guns}hydralisk_{shard}/events.jsonl'
                group.update(replay(path, control, variant, full_first=shard == 0))
            assert group['protected'] and group['post_contact'] and bool(group['changed']) == (not banes)
            total.update(group)
        assert all(total[k] > 0 for k in ('covered_flank', 'limit_separation', 'with_vision', 'without_vision'))
        print('PASS: same-observation final commands changed as intended;', dict(total))
        print('Coverage and endpoints are predictions, not observations of variant damage, contact or win rate.')


if __name__ == '__main__':
    main()
