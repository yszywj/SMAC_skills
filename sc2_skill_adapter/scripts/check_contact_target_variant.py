"""Offline A/control comparison via two private packages and the real SkillManager."""
import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

from contact_target_variant import ROOT, SOURCE, apply_variant


def main():
    sys.path.insert(0, str(ROOT))
    from skills.tactical_v1.tests.fixtures import enemy, scene, with_questions

    with tempfile.TemporaryDirectory() as tmp:
        sys.path.insert(0, tmp)
        managers = []
        for name in ('control', 'experiment_a'):
            package = Path(tmp) / name
            shutil.copytree(SOURCE, package)
            if name == 'experiment_a':
                apply_variant(package)
            managers.append(importlib.import_module(name + '.skill_manager').SkillManager)

        def run(kind, hp, formation=True):
            # Two ready formation-owned guns; e1 is next to u0 but u1 is
            # outside the blast. Neither gun has to move to switch targets.
            enemies = {'e0': enemy((13., 11.), 'hydralisk', health=90., max_health=90.),
                       'e1': enemy((11., 10.), kind, health=hp)}
            if not formation:
                enemies['e2'] = enemy((14., 14.), 'zergling')  # activate frozen snake+ling lock
            state = scene([((10., 10.), 'hydralisk' if formation else 'zergling'),
                           ((10., 14.), 'hydralisk')], enemies, map_name='10gen_zerg')
            state['unit_type_profiles']['9'] = json.loads(
                (SOURCE / 'burst_native_profile.json').read_text())['catalog_contract']
            result = []
            for cls in managers:
                mgr = cls()
                actions, summary = mgr.step(with_questions(state), 2)
                if kind == 'baneling' and actions[1] == 7:
                    assert 'loss_control_intercept' in str(summary['chosen_plans'])
                mgr.commit_final_actions(actions)
                result.append(actions)
            print(kind, hp, 'control / A:', result)
            return result

        assert run('zergling', 35.) == [[6, 6], [6, 6]]
        control, variant = run('zergling', 10.)
        assert control == [6, 6] and sorted(variant) == [6, 7]
        control, variant = run('baneling', 10.)
        assert control == [6, 6] and variant == [6, 7]
        control, variant = run('baneling', 10., formation=False)
        assert control[1] == 6 and variant[1] == 7


if __name__ == '__main__':
    main()
