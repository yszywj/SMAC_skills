"""Replay recorded input only; no SC2 imports, process or simulator."""
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from contact_formation_variant import FROZEN, MODULE, apply_variant


def observations(roster, count):
    path = FROZEN.parents[1] / f'zerg_{roster}_0/events.jsonl'
    with path.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event['event'] == 'skill_observation':
                yield event['payload']
                count -= 1
                if not count:
                    return


class FormationVariantTest(unittest.TestCase):
    def test_recorded_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            skills = Path(tmp) / 'skills'
            shutil.copytree(FROZEN.parent, skills)
            variant = skills / 'formation_b'
            shutil.copytree(skills / 'tactical_v1', variant)
            with self.assertRaises(ValueError):
                apply_variant(FROZEN)
            apply_variant(variant)
            self.assertEqual([p.relative_to(variant) for p in variant.rglob('*') if p.is_file()
                              and p.read_bytes() != (FROZEN / p.relative_to(variant)).read_bytes()], [MODULE])
            with patch.object(sys, 'path', [tmp, str(skills), *sys.path]):
                ref = importlib.import_module('tactical_v1.skill_manager').SkillManager
                exp = importlib.import_module('formation_b.skill_manager').SkillManager
                for roster, pure in [('3zergling_0baneling_7hydralisk', True),
                                     ('4zergling_1baneling_5hydralisk', False)]:
                    original, modified = ref(), exp()
                    disabled = exp()
                    disabled.skills['ranged_fire_formation'].enabled = False
                    locked = forced_count = changed = 0
                    for payload in observations(roster, 7):
                        base, _ = original.step(payload, 10)
                        acts, _ = modified.step(payload, 10)
                        ctx = modified._pending_commit[-1]
                        orders = ctx.formation_orders
                        if pure:
                            self.assertEqual(acts, base)
                            self.assertEqual(orders, original._pending_commit[-1].formation_orders)
                        else:
                            self.assertFalse(original._pending_commit[-1].formation_orders)
                            for actor, cmd in orders.items():
                                self.assertEqual(ctx.snapshot.allies[actor].type_name, 'hydralisk')
                                self.assertFalse(ctx.attackable_enemies(actor))
                                self.assertEqual(cmd, acts[int(actor[1:])])
                                self.assertIn(ctx.decode(actor, cmd).kind, ('move', 'stop'))
                                changed += cmd != base[int(actor[1:])]
                            off, _ = disabled.step(payload, 10)
                            self.assertFalse(disabled._pending_commit[-1].formation_orders)
                            disabled.commit_final_actions(off)
                        for i, cmd in enumerate(acts):
                            a = f'u{i}'
                            self.assertIn(cmd, ctx.legal_actions(a))
                            forced = ctx.forced_hydra_targets(a)
                            self.assertEqual(forced, original._pending_commit[-1].forced_hydra_targets(a))
                            if forced:
                                self.assertIn(ctx.decode(a, cmd).target, forced)
                                forced_count += 1
                        locked += len(orders)
                        original.commit_final_actions(base)
                        modified.commit_final_actions(acts)
                    self.assertGreater(locked, 0)
                    if not pure:
                        self.assertGreater(forced_count, 0)
                        self.assertGreater(changed, 0)


if __name__ == '__main__':
    unittest.main()
