import json
from pathlib import Path
import tempfile
import unittest

from decision_tree.scripts.linux.summarize_squad_ablation import contact,effect


class AblationMetricsTest(unittest.TestCase):
    def test_counts_draws_without_silently_calling_them_losses(self):
        rows=[dict(a='loss',b='win'),dict(a='win',b='draw'),dict(a='draw',b='win')]
        self.assertEqual(effect(rows,'a','b'),dict(pairs=3,net_wins=1,gained=2,lost=1,
                       transitions={'loss_to_win':1,'win_to_draw':1,'draw_to_win':1}))

    def test_contact_is_shared_target_not_any_three_legal_shooters(self):
        def observation(loop, targets):
            rel={f'u{i}':{e:dict(smac_attack_available=True) for e in targets[i]} for i in range(len(targets))}
            return json.dumps(dict(event='skill_observation',payload=dict(state=dict(game_loop=loop,relations=rel))))+'\n'
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'events.jsonl'
            with path.open('w') as f:f.write(observation(2,['a','b','c']))
            self.assertIsNone(contact(path))
            with path.open('a') as f:f.write(observation(34,['ab','ac','c']))
            self.assertEqual(contact(path),2)
