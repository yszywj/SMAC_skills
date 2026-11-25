"""Exercise the optional controller against a real archived random-panel opening."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bridge import runner, zerg_shadow
from bridge.skill_runtime import manager_configuration
from bridge.zerg_distill import Distill
from decision_tree.zerg.opening_support_v2 import RULE_ID, ranked_moves, rule_hash


ROOT = Path(__file__).resolve().parents[2]
PAIR = ROOT/'runs/zerg_policy_eval_random_20261124_165149/pair-106'


@unittest.skipUnless(PAIR.exists(), 'archive unavailable')
class Opening(unittest.TestCase):
    def setUp(self):
        self.root = PAIR/'tree/runner'
        with (self.root/'events.jsonl').open() as f:
            self.payload = next(json.loads(l)['payload'] for l in f if '"skill_observation"' in l[:150])
        self.d = json.loads((self.root/'distill/episode-0001.json').read_text())['decisions'][0]

    def test_contact_damage_and_attack_veto(self):
        p, base = self.payload, self.d['final_actions']
        self.assertTrue(ranked_moves(p,base))
        for q in p['questions'].values():
            with patch.dict(q['criteria'], {'6':'legal shot'}):
                actor = next(a for a,row in p['questions'].items() if row is q)
                self.assertNotIn(actor,[a for _,_,a,_ in ranked_moves(p,base)])
        with patch.dict(p['state']['allies']['u0'],health=0):
            self.assertFalse(ranked_moves(p,base))
        with patch.dict(p['state']['relations'],u0={'e0':{'in_enemy_weapon_range':True,'in_weapon_range':False}}):
            self.assertFalse(ranked_moves(p,base))

    def test_unique_registration_exact_commit_and_micro(self):
        manifest=json.loads((PAIR/'manifest.json').read_text())
        shadow=json.loads((self.root/'run.json').read_text())['zerg_shadow']
        args=runner.parser().parse_args(['run','--manifest','unused','--output-dir','unused','--method',
            'adaptive-skill','--skill-version','tactical_v1','--execution-mode','micro8-all'])
        with TemporaryDirectory() as tmp, patch.dict(os.environ, ZERG_DISTILL_VERSION='v2',
                                                      ZERG_DISTILL_OPENING_SHA256=rule_hash()):
            root=Path(tmp); rt,ctrl=zerg_shadow.start_shadow(str(root/'shadow'),shadow['scene_config_id'],
                manifest,'adaptive-skill',args,manager_configuration(args)['manager_kwargs'])
            model=PAIR.parent/'model.json';hook=Distill(root,model,shadow)
            scene=manifest['episodes'][0];hook.start(1,scene,'offline','4.10.0.75689')
            rt.episode_seed=scene['sc2_seed'];ctrl.reset();m=ctrl.manager;m.micro_all_attackers=True
            ctrl.select('adaptive_focus_fire',posture='balanced');p=self.payload
            env=SimpleNamespace(n_agents=10,get_avail_agent_actions=lambda i:[str(a) in p['questions'][f'u{i}']['criteria'] for a in range(16)])
            actions,summary=hook.step(0,p['state']['game_loop'],ctrl,p,env)
            row,=rt._buffer.values(); card,=[c for c in row['candidates'] if c['candidate_id']==RULE_ID]
            self.assertNotEqual(row['teacher_choice'],RULE_ID)
            self.assertEqual(card['proposed_actions'],summary['actions'])
            self.assertEqual(row['selected_candidate'],RULE_ID)
            self.assertEqual(hook.decision['selection_reason'],'generated_opening_support')
            self.assertEqual(hook.decision['selector_audit']['v1_actions'],[self.d['final_actions'][f'u{i}'] for i in range(10)])
            self.assertFalse(any(e.actions == summary['actions'] for e in m.coordinator._cache.values()))
            bad=list(actions);bad[0]=-1
            with self.assertRaisesRegex(ValueError,'COMMIT'):hook.before_commit(m,bad)
            hook.before_commit(m,actions);ctrl.commit_final_actions(actions);hook.committed(m,actions)
            self.assertEqual(m.memory.last_committed_actions,card['proposed_actions'])
            changed,=[u for u,a in summary['actions'].items() if a!=self.d['final_actions'][u]]
            self.assertNotIn(changed,[g['actor'] for g in m.last_micro_intents]);hook.engine({})
            self.assertEqual(hook.decision['macro_receipt'],'unknown')
            with patch.dict(os.environ,ZERG_DISTILL_VERSION='v1'):
                with self.assertRaisesRegex(ValueError,'version mismatch'):Distill(root,model,shadow)


if __name__=='__main__':unittest.main()
