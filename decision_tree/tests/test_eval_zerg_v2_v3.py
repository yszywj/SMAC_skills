"""Full policy identity, replay panels, and exact archived native decisions. Never starts SC2."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from decision_tree.scripts.linux import eval_zerg_v2_v3 as trial

ev = trial.ev


class FullTrial(unittest.TestCase):
    @unittest.skipUnless(sys.executable == ev.RSI_PYTHON, 'archive interpreter required')
    def test_panels_and_pins(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(ev, 'play', side_effect=AssertionError):
            root = Path(tmp)
            with patch.object(ev, 'HISTORY', root/'reservation'):
                trial.main(['--out',str(root/'fresh'),'--dry-run'])
                d = ev.read_json(root/'fresh/design.json')
                self.assertEqual((d['workers'],len(list((root/'fresh').glob('pair-*/v[23]')))), (60,480))
                for mode, name in trial.ALIASES.items():
                    out = root/name
                    r = trial.prepare(out, batch=mode, ids=[42,11], dry=True)
                    old = trial.archive(name)
                    self.assertEqual(r['pairs'], [old['pairs'][41], old['pairs'][10]])
                    self.assertNotEqual(r['trial_id'],d['trial_id'])
                    for a in trial.ARMS:
                        _, env = ev.command(out,r['pairs'][0],a,r['policies'][a])
                        self.assertEqual(env['ZERG_DISTILL_VERSION'],a)
                        self.assertEqual(env['ZERG_DISTILL_OPENING_SHA256'],old['policies']['v2']['opening_sha256'])
                        self.assertEqual(env['ZERG_DISTILL_TRIAL_ID'],r['trial_id'])
                    with self.assertRaises(ValueError):
                        trial.compare(out)
                with self.assertRaisesRegex(ValueError,'dependency changed'):
                    trial.policies(dict(d['sources'],**{trial.OPENING:'bad'}),{},'trial')

    def test_native_opening_and_first_volley_change(self):
        from bridge import runner, zerg_shadow
        from bridge.skill_runtime import manager_configuration
        from bridge.zerg_distill import Distill
        from decision_tree.zerg.opening_support_v2 import rule_hash
        archive = ev.RUNS/trial.FRESH
        d = trial.archive(trial.FRESH)
        args = runner.parser().parse_args(['run','--manifest','unused','--output-dir','unused','--method',
            'adaptive-skill','--skill-version','tactical_v1','--execution-mode','micro8-all'])
        for number, last in ((11,0),(42,2)):
            p = d['pairs'][number-1]
            worker = archive/p['pair_id']/'v2/runner'
            shadow = ev.read_json(worker/'run.json')['zerg_shadow']
            old = ev.read_json(worker/'distill/episode-0001.json')['decisions']
            manifest = ev.read_json(archive/p['manifest'])
            with (worker/'events.jsonl').open() as stream:
                events = [json.loads(line) for line in stream]
            pairs = list(zip([e['payload'] for e in events if e['event']=='skill_observation'],
                             [e for e in events if e['event']=='step']))
            for arm in trial.ARMS:
                with self.subTest(pair=number, arm=arm), tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,
                        ZERG_DISTILL_VERSION=arm, ZERG_DISTILL_TRIAL_ID='test', ZERG_DISTILL_OPENING_SHA256=rule_hash()):
                    root = Path(tmp)
                    rt, ctrl = zerg_shadow.start_shadow(str(root/'shadow'),p['roster'],manifest,'adaptive-skill',
                                                        args, manager_configuration(args)['manager_kwargs'])
                    hook = Distill(root, archive/'model.json', shadow)
                    scene, = manifest['episodes']
                    hook.start(1,scene,'offline','4.10.0.75689')
                    rt.episode_seed = scene['sc2_seed']
                    ctrl.reset(); ctrl.manager.micro_all_attackers = True
                    for step,(pre,event) in enumerate(pairs[:last+1]):
                        if not ctrl.active:
                            ctrl.select('adaptive_focus_fire',posture='balanced')
                        def mask(i):
                            q = pre['questions'].get(f'u{i}')
                            return [str(a) in q['criteria'] if q else a==0 for a in range(16)]
                        acts,_ = hook.step(step,event['game_loop'],ctrl,pre,SimpleNamespace(n_agents=10,get_avail_agent_actions=mask))
                        hook.before_commit(ctrl.manager,acts); ctrl.commit_final_actions(acts); hook.committed(ctrl.manager,acts)
                        row = hook.decision
                        changed = arm=='v3' and number==42 and step==2
                        self.assertEqual(acts != event['actions'],changed)
                        hook.engine({} if changed else event['feedback'])
                        self.assertEqual(row['macro_receipt'],'unknown' if changed else 'accepted')
                        if not changed:
                            self.assertEqual(row['plan_ids'],old[step]['plan_ids'])
                        if number==11:
                            self.assertEqual(row['selection_reason'],'generated_opening_support')
                        if arm=='v3':
                            self.assertEqual(row['v2_reference_actions'],old[step]['final_actions'])
                            self.assertEqual((row['v3_triggered'],row['v3_actions_changed']),(changed,changed))
                        record = rt._buffer[rt.run_id,'1',row['snapshot_id'],row['game_loop']]
                        cards = {c['candidate_id']:c for c in record['candidates']}
                        if changed:
                            with self.assertRaisesRegex(ValueError,'native receipt'):
                                trial.check_trial_decision(row,cards,arm,hook.contract)
                        else:
                            trial.check_trial_decision(row,cards,arm,hook.contract)
                            with self.assertRaisesRegex(ValueError,'wrong trial'):
                                trial.check_trial_decision(dict(row,trial_id='different'),cards,arm,hook.contract)


if __name__ == '__main__':
    unittest.main()
