"""Route launcher, actual native menu replay and offline receipt checks. No SC2."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from decision_tree.scripts.linux import eval_zerg_route as trial

ev=trial.ev


@unittest.skipUnless(sys.executable==ev.RSI_PYTHON, 'archive environment required')
class Route(unittest.TestCase):
    def test_native_first_anchor_and_shared_opening(self):
        from bridge import runner,zerg_shadow
        from bridge.skill_runtime import manager_configuration
        from bridge.zerg_distill import Distill
        args=runner.parser().parse_args(['run','--manifest','unused','--output-dir','unused','--method',
            'adaptive-skill','--skill-version','tactical_v1','--execution-mode','micro8-all'])
        with tempfile.TemporaryDirectory() as temp, patch.object(ev,'play',side_effect=AssertionError):
            root=Path(temp)
            with patch.object(ev,'HISTORY',root/'reservations'):
                d=trial.prepare(root/'d',2,'fresh',[3,11],True,None)
                for p in d['pairs']:
                    m=ev.read_json(root/'d'/p['manifest'])
                    archive=ev.RUNS/trial.old.FRESH/p['pair_id']/'v2/runner'
                    events=[json.loads(s) for s in (archive/'events.jsonl').read_text().splitlines()]
                    pre=next(e['payload'] for e in events if e['event']=='skill_observation')
                    real=next(e for e in events if e['event']=='step')
                    anchors=[]
                    gate_model=dict(schema='zerg_route_gate_v1',default='v2',tree=dict(feature='changed_orders',op='gt',threshold=0),policy_sha256=d['policy_sha256'])
                    (root/'d/gate.json').write_text(json.dumps(gate_model))
                    for a in ('v2','alternative','route'):
                        policy=d['policies'][a] if a!='route' else dict(d['policies']['alternative'],kind='zerg_distill_route',policy_version=a,gate_sha256=ev.sha(root/'d/gate.json'))
                        _,env=ev.command(root/'d',p,a,policy)
                        work=root/p['pair_id']/a
                        work.mkdir(parents=True)
                        with patch.dict(os.environ,env):
                            with patch.object(runner,'_evaluate') as dispatch:
                                asyncio.run(runner.evaluate(args,m,'adaptive-skill',work))
                                dispatch.assert_awaited_once()
                            rt,ctrl=zerg_shadow.start_shadow(str(work/'shadow'),p['roster'],m,'adaptive-skill',args,manager_configuration(args)['manager_kwargs'])
                            hook=Distill(work,root/'d/model.json',ev.read_json(archive/'run.json')['zerg_shadow'])
                            scene,=m['episodes']
                            hook.start(1,scene,'offline','4.10.0.75689')
                            rt.episode_seed=scene['sc2_seed']
                            ctrl.reset();ctrl.manager.micro_all_attackers=True
                            ctrl.select('adaptive_focus_fire',posture='balanced')
                            def mask(i):
                                q=pre['questions'].get(f'u{i}')
                                return [str(a) in q['criteria'] if q else a==0 for a in range(16)]
                            acts,_=hook.step(0,real['game_loop'],ctrl,pre,SimpleNamespace(n_agents=10,get_avail_agent_actions=mask))
                            hook.before_commit(ctrl.manager,acts);ctrl.commit_final_actions(acts);hook.committed(ctrl.manager,acts)
                            row=hook.decision
                            self.assertEqual(row['parent_plan']['actions'],real['actions'])
                            anchors.append(row['route_anchor'])
                            if p['pair_id']=='pair-003':
                                self.assertEqual(row['route_choice'],'v2' if a=='v2' else 'alternative')
                                self.assertEqual(acts==real['actions'],a=='v2')
                                self.assertEqual(row['route_anchor']['features']['changed_orders'],2)
                            else:
                                self.assertIsNone(row['route_choice'])
                                self.assertEqual(row['selection_reason'],'generated_opening_support')
                    self.assertEqual(anchors,[anchors[0]]*3)

    def test_archived_complete_micro_join_and_tamper(self):
        archive=ev.RUNS/trial.old.FRESH/'pair-042/v2'
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp)
            runner=work/'runner'
            (runner/'distill').mkdir(parents=True)
            (work/'shadow').symlink_to(archive/'shadow')
            (runner/'events.jsonl').symlink_to(archive/'runner/events.jsonl')
            name='distill/episode-0001.json'
            receipt=ev.read_json(archive/'runner'/name)
            policy=dict(policy_sha256={},trial_id='offline',opening_sha256=receipt['opening_sha256'])
            for row in receipt['decisions']:
                acts=row['final_actions']
                plan=dict(actions=[acts[f'u{i}'] for i in range(len(acts))],candidate_id=row['selected_candidate'],plan_ids=row['plan_ids'])
                row.update(parent_plan=plan,alt_plan=plan,route_choice=None,route_anchor=None,trial_id='offline',opening_sha256=policy['opening_sha256'])
            ev.write_json(runner/name,receipt)
            prefix,anchor,choice=trial.trace(work,'v2',policy,None)
            self.assertEqual(len(prefix),len(receipt['decisions']))
            self.assertIsNone(anchor)
            receipt['decisions'][0]['alt_plan']=dict(plan,actions=receipt['decisions'][0]['parent_plan']['actions'],candidate_id=receipt['decisions'][0]['selected_candidate'],plan_ids=['other'])
            (runner/name).write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'same actions, different plans'):
                trial.trace(work,'v2',policy,None)


if __name__=='__main__':
    unittest.main()
