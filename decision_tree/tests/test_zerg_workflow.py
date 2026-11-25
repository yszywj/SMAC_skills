import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from decision_tree.scripts.linux import zerg_workflow as wf


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.work = Path(temp.name)/'work'
        self.fit = Path(temp.name)/'base'; self.fit.mkdir()
        model = dict(target='imitation',scorer='joint_scorer')
        wf.freeze(self.fit/'model.json',model)
        wf.freeze(self.fit/'report.json',dict(model_sha256=wf.digest(model)))
        wf.freeze(self.fit/'split.json',{'one':dict(partition='train')})
        self.run_cli('init','--fit',self.fit)

    def run_cli(self,*args):
        wf.main(['--work',str(self.work),*map(str,args)])

    def test_round_resumes_offline_and_accumulates(self):
        trace = []
        def invoke(name,*args,capture=False):
            fields = list(map(str,args)); trace.append((name,fields))
            out = Path(fields[fields.index('--out')+1]) if '--out' in fields else None
            if name == 'plan_zerg_forks':
                wf.freeze(out,dict(selection=fields[fields.index('--seed')+1]))
            elif name == 'run_zerg_forks':
                out.mkdir(); wf.freeze(out/'plan.json',wf.read(fields[fields.index('--plan')+1]))
            elif name == 'compare_zerg_forks':
                from types import SimpleNamespace
                return SimpleNamespace(stdout=json.dumps({'planned_pairs':24,'accepted_pairs':24}))
            elif name == 'export_zerg_gains':
                out.mkdir(); wf.freeze(out/'envelope.json',dict(experiment_id=Path(fields[1]).name,pairs_trainable=24))
            elif name == 'fit_zerg_gains':
                out.mkdir()
                for fn in ('model','report'):
                    wf.freeze(out/(fn+'.json'),{})
                wf.freeze(out/'pair_facts.json',dict(model_id='test',pairs=[{'partition':'train'}]*24*(fields.index('--out')-1)))
        with patch.object(wf,'call',side_effect=invoke):
            self.run_cli('round',1,'--plan-only')
            self.assertEqual([k for k,_ in trace],['plan_zerg_forks'])
            self.run_cli('round',1)
            self.assertEqual(len(trace),5)
            self.run_cli('round',1)
            self.assertEqual(len(trace),5)
            self.run_cli('round',2)
            self.assertIn('--exclude-plan',trace[5][1]); self.assertEqual(trace[-1][1].index('--out'),3)
            self.assertNotEqual(wf.read(self.work/'round-001/complete.json')['run'],wf.read(self.work/'round-002/complete.json')['run'])

    def test_cannot_skip_round_or_change_frozen_inputs(self):
        with self.assertRaises(SystemExit):
            self.run_cli('round',2)
        first = self.work/'round-001'; first.mkdir()
        wf.freeze(first/'inputs.json',dict(mode='round',fit=str(self.fit),seed=123))
        with self.assertRaises(SystemExit):
            self.run_cli('round',1,'--seed',124)

    def test_adopt_cannot_launch_games_or_refit_baseline(self):
        run = self.work.parent/'pilot'; run.mkdir()
        data = run/'gain_dataset_v1'; data.mkdir()
        prior = run/'prior_fit'; prior.mkdir(); wf.freeze(prior/'model.json',{})
        wf.freeze(run/'plan.json',dict(fit=str(self.fit)))
        wf.freeze(data/'envelope.json',dict(experiment_id=run.name,pairs_trainable=24))
        calls = []
        def offline(name,*args,capture=False):
            calls.append(name)
            if name == 'compare_zerg_forks':
                from types import SimpleNamespace
                return SimpleNamespace(stdout=json.dumps({'accepted_pairs':24,'planned_pairs':24}))
            self.assertEqual(name,'fit_zerg_gains'); self.assertIn('--review-existing',args)
            out = Path(args[args.index('--out')+1]); out.mkdir()
            for fn in ('model','report'):
                wf.freeze(out/(fn+'.json'),{})
            wf.freeze(out/'pair_facts.json',dict(model_id='test',pairs=[{'partition':'train'}]*18+[{'partition':'validation'}]*6))
        with patch.object(wf,'call',side_effect=offline):
            self.run_cli('adopt','--run',run,'--gain-fit',prior)
            self.run_cli('round',1)
            self.assertEqual(calls,['compare_zerg_forks','fit_zerg_gains'])
            with self.assertRaises(SystemExit):
                self.run_cli('init','--collect','--first-round')
