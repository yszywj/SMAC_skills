from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from decision_tree.scripts.linux import eval_r1_r6_r7 as launch


class TestStartupRetry(unittest.TestCase):
    def exercise(self, errors, retries, started=False):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        root = Path(tmp.name); work = root/'pair-001'/'R6'; work.mkdir(parents=True)
        calls = []
        def run(cmd, **kw):
            calls.append(cmd); msg = errors[len(calls)-1]
            if msg:
                kw['stdout'].write(msg)
                if started:
                    (work/'runner').mkdir(); (work/'runner/events.jsonl').write_text('{"event":"episode_started"}\n')
                raise subprocess.CalledProcessError(1, cmd)
        with patch.object(launch.subprocess,'run',side_effect=run), patch.object(launch.time,'sleep') as sleep:
            ok=launch.run_arm(root,work,['sc2'],{},0,retries)
        return ok,len(calls),sleep.call_count,len(list(root.glob('old-attempt-*/pair-001-R6/worker.log')))

    signature='env.reset_record(scenario)\npysc2.lib.remote_controller.ConnectError: Connection rejected.'

    def test_recovers(self):
        self.assertEqual(self.exercise([self.signature,None],3),(True,2,1,1))

    def test_exhaustion_and_disable(self):
        self.assertEqual(self.exercise([self.signature]*3,2),(False,3,2,2))
        self.assertEqual(self.exercise([self.signature],0),(False,1,0,0))

    def test_policy_failure_and_midgame_not_retried(self):
        for msg,started in [('invalid policy',False),(self.signature,True)]:
            with self.assertRaises(subprocess.CalledProcessError):self.exercise([msg],3,started)
