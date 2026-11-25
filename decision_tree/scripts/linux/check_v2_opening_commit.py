"""Offline integration test: replay the old tree until the first changed decision, then stop."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'sc2_skill_adapter')]
from bridge import runner, zerg_shadow
from bridge.zerg_distill import Distill
from bridge.skill_runtime import manager_configuration


def check(folder):
    from decision_tree.zerg import opening_support_v2
    policy=ROOT/'runs/zerg_policy_distill_v1/model.json'
    old=folder/'tree/runner'; meta=json.loads((old/'run.json').read_text()); arch=meta['zerg_shadow']
    manifest=json.loads((folder/'manifest.json').read_text());scene,=manifest['episodes']
    args=runner.parser().parse_args(['run','--manifest','unused','--output-dir','unused','--method','adaptive-skill',
                                     '--skill-version','tactical_v1','--execution-mode','micro8-all'])
    h=hashlib.sha256(Path(opening_support_v2.__file__).read_bytes()).hexdigest()
    with TemporaryDirectory() as tmp, patch.dict(os.environ,{'ZERG_DISTILL_VERSION':'v2','ZERG_DISTILL_OPENING_SHA256':h}):
        root=Path(tmp);rt,ctrl=zerg_shadow.start_shadow(str(root/'shadow'),arch['scene_config_id'],manifest,'adaptive-skill',args,manager_configuration(args)['manager_kwargs'])
        hook=Distill(root,policy,arch);hook.start(1,scene,'archived','4.10.0.75689');ctrl.reset()
        rt.episode_seed=scene['sc2_seed'];ctrl.manager.micro_all_attackers=True
        with (old/'events.jsonl').open() as stream:
            for line in stream:
                e=json.loads(line)
                if e['event']=='skill_observation':
                    pre=e['payload'];loop=pre['state']['game_loop']
                    if loop>82:break
                    def mask(i):
                        q=pre['questions'].get(f'u{i}');ids=set(map(int,q['criteria'])) if q else {0}
                        return [j in ids for j in range(16)]
                    if not ctrl.active:ctrl.select('adaptive_focus_fire',posture='balanced')
                    acts,_=hook.step(e['step'],loop,ctrl,pre,SimpleNamespace(n_agents=10,get_avail_agent_actions=mask))
                    hook.before_commit(ctrl.manager,acts);ctrl.commit_final_actions(acts);hook.committed(ctrl.manager,acts)
                elif e['event']=='step':
                    if acts!=e['actions']:
                        reason=hook.decision['selection_reason']
                        if reason not in ('generated_opening_support','protect_hydra_support_gap'):
                            raise ValueError(f'V1 prefix failed to reproduce: {folder}:{loop}, {reason}')
                        return dict(scene=folder.name,loop=loop,change=reason,actual_manager_commit=True,
                                    old=e['actions'],new=acts,live_engine_confirmation=False)
                    hook.engine(e['feedback'])
                    for sub in e['feedback'].get('micro_submissions',[]):
                        if sub.get('sc2_response')==[1]:ctrl.manager.record_micro_action(sub['actor'],sub['submitted_action_id'])
    return dict(scene=folder.name,change=None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('scenes',type=Path,nargs='+');opts=p.parse_args()
    for s in opts.scenes:print(json.dumps(check(s),allow_nan=False))
