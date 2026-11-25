"""Outcome-blind, offline eligibility checks for new one-step fork rounds."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from decision_tree.zerg.schema import digest
from sc2_skill_adapter.bridge.config import WORKSPACE
from sc2_skill_adapter.bridge.zerg_fork import Fork, require, rows
from sc2_skill_adapter.bridge.zerg_shadow import start_shadow


def excluded_groups(paths, plan, split):
    groups, hashes = set(), set()
    for path in paths:
        prior = json.loads(Path(path).read_text())
        if any(prior.get(k) != plan[k] for k in
               ('kind', 'model_id', 'split_hash', 'source_hash', 'collection_context', 'horizon_steps')):
            raise ValueError(f'excluded plan belongs to a different fit: {path}')
        if not isinstance(prior['pairs'], list) or not prior['pairs']:
            raise ValueError(f'empty or invalid excluded plan: {path}')
        for pair in prior['pairs']:
            g = pair['episode_group']
            ep = split.get(g, {})
            if (g != f"{pair['run_id']}:{pair['episode_id']}" or ep.get('partition') != 'train'
                    or ep['roster'] != pair['roster'] or ep['seed'] != pair['seed']):
                raise ValueError(f'invalid excluded episode {g}: {path}')
            groups.add(g)
        hashes.add(digest(prior))
    return groups, sorted(hashes)


def replay_prefix(fork, controller, events, peek_hash=None):
    """Apply only original prefix observations and accepted micro commands, stopping at B commit."""
    awaiting_feedback = None
    for event in events:
        if event.get('episode') != 1:
            continue
        if event['event'] == 'skill_observation':
            require(awaiting_feedback is None, 'missing prefix step feedback')
            obs = event['payload']
            loop, step = obs['state']['game_loop'], event['step']
            def mask(idx):
                criteria = obs['questions'].get(f'u{idx}', {}).get('criteria', {'0': ''})
                return [str(a) in criteria for a in range(16)]  # 10gen_zerg: 6 basic + 10 targets
            if not controller.active:
                controller.select('adaptive_focus_fire', posture='balanced')
            env = SimpleNamespace(n_agents=10, get_avail_agent_actions=mask)
            actions, summary = fork.step(step, loop, controller, obs, env)
            proposal = (fork.peek_c(controller, summary, obs, env, peek_hash)
                        if step == fork.anchor and peek_hash is not None else None)
            controller.commit_final_actions(actions)
            if step == fork.anchor:
                fork.committed(controller.manager, actions)
                return proposal
            awaiting_feedback = step + 1
        elif event['event'] == 'step':
            require(event['step'] == awaiting_feedback, 'out-of-order prefix step feedback')
            for sub in event['feedback'].get('micro_submissions', []):
                if sub.get('sc2_response') == [1]:
                    controller.manager.record_micro_action(sub['actor'], sub['submitted_action_id'])
            awaiting_feedback = None
    require(False, 'archive ended before B commit')


def verify_archived_b(pair, plan):
    """Raise on an uncommittable archived B. This does not simulate the engine or certify live outcomes."""
    return verify_archived_arm(pair, plan, 'B')


def verify_archived_arm(pair, plan, arm, *, peek=False):
    if peek and arm != 'A':
        raise ValueError('C discovery requires untouched A')
    source = Path(pair['manifest'])
    manifest = json.loads(source.read_text())
    with TemporaryDirectory(prefix='zerg-fork-preflight-') as directory:
        tmp = Path(directory)
        spec = tmp / 'spec.json'
        spec.write_text(json.dumps(dict(pair=pair, arm=arm, source_hash=plan['source_hash'],
                                        collection_context=plan['collection_context'],
                                        **({'rule_hash':plan['rule_hash']} if arm == 'C' else {}))))
        fork = Fork(tmp / 'receipt.json')
        fork.load(spec, manifest)
        old = fork.original_run
        if ((old['method'], old['skill_version'], old['execution_mode'], old['macro_loops'], old['engine_step_loops'],
             old['skill_parameters']['posture'], old['skills']) !=
                ('adaptive-skill', 'tactical_v1', 'micro8-all', 8, 1, 'balanced', 'adaptive_focus_fire')):
            raise ValueError('not a supported archive protocol')
        map_file = Path(old['map_file'])
        metadata = {**old, 'source_hashes': {k: hashlib.sha256(map_file.read_bytes()).hexdigest()
                    for k in old['source_hashes'] if k.endswith('.SC2Map') and (WORKSPACE / k).resolve() == map_file}}
        args = SimpleNamespace(skill_version='tactical_v1', local_policy=None)
        rt, ctrl = start_shadow(str(tmp / 'shadow'), pair['roster'], manifest, old['method'], args, {})
        ctrl.reset()
        ctrl.manager.micro_all_attackers = True
        rt.episode_seed = pair['seed']
        metadata['zerg_shadow'] = {**old['zerg_shadow'], 'source_fingerprint': rt.source_fingerprint,
                                   'execution_context_hash': rt.execution_context_hash}
        fork.check_context(metadata, rt)
        start = fork.started
        fork.start(manifest['episodes'][0], start['initial_state_sha256'], start['game_version'], start['game_loop'])
        return replay_prefix(fork, ctrl, rows(source.parent / 'runner/events.jsonl'),
                             plan['rule_hash'] if peek else None)
