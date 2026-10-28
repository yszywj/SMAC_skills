"""Replay recorded observations: action legality and unchanged off-profile.

This is not a new SC2 match or a win-rate evaluation: future recorded states
were produced by the old policy, not by these new actions.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
ADAPTER = WORK / 'sc2_skill_adapter'
sys.path.insert(0, str(ADAPTER))
from bridge.skill_runtime import load_manager, source_files, LEGACY_COMBAT_PARAMETERS, NEW_COMBAT_PARAMETERS


def verify(race, max_episodes):
    run = ADAPTER / 'runs' / f'cross-20261027-{race}-10v10-v7-n40-seed170-interface-v2'
    metadata = json.loads((run / 'run.json').read_text(encoding='utf-8'))
    parameters = metadata['skill_parameters']
    kwargs = dict(max_steps=parameters['max_steps'], combat_params={
        k: parameters[k] for k in (*LEGACY_COMBAT_PARAMETERS, *NEW_COMBAT_PARAMETERS)},
        unit_value_params=metadata['unit_value_parameters'])
    baseline = load_manager('baseline_v7_20261024')(**kwargs)
    generic = load_manager()(**kwargs)
    specialized = load_manager()(race_profile=race, **kwargs)
    managers = [baseline, generic, specialized]
    counts = Counter()
    current_episode = None
    for line in (run / 'events.jsonl').open(encoding='utf-8'):
        event = json.loads(line)
        if event['event'] != 'skill_observation':
            continue
        if event['episode'] > max_episodes:
            break
        if current_episode != event['episode']:
            for m in managers:m.reset()
            current_episode = event['episode']
            counts['episodes'] += 1
        payload = event['payload'];before = copy.deepcopy(payload)
        outputs = []
        for manager in managers:
            if not manager.active:manager.select('adaptive_focus_fire', posture=parameters['posture'])
            actions, telemetry = manager.step(payload, 10)
            outputs.append(actions)
            for i, action in enumerate(actions):
                actor = f'u{i}'
                if actor in payload['questions']:
                    assert str(action) in payload['questions'][actor]['criteria'], (race,event['episode'],event['step'],actor,action)
                else:assert action == 0
        assert outputs[0] == outputs[1], ('off profile changed', race,event['episode'],event['step'])
        assert payload == before, 'mutated input'
        # An identical observation must not consume another window or budget.
        memory = copy.deepcopy(specialized._race_memory)
        elapsed = specialized.elapsed_steps
        duplicate, _ = specialized.step(payload, 10)
        assert duplicate == outputs[2] and elapsed == specialized.elapsed_steps
        assert memory == specialized._race_memory
        counts['observations'] += 1
        counts['changed_actions_vs_generic'] += sum(a != b for a,b in zip(outputs[1],outputs[2]))
        counts['actions_per_policy'] += 10
    return dict(race=race,**counts)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--episodes', type=int, default=40)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if not 1 <= args.episodes <= 40:cli.error('--episodes must be between 1 and 40')
    hashes = {str(p.relative_to(WORK)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files()}
    results = [verify(race,args.episodes) for race in ('terran','zerg')]
    assert hashes == {str(p.relative_to(WORK)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files()}
    report = dict(kind='offline_recorded_observation_replay_not_new_win_rate', results=results,source_hashes=hashes)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False,indent=2))
