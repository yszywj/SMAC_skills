"""Run the frozen BASE cohort and produce an anchor plan with an idle watchdog.

Reuses the original study's runner flags and candidate menu. New candidates are
executed separately through local_campaign_recover. --dry-run starts no games.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from decision_tree.local import study
from decision_tree.local.runtime import digest, load_policy
from decision_tree.scripts.local_campaign_recover import terminate_owned


def progress_marker(output):
    """During a game, each decision appends to the log; detect missing progress."""
    events = output / 'base/events.jsonl'
    try:
        stat = events.stat()
        return stat.st_size, stat.st_mtime_ns
    except FileNotFoundError:
        return None


def watch(process, command, output, idle_seconds):
    record = {'pid': process.pid, 'command_sha256': digest(command)}
    marker, last_progress = progress_marker(output), time.monotonic()
    while True:
        try:
            return process.wait(timeout=min(idle_seconds, 5))
        except subprocess.TimeoutExpired:
            current = progress_marker(output)
            now = time.monotonic()
            if current != marker:
                marker, last_progress = current, now
                continue
            if now - last_progress < idle_seconds:
                continue
            # Once BASE has written its summary, make_plan audits the entire
            # log and writes inputs for every arm before emitting plan.json.
            # This bounded planning allowance does not apply to a stuck game.
            if (output / 'base/summary.json').is_file() and now - last_progress < max(idle_seconds, 600):
                continue
            terminate_owned(process, command, record)
            raise TimeoutError(f'BASE planner made no progress for {int(now-last_progress)}s')


def plan(template, manifest, policy, output, *, reference_policy=None, anchors=3,
         min_gap=4, max_candidate_games=960, idle_seconds=120, dry_run=False):
    template, output = Path(template).resolve(), Path(output).resolve()
    manifest, policy = Path(manifest).resolve(), Path(policy).resolve()
    artifacts = (study.ROOT / 'decision_tree/artifacts').resolve()
    if not output.is_relative_to(artifacts) or output == template:
        raise ValueError('Choose a separate study output inside repository artifacts')
    if idle_seconds < 30 or idle_seconds > 3600:
        raise ValueError('idle_seconds must be between 30 and 3600')
    frozen = study.read(template / 'protocol.json')
    original = study.read(template / 'manifest.json')
    base_policy = load_policy(template / 'base_policy.json')
    new_manifest = study.read(manifest)
    if (frozen['kind'] != 'local_patch_collection' or digest(original) != frozen['manifest_sha256']
            or digest(base_policy) != frozen['policy_sha256'] or load_policy(policy) != base_policy):
        raise ValueError('Original collection or requested BASE policy binding changed')
    if new_manifest['env_args'] != original['env_args'] or study.source_hashes(new_manifest) != frozen['sources']:
        raise ValueError('Requested scene environment or sources differ from the frozen collection')
    if reference_policy is not None:
        reference_policy = Path(reference_policy).resolve()
        reference = load_policy(reference_policy)
        original_context = study.read(template / 'base/run.json')['local_policy']['context_hash']
        if (reference['selection']['mode'] != 'model' or reference['candidates'] != base_policy['candidates']
                or reference['model']['context_hash'] != original_context):
            raise ValueError('Reference policy does not match the frozen BASE context')
    preflight = study.collect(manifest, policy, output, flags=frozen['flags'], anchors=anchors,
                              anchor_rule='family_coverage', min_gap=min_gap,
                              max_candidate_games=max_candidate_games, workers=1,
                              python=frozen['python'], sc2_path=frozen.get('sc2_path'),
                              reference_policy=reference_policy, dry_run=True)
    flags = output / 'frozen_flags.json'
    command = [frozen['python'], '-B', '-m', 'decision_tree.local.study', 'collect',
               '--manifest', str(manifest), '--policy', str(policy), '--output', str(output),
               '--flags-json', str(flags), '--anchors', str(anchors), '--anchor-rule', 'family_coverage',
               '--min-gap', str(min_gap), '--max-candidate-games', str(max_candidate_games),
               '--python', frozen['python'], '--workers', '1', '--plan-only']
    if reference_policy is not None:
        command.extend(['--reference-policy', str(reference_policy)])
    if frozen.get('sc2_path'):
        command.extend(['--sc2-path', frozen['sc2_path']])
    if dry_run:
        return {**preflight, 'base_only_command': command, 'idle_seconds': idle_seconds}
    output.mkdir(parents=True, exist_ok=True)
    study.save_once(flags, frozen['flags'])
    study.save_once(output / 'base_planner_protocol.json',
                    {'template_protocol_sha256': digest(frozen), 'idle_seconds': idle_seconds, 'command': command})
    attempt = 1
    while (output / f'base_planner_{attempt:03d}.console.log').exists():
        attempt += 1
    with (output / f'base_planner_{attempt:03d}.console.log').open('x', encoding='utf-8') as stream:
        launch = {'cwd': study.ROOT, 'env': {**os.environ, 'PYTHONUNBUFFERED': '1'},
                  'stdin': subprocess.DEVNULL, 'stdout': stream, 'stderr': subprocess.STDOUT}
        if os.name == 'nt':
            launch['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        else:
            launch['start_new_session'] = True
        process = subprocess.Popen(command, **launch)
        try:
            exit_code = watch(process, command, output, idle_seconds)
        except TimeoutError as exc:
            study.save_once(output / f'base_planner_{attempt:03d}.timeout.json',
                            {'pid': process.pid, 'command_sha256': digest(command), 'error': str(exc),
                             'last_events_progress': progress_marker(output)})
            raise
        finally:
            if process.poll() is None:
                terminate_owned(process, command, {'pid': process.pid, 'command_sha256': digest(command)})
    if exit_code:
        raise RuntimeError(f'BASE planner exited with code {exit_code}; see {stream.name}')
    frozen_plan = study.read(output / 'plan.json')
    return {'study': str(output), 'scenes': len(frozen_plan['scenes']), 'anchors': len(frozen_plan['anchors']),
            'candidate_games': len(frozen_plan['arms']), 'started_candidate_games': 0, 'plan_sha256': digest(frozen_plan)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('template', 'manifest', 'policy', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--reference-policy', type=Path)
    parser.add_argument('--anchors', type=int, default=3, choices=(1, 2, 3))
    parser.add_argument('--min-gap', type=int, default=4)
    parser.add_argument('--max-candidate-games', type=int, default=960)
    parser.add_argument('--idle-seconds', type=int, default=120)
    parser.add_argument('--dry-run', action='store_true')
    print(json.dumps(plan(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
